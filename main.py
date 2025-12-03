from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import os
import re
from typing import List, Optional
import logging

# ----- Settings / Paths -----
DATA_FILE = "recipes_with_prices21.csv.gz"
SCALER_FILE = "scaler3.pkl"
MODEL_FILE = "diet_model00.keras"
ENCODED_FILE = "encoded_recipes.npy"

# ----- Globals -----
model = None
recipes_df = pd.DataFrame()
scaler = None
encoded_recipes = None
resources_loaded = False

# ----- FastAPI App -----
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("diet_recommender")

# ----- Startup loading -----
@app.on_event("startup")
def startup_event():
    global model, recipes_df, scaler, encoded_recipes, resources_loaded

    logger.info("Starting resource loading...")
    try:
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found.")
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        logger.info("Model loaded")

        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Data file '{DATA_FILE}' not found.")
        recipes_df = pd.read_csv(DATA_FILE, compression='gzip')
        recipes_df.columns = recipes_df.columns.astype(str)

        expected_columns = [
            'Calories', 'Keywords', 'Name', 'MealType', 'EstimatedPriceEGP',
            'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
            'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent',
            'RecipeIngredientQuantities', 'RecipeIngredientParts'
        ]
        for col in expected_columns:
            if col not in recipes_df.columns:
                if col in ['Name', 'Keywords', 'MealType', 'RecipeIngredientParts', 'RecipeIngredientQuantities']:
                    recipes_df[col] = ""
                else:
                    recipes_df[col] = 0.0

        text_cols = ['Name', 'Keywords', 'MealType', 'RecipeIngredientParts']
        for c in text_cols:
            recipes_df[c] = recipes_df[c].astype(str).fillna("")

        if not os.path.exists(SCALER_FILE):
            raise FileNotFoundError(f"Scaler file '{SCALER_FILE}' not found.")
        scaler = joblib.load(SCALER_FILE)
        logger.info("Scaler loaded")

        if os.path.exists(ENCODED_FILE):
            encoded_recipes = np.load(ENCODED_FILE)
            logger.info("Encoded recipes loaded from file")
        else:
            nutrition_columns = [
                'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
                'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
            ]
            for c in nutrition_columns:
                recipes_df[c] = pd.to_numeric(recipes_df[c], errors='coerce').fillna(0.0)

            scaled_data = scaler.transform(recipes_df[nutrition_columns])
            encoded_recipes = model.predict(scaled_data)
            np.save(ENCODED_FILE, encoded_recipes)
            logger.info("Encoded recipes computed and saved")

        resources_loaded = True
        logger.info("Resources loaded successfully")

    except Exception as e:
        resources_loaded = False
        logger.exception("Error while loading resources: %s", e)


# ----- Request model -----
class UserInput(BaseModel):
    gender: str
    weight: float
    height: float
    age: int
    activity_level: str
    goal: str
    daily_budget: float = Field(..., ge=0)
    dietary_restrictions: Optional[List[str]] = Field(default_factory=list)

    @validator('gender')
    def gender_must_be_valid(cls, v):
        if v.lower() not in ('male', 'female'):
            raise ValueError("gender must be 'male' or 'female'")
        return v.lower()

    @validator('activity_level')
    def activity_must_be_valid(cls, v):
        allowed = {'sedentary','lightly_active','moderately_active','very_active','extra_active'}
        if v not in allowed:
            raise ValueError(f"activity_level must be one of {allowed}")
        return v

    @validator('goal')
    def goal_must_be_valid(cls, v):
        allowed = {'weight_loss','muscle_gain','health_maintenance'}
        if v not in allowed:
            raise ValueError(f"goal must be one of {allowed}")
        return v


# ----- Helpers -----
def compute_bmr(gender: str, weight: float, height: float, age: int) -> float:
    if gender == 'male':
        return 10 * weight + 6.25 * height - 5 * age + 5
    return 10 * weight + 6.25 * height - 5 * age - 161


def compute_daily_caloric_intake(bmr: float, activity_level: str, goal: str) -> int:
    intensity_multipliers = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extra_active': 1.9
    }
    objective_adjustments = {
        'weight_loss': 0.8,
        'muscle_gain': 1.2,
        'health_maintenance': 1.0
    }
    return int(round(bmr * intensity_multipliers.get(activity_level, 1.2) * objective_adjustments.get(goal, 1.0)))


def _build_restriction_pattern(restrictions: List[str]):
    if not restrictions:
        return None
    tokens = [re.escape(t.strip().lower()) for t in restrictions if t.strip()]
    if not tokens:
        return None
    pattern = r"\b(?:" + "|".join(tokens) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def _build_user_nutrition_vector(target_calories: float, user: UserInput):
    weight_kg = max(30.0, user.weight)
    if user.goal == "muscle_gain":
        protein_g_per_kg = 1.8
    elif user.goal == "weight_loss":
        protein_g_per_kg = 1.4
    else:
        protein_g_per_kg = 1.2
    protein = weight_kg * protein_g_per_kg

    fat_cal = target_calories * 0.25
    fat = fat_cal / 9.0
    protein_cal = protein * 4.0
    remaining_cal = max(0.0, target_calories - (protein_cal + fat_cal))
    carbs = remaining_cal / 4.0
    saturated = fat * 0.3
    sugar = carbs * 0.1
    cholesterol = 50.0
    sodium = 300.0
    fiber = max(1.0, carbs * 0.05)

    vec = np.array([[target_calories, fat, saturated, cholesterol, sodium, carbs, fiber, sugar, protein]])
    return vec


# ----- Recommendation logic -----
def suggest_recipes(total_calories: float, meal_type: str, daily_budget: float, dietary_restrictions: List[str], user: UserInput, top_n: int = 1):
    meal_split = {
        'breakfast': (0.20, 0.20),
        'snack':     (0.15, 0.15),
        'lunch':     (0.35, 0.35),
        'dinner':    (0.30, 0.30)
    }
    cal_ratio, budget_ratio = meal_split.get(meal_type.lower(), (0.25, 0.25))
    target_calories = float(total_calories) * cal_ratio
    target_budget = float(daily_budget) * budget_ratio

    nutrition_vector = _build_user_nutrition_vector(target_calories, user)
    try:
        scaled_vec = scaler.transform(nutrition_vector)
        predicted_latent = model.predict(scaled_vec)
    except Exception as e:
        logger.exception("Prediction error: %s", e)
        raise HTTPException(status_code=500, detail="Internal model prediction failed")

    sims = cosine_similarity(predicted_latent, encoded_recipes)[0]
    top_idx = np.argsort(sims)[::-1]

    candidates = recipes_df.iloc[top_idx].copy()
    candidates['MealType'] = candidates['MealType'].astype(str).str.lower()
    candidates['EstimatedPriceEGP'] = pd.to_numeric(candidates['EstimatedPriceEGP'], errors='coerce').fillna(np.inf)
    candidates['Calories'] = pd.to_numeric(candidates['Calories'], errors='coerce').fillna(0)
    candidates = candidates[(candidates['MealType'] == meal_type.lower()) &
                            (candidates['EstimatedPriceEGP'] <= target_budget) &
                            (candidates['Calories'] <= target_calories)]

    pattern = _build_restriction_pattern(dietary_restrictions)
    if pattern is not None:
        mask_name = ~candidates['Name'].str.contains(pattern, na=False)
        mask_ing = ~candidates['RecipeIngredientParts'].astype(str).str.contains(pattern, na=False)
        mask_kw = ~candidates['Keywords'].astype(str).str.contains(pattern, na=False)
        candidates = candidates[mask_name & mask_ing & mask_kw]

    if candidates.empty:
        return pd.DataFrame([{
            'Name': 'No meal found',
            'MealType': meal_type,
            'Calories': None,
            'EstimatedPriceEGP': None,
            'RecipeIngredientParts': None,
            'RecipeIngredientQuantities': None
        }])

    candidates['CalorieDiff'] = np.abs(candidates['Calories'] - target_calories)
    candidates['Similarity'] = sims[top_idx][:len(candidates)]
    ranked = candidates.sort_values(by=['CalorieDiff', 'EstimatedPriceEGP', 'Similarity'], ascending=[True, True, False])

    top_k = min(15, len(ranked))
    selected = ranked.head(top_k).sample(1)

    cols = ['Name', 'MealType', 'Calories', 'EstimatedPriceEGP', 'RecipeIngredientParts', 'RecipeIngredientQuantities']
    return selected[cols]


def suggest_full_day_meal_plan(total_calories: float, daily_budget: float, dietary_restrictions: List[str], user: UserInput):
    plan = {}
    for meal in ['breakfast', 'snack', 'lunch', 'dinner']:
        recipes = suggest_recipes(total_calories, meal, daily_budget, dietary_restrictions, user, top_n=1)
        plan[meal] = recipes.reset_index(drop=True).to_dict(orient="records")
    return plan


# ----- Routes -----
@app.post("/personalized_recommend")
def personalized_recommendation(user: UserInput):
    if not resources_loaded:
        raise HTTPException(status_code=503, detail="Resources are still loading or failed to load.")

    bmr = compute_bmr(user.gender, user.weight, user.height, user.age)
    target_calories = compute_daily_caloric_intake(bmr, user.activity_level, user.goal)
    suggestions = suggest_full_day_meal_plan(target_calories, user.daily_budget, user.dietary_restrictions, user)
    per_meal_target = round(target_calories / 4)

    return {
        "daily_calories": target_calories,
        "per_meal_target": per_meal_target,
        "suggested_recipes": suggestions
    }


@app.get("/")
def read_root():
    return {"message": "Service is running "}
