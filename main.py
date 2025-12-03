from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
import os
import re
from typing import List, Optional, Dict
import logging
import math
import random

# Try import faiss, else fallback
try:
    import faiss
    HAS_FAISS = True
except Exception:
    from sklearn.neighbors import NearestNeighbors
    HAS_FAISS = False

# ----- Settings / Paths -----
DATA_FILE = "recipes_with_prices21.csv.gz"
SCALER_FILE = "scaler3.pkl"
MODEL_FILE = "diet_model00.keras"
ENCODED_FILE = "encoded_recipes.npy"  # should be float32 latent vectors

# ----- Globals -----
model = None
recipes_df = pd.DataFrame()
scaler = None
encoded_recipes = None          # full latent matrix (N x D), float32, normalized
resources_loaded = False

# Per-meal precomputed structures
meal_partitions: Dict[str, Dict] = {}  # meal -> {indices, df, faiss_index or nn_model, encoded_matrix}

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

# ----- Config -----
USECOLS = [
    'Calories', 'Keywords', 'Name', 'MealType', 'EstimatedPriceEGP',
    'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'SodiumContent',
    'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent',
    'RecipeIngredientQuantities', 'RecipeIngredientParts'
]

# ANN search params
DEFAULT_TOP_K = 5    # sample from top_k after ranking
ANN_SEARCH_K = 50    # how many neighbours to return from index before filtering

# ----- Startup loading (heavy precompute done once) -----
@app.on_event("startup")
def startup_event():
    global model, recipes_df, scaler, encoded_recipes, resources_loaded, meal_partitions

    logger.info("Starting resource loading and preprocessing...")
    try:
        # --- model & scaler ---
        if not os.path.exists(MODEL_FILE):
            raise FileNotFoundError(f"Model file '{MODEL_FILE}' not found.")
        model = tf.keras.models.load_model(MODEL_FILE, compile=False)
        logger.info("Model loaded")

        if not os.path.exists(SCALER_FILE):
            raise FileNotFoundError(f"Scaler file '{SCALER_FILE}' not found.")
        scaler = joblib.load(SCALER_FILE)
        logger.info("Scaler loaded")

        # --- load dataset (only needed columns) ---
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"Data file '{DATA_FILE}' not found.")
        # usecols to reduce memory; compression gzip
        recipes_df_local = pd.read_csv(DATA_FILE, compression='gzip', usecols=lambda c: c in USECOLS)
        recipes_df_local.columns = recipes_df_local.columns.astype(str)
        logger.info("Dataset loaded with %d rows", len(recipes_df_local))

        # ensure expected columns exist (fill missing)
        for col in USECOLS:
            if col not in recipes_df_local.columns:
                if col in ['Name', 'Keywords', 'MealType', 'RecipeIngredientParts', 'RecipeIngredientQuantities']:
                    recipes_df_local[col] = ""
                else:
                    recipes_df_local[col] = 0.0

        # Normalize text fields once
        text_cols = ['Name', 'Keywords', 'MealType', 'RecipeIngredientParts']
        for c in text_cols:
            recipes_df_local[c] = recipes_df_local[c].astype(str).fillna("")

        # Ensure numeric nutrition columns
        nutrition_columns = [
            'Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent',
            'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 'ProteinContent'
        ]
        for c in nutrition_columns:
            recipes_df_local[c] = pd.to_numeric(recipes_df_local[c], errors='coerce').fillna(0.0)

        # ensure price numeric
        recipes_df_local['EstimatedPriceEGP'] = pd.to_numeric(recipes_df_local['EstimatedPriceEGP'], errors='coerce').fillna(np.inf)

        # store cleaned df globally (immutable after startup)
        recipes_df = recipes_df_local.reset_index(drop=True)

        # --- load or compute encoded_recipes ---
        if os.path.exists(ENCODED_FILE):
            # load with mmap to reduce peak memory
            encoded_recipes = np.load(ENCODED_FILE, mmap_mode='r')
            # If dtype is not float32, convert and save new file
            if encoded_recipes.dtype != np.float32:
                encoded_recipes = encoded_recipes.astype(np.float32)
        else:
            # compute latent encodings once
            scaled = scaler.transform(recipes_df[nutrition_columns])
            enc = model.predict(scaled, batch_size=1024)
            enc = np.asarray(enc, dtype=np.float32)
            # save for subsequent runs
            np.save(ENCODED_FILE, enc)
            encoded_recipes = enc

        # --- normalize encoded vectors to unit norm for cosine via inner product ---
        # convert to float32 and make separate normalized array (in memory)
        enc_mat = np.asarray(encoded_recipes, dtype=np.float32)
        norms = np.linalg.norm(enc_mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        enc_normed = (enc_mat / norms).astype(np.float32)

        # free encoded_recipes reference (we keep enc_normed)
        encoded_recipes = enc_normed  # N x D float32 normalized

        logger.info("Encoded matrix prepared shape=%s dtype=%s", encoded_recipes.shape, encoded_recipes.dtype)

        # --- partition by MealType and build ANN per partition ---
        meal_partitions = {}
        meal_types = recipes_df['MealType'].str.lower().fillna("").unique()
        for mt in meal_types:
            if not mt:
                continue
            mask = recipes_df['MealType'].str.lower() == mt
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue
            enc_sub = encoded_recipes[indices]  # view
            partition = {'indices': indices, 'df': recipes_df.loc[indices].reset_index(drop=True)}
            if HAS_FAISS:
                # use inner product on normalized vectors -> cosine similarity
                d = enc_sub.shape[1]
                index = faiss.IndexFlatIP(d)  # exact index but fast; you can replace with IVF for huge scale
                index.add(enc_sub)
                partition['faiss_index'] = index
                partition['use_faiss'] = True
            else:
                # fallback: build NearestNeighbors with cosine metric brute-force (uses sklearn)
                nn = NearestNeighbors(n_neighbors=min(ANN_SEARCH_K, len(enc_sub)), metric='cosine', algorithm='brute', n_jobs=-1)
                # sklearn's cosine distance; we'll convert to similarity later
                nn.fit(enc_sub)
                partition['nn_model'] = nn
                partition['use_faiss'] = False

            meal_partitions[mt] = partition
            logger.info("Prepared partition '%s' size=%d use_faiss=%s", mt, len(indices), partition['use_faiss'])

        # assign globals
        globals()['recipes_df'] = recipes_df
        globals()['encoded_recipes'] = encoded_recipes
        globals()['meal_partitions'] = meal_partitions
        resources_loaded = True
        logger.info("Startup preprocessing complete. Resources loaded successfully.")

    except Exception as e:
        resources_loaded = False
        logger.exception("Error during startup: %s", e)


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


# ----- Helpers: nutrition vector builder & regex -----
def _build_restriction_pattern(restrictions: List[str]):
    if not restrictions:
        return None
    tokens = [re.escape(t.strip().lower()) for t in restrictions if t.strip()]
    if not tokens:
        return None
    pattern = r"\b(?:" + "|".join(tokens) + r")\b"
    return re.compile(pattern, flags=re.IGNORECASE)


def _build_user_nutrition_vector(target_calories: float, user: UserInput):
    # same heuristic as before — keep consistent with your training preprocessing
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
    vec = np.array([[target_calories, fat, saturated, cholesterol, sodium, carbs, fiber, sugar, protein]], dtype=np.float32)
    return vec


# ----- Core recommendation using ANN per meal partition -----
def _query_partition_for_user(mt: str, user: UserInput, target_calories: float, target_budget: float, dietary_restrictions: List[str], top_k_sample: int = DEFAULT_TOP_K):
    """
    Return a single chosen recipe record (dict) or 'No meal found' dict.
    Steps:
     1) Build user vector -> encode -> normalize
     2) Query ANN within meal partition (faiss or sklearn)
     3) From returned candidate indices, filter by budget/calories/dietary restrictions
     4) Rank by calorie diff + price + similarity; then sample 1 from top_k_sample
    """
    mt_key = mt.lower()
    if mt_key not in meal_partitions:
        return {'Name': 'No meal found', 'MealType': mt, 'Calories': None, 'EstimatedPriceEGP': None, 'RecipeIngredientParts': None, 'RecipeIngredientQuantities': None}

    partition = meal_partitions[mt_key]
    indices = partition['indices']            # indices into global recipes_df
    df_part = partition['df']                # local df copy aligned with indices
    enc_part = encoded_recipes[indices]      # normalized latent vectors for this partition

    # build user latent
    user_vec = _build_user_nutrition_vector(target_calories, user)
    user_scaled = scaler.transform(user_vec)
    user_enc = model.predict(user_scaled)    # shape (1, D)
    # normalize to unit norm
    user_enc = np.asarray(user_enc, dtype=np.float32)
    user_norm = np.linalg.norm(user_enc)
    if user_norm == 0:
        user_norm = 1.0
    user_enc = (user_enc / user_norm).astype(np.float32)

    # Query ANN
    if partition.get('use_faiss', False):
        index = partition['faiss_index']
        # faiss returns inner product scores (cosine since normalized)
        k = min(ANN_SEARCH_K, enc_part.shape[0])
        D, I = index.search(user_enc, k)  # I shape (1,k)
        cand_pos = I[0]                  # positions in enc_part (0..len(enc_part)-1)
        sims = D[0]                      # similarity scores
    else:
        nn = partition['nn_model']
        k = min(ANN_SEARCH_K, enc_part.shape[0])
        dist, cand_pos = nn.kneighbors(user_enc, n_neighbors=k, return_distance=True)
        # sklearn returns distances for cosine: similarity = 1 - dist
        sims = 1.0 - dist[0]
        cand_pos = cand_pos[0]

    if len(cand_pos) == 0:
        return {'Name': 'No meal found', 'MealType': mt, 'Calories': None, 'EstimatedPriceEGP': None, 'RecipeIngredientParts': None, 'RecipeIngredientQuantities': None}

    # Map cand_pos to global indices and df rows quickly
    global_pos = indices[cand_pos]  # global indices into recipes_df
    candidates = recipes_df.loc[global_pos].copy()
    # attach similarity and candidate order
    candidates['_sim'] = sims
    candidates['_global_idx'] = global_pos

    # fast numeric filtering (vectorized)
    candidates['EstimatedPriceEGP'] = pd.to_numeric(candidates['EstimatedPriceEGP'], errors='coerce').fillna(np.inf)
    candidates['Calories'] = pd.to_numeric(candidates['Calories'], errors='coerce').fillna(0)

    # apply budget and calories filter
    candidates = candidates[(candidates['EstimatedPriceEGP'] <= target_budget) & (candidates['Calories'] <= target_calories)]
    if candidates.empty:
        # fallback: return the closest by calorie difference among the original cand set regardless of budget
        fallback = recipes_df.loc[global_pos].copy()
        fallback['Calories'] = pd.to_numeric(fallback['Calories'], errors='coerce').fillna(0)
        fallback['CalorieDiff'] = np.abs(fallback['Calories'] - target_calories)
        fallback_sorted = fallback.sort_values(by='CalorieDiff')
        r = fallback_sorted.iloc[0]
        return {k: r.get(k, None) for k in ['Name', 'MealType', 'Calories', 'EstimatedPriceEGP', 'RecipeIngredientParts', 'RecipeIngredientQuantities']}

    # dietary restrictions via compiled regex
    pattern = _build_restriction_pattern(dietary_restrictions)
    if pattern is not None:
        mask_name = ~candidates['Name'].str.contains(pattern, na=False)
        mask_ing = ~candidates['RecipeIngredientParts'].astype(str).str.contains(pattern, na=False)
        mask_kw = ~candidates['Keywords'].astype(str).str.contains(pattern, na=False)
        candidates = candidates[mask_name & mask_ing & mask_kw]
        if candidates.empty:
            # fallback to previous behavior: return closest by calorie diff among budget-compliant set
            fallback = recipes_df.loc[global_pos].copy()
            fallback['Calories'] = pd.to_numeric(fallback['Calories'], errors='coerce').fillna(0)
            fallback['CalorieDiff'] = np.abs(fallback['Calories'] - target_calories)
            fallback_sorted = fallback.sort_values(by='CalorieDiff')
            r = fallback_sorted.iloc[0]
            return {k: r.get(k, None) for k in ['Name', 'MealType', 'Calories', 'EstimatedPriceEGP', 'RecipeIngredientParts', 'RecipeIngredientQuantities']}

    # ranking: calorie closeness asc, price asc, similarity desc
    candidates['CalorieDiff'] = np.abs(candidates['Calories'] - target_calories)
    candidates = candidates.sort_values(by=['CalorieDiff', 'EstimatedPriceEGP', '_sim'], ascending=[True, True, False])

    # take top_k_sample then pick one randomly among them to vary results per request
    top_k = min(top_k_sample, len(candidates))
    chosen_row = candidates.head(top_k).sample(1).iloc[0]

    return {k: chosen_row.get(k, None) for k in ['Name', 'MealType', 'Calories', 'EstimatedPriceEGP', 'RecipeIngredientParts', 'RecipeIngredientQuantities']}


def suggest_full_day_meal_plan(total_calories: float, daily_budget: float, dietary_restrictions: List[str], user: UserInput):
    plan = {}
    meal_types = ['breakfast', 'snack', 'lunch', 'dinner']
    for meal in meal_types:
        # compute per-meal target and budget here (same as before)
        ratio_map = {'breakfast': 0.20, 'snack': 0.15, 'lunch': 0.35, 'dinner': 0.30}
        cal_ratio = ratio_map.get(meal, 0.25)
        target_cal = total_calories * cal_ratio
        budget_ratio = cal_ratio  # re-use same split for budget
        target_budget = daily_budget * budget_ratio

        rec = _query_partition_for_user(meal, user, target_cal, target_budget, dietary_restrictions, top_k_sample=DEFAULT_TOP_K)
        plan[meal] = [rec]
    return plan


# ----- Routes -----
@app.post("/personalized_recommend")
def personalized_recommendation(user: UserInput):
    if not resources_loaded:
        raise HTTPException(status_code=503, detail="Resources are still loading or failed to load.")

    bmr = 10 * user.weight + 6.25 * user.height - 5 * user.age + (5 if user.gender == 'male' else -161)
    target_calories = int(round(bmr * {
        'sedentary': 1.2, 'lightly_active': 1.375, 'moderately_active': 1.55,
        'very_active': 1.725, 'extra_active': 1.9
    }.get(user.activity_level, 1.2) * {
        'weight_loss': 0.8, 'muscle_gain': 1.2, 'health_maintenance': 1.0
    }.get(user.goal, 1.0)))

    suggestions = suggest_full_day_meal_plan(target_calories, user.daily_budget, user.dietary_restrictions or [], user)
    per_meal_target = round(target_calories / 4)

    return {
        "daily_calories": target_calories,
        "per_meal_target": per_meal_target,
        "suggested_recipes": suggestions
    }


@app.get("/")
def read_root():
    return {"message": "Service is running"}
