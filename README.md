# Diet Recommendation System - FastAPI

This project is a **personalized daily meal recommendation system** built with **FastAPI** and a **TensorFlow Autoencoder model**.  
It provides healthy meal suggestions based on daily caloric needs, dietary goals, budget, and dietary restrictions.

---

## Features

- **One recipe per meal**: Breakfast, Snack, Lunch, Dinner.  
- Recommendations **vary randomly** from the top 5 best-matching recipes to ensure variety.  
- Supports dietary restrictions such as **allergies or vegetarian diets**.  
- Considers the user's daily budget.  
- Uses a **TensorFlow Autoencoder model** to encode recipe nutritional features.  
- Easy to expand with new recipes.

---

## Requirements

- Python >= 3.10
- Python packages:
  - fastapi
  - uvicorn
  - pandas
  - numpy
  - scikit-learn
  - tensorflow
  - joblib

---

## Key Files

- `main.py` : Main FastAPI application.  
- `recipes_with_prices21.csv.gz` : Recipe database with nutrition and pricing info.  
- `scaler3.pkl` : Scaler for normalizing nutritional features.  
- `diet_model00.keras` : Pretrained Autoencoder model.  
- `encoded_recipes.npy` : Latent vectors for recipes, generated on first run.

---

## How to Run?

1. Make sure all files exist: `CSV`, `Scaler`, `Model`.  
2. Install required packages:
```bash
pip install fastapi uvicorn pandas numpy scikit-learn tensorflow joblib
```
3. Start the server:
```bash
uvicorn main:app --reload
```
4. Open your browser like as:
```
http://127.0.0.1:8000/
```
You should see:
```json
{"message": "Service is running "}
```

---

## API Endpoints

### 1. POST `/personalized_recommend`

**Description:** Returns a personalized daily meal plan based on user input.

**Request Body Example:**
```json
{
  "gender": "male",
  "weight": 70,
  "height": 175,
  "age": 30,
  "activity_level": "moderately_active",
  "goal": "weight_loss",
  "daily_budget": 150,
  "dietary_restrictions": ["nuts", "gluten"]
}
```

**Response Example:**
```json
{
  "daily_calories": 2100,
  "per_meal_target": 525,
  "suggested_recipes": {
    "breakfast": [
      {
        "Name": "Oatmeal with Banana",
        "MealType": "breakfast",
        "Calories": 400,
        "EstimatedPriceEGP": 35,
        "RecipeIngredientParts": "Oats, Banana, Milk",
        "RecipeIngredientQuantities": "50g, 1, 200ml"
      }
    ],
    "snack": [
      {
        "Name": "Fruit Salad",
        "MealType": "snack",
        "Calories": 300,
        "EstimatedPriceEGP": 20,
        "RecipeIngredientParts": "Apple, Orange, Grapes",
        "RecipeIngredientQuantities": "1, 1, 50g"
      }
    ],
    "lunch": [
      {
        "Name": "Grilled Chicken with Rice",
        "MealType": "lunch",
        "Calories": 600,
        "EstimatedPriceEGP": 50,
        "RecipeIngredientParts": "Chicken, Rice, Vegetables",
        "RecipeIngredientQuantities": "150g, 100g, 100g"
      }
    ],
    "dinner": [
      {
        "Name": "Vegetable Stir Fry",
        "MealType": "dinner",
        "Calories": 500,
        "EstimatedPriceEGP": 45,
        "RecipeIngredientParts": "Broccoli, Carrot, Bell Pepper",
        "RecipeIngredientQuantities": "100g, 50g, 50g"
      }
    ]
  }
}
```

---

### 2. GET `/`

**Description:** Simple health check endpoint.  
**Response:**
```json
{"message": "Service is running "}
```

---

## Notes

- Each meal selects **one recipe randomly** from the top 15 best-matching recipes to ensure **variety**.  
- If no suitable meal is found that meets calories or budget, `"No meal found"` is returned.  
- You can modify the calorie distribution per meal in `main.py` by editing the `meal_split` dictionary.

---

## Extending the System

- Add more meals or extra snacks.  
- Support more dietary restrictions (e.g., **low-sodium**, **vegan**).  
- Save user history to provide daily evolving recommendations.
