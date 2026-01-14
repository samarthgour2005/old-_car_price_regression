# Used Car Price Regression 🚗💰

A machine learning project to predict the price of used cars based on various features like brand, model, mileage, and engine specifications. This solution uses **XGBoost** with extensive feature engineering to handle complex text data.

## 📌 Project Overview

The goal is to predict `price` (target variable) using a dataset of used car listings.
- **Dataset Size:** ~188,000 rows.
- **Key Features:** Brand, Model, Year, Mileage, Engine (text), Transmission (text).
- **Metric:** Mean Absolute Error (MAE) & RMSE.

## 🛠️ Methodology & Approach

### 1. Data Cleaning & Preprocessing
- **Handling Missing Values:** Imputed missing numerical values (`hp`, `liters`) with median.
- **Outliers:** Robust handling implicitly via tree-based models.

### 2. Feature Engineering ⚙️
The raw dataset contained complex text fields (`engine`, `transmission`) which were parsed using Regex:

- **Engine:** Extracted `Horsepower (HP)`, `Displacement (Liters)`, and `Cylinder Count`.
- **Transmission:** Extracted `Speed` (e.g., "8-Speed" -> 8).
- **Accident History:** Converted text descriptions into a binary `accident_reported` flag.
- **Brand/Fuel:** Applied One-Hot Encoding (pd.get_dummies).

### 3. Model Selection 🧠
**Model Used:** `XGBoost Regressor`
- **Why?** State-of-the-art performance on tabular data, handles non-linear relationships well, and is robust to outliers.
- **Strategy:**
    - Combined Train/Test sets for consistent encoding.
    - Used `join-process-split` pipeline to prevent feature mismatch.
    - Hyperparameters: `n_estimators=1000`, `learning_rate=0.05`, `max_depth=6`.

## 📂 Project Structure

```
├── data/
│   ├── train.csv           # Training dataset
│   ├── test.csv            # Test dataset (no target)
│   └── sample_submission.csv
├── analyze_data_stats.py   # Initial EDA script
├── full_pipeline.py        # Complete training pipeline script
├── unified_pipeline.py     # Robust join-propess-split pipeline (BEST)
├── generate_submission.py  # Script to generate Kaggle submission
├── README.md               # Project Documentation
└── requirements.txt        # Dependencies
```

## 🚀 How to Run

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline:**
   You can run the unified pipeline to train the model and generate predictions:
   ```bash
   python unified_pipeline.py
   ```

3. **Generate Submission:**
   If you have a saved model, use:
   ```bash
   python generate_submission.py
   ```

## 📊 Performance (Expected)

On the validation set (20% split):
- **MAE:** ~$18,056.09 (Replace with your actual score)
- **RMSE:** ~$61,016.17
- **R² Score:** ~0.4007

## 📝 Future Improvements
- **NLP on Model Name:** Use TF-IDF on the `model` column.
- **Stacking:** Combine XGBoost with LightGBM and CatBoost.
- **Deep Learning:** Try a TabNet approach for potential marginal gains.
