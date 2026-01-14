import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==========================================
# 1. Define One Robust Preprocessing Function
# ==========================================
def preprocess_data(df, is_train=True):
    df = df.copy()
    
    # Text Extraction Helper
    def extract_pattern(text, pattern, type_func=float):
        match = re.search(pattern, str(text))
        return type_func(match.group(1)) if match else None

    # Feature Engineering
    df['hp'] = df['engine'].apply(lambda x: extract_pattern(x, r'(\d+\.?\d*)HP', float))
    df['liters'] = df['engine'].apply(lambda x: extract_pattern(x, r'(\d+\.?\d*)L', float))
    df['cylinders'] = df['engine'].apply(lambda x: extract_pattern(x, r'(\d+)\s+Cylinder', int))
    df['trans_speed'] = df['transmission'].apply(lambda x: extract_pattern(x, r'(\d+)-Speed', int) or 0)

    # Impute Missing Values (Hardcoded for consistency between Train/Test)
    # in a real prod system, you'd learn these from train and apply to test
    df['hp'] = df['hp'].fillna(250.0)
    df['liters'] = df['liters'].fillna(3.0)
    df['cylinders'] = df['cylinders'].fillna(6)
    
    # Binary Features
    df['accident_clean'] = df['accident'].apply(lambda x: 1 if x == 'At least 1 accident or damage reported' else 0)

    # Drop Columns
    drop_cols = ['id', 'clean_title', 'model', 'engine', 'transmission', 'ext_col', 'int_col', 'accident']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    return df

# ==========================================
# 2. Load Data
# ==========================================
df_train_raw = pd.read_csv(r"C:\Users\samar\car_price\old-_car_price_regression\data\train.csv")
df_test_raw = pd.read_csv(r"C:\Users\samar\car_price\old-_car_price_regression\data\test.csv")

# ==========================================
# 3. Preprocess & Encode
# ==========================================
# Apply base cleaning
train_clean = preprocess_data(df_train_raw, is_train=True)
test_clean = preprocess_data(df_test_raw, is_train=False)

# Separate Target
X = train_clean.drop(columns=['price'])
y = train_clean['price']
X_test_input = test_clean # No price column here

# One-Hot Encoding (Apply to both to handle categories)
# Align columns by using pd.get_dummies on combined data OR reindexing
# Strategy: Reindex Test to match Train
X_encoded = pd.get_dummies(X, columns=['brand', 'fuel_type'], drop_first=True)
X_test_encoded = pd.get_dummies(X_test_input, columns=['brand', 'fuel_type'], drop_first=True)

# FORCE ALIGNMENT: Ensure Test has exactly the same columns as Train
# 1. Add missing columns to Test (fill 0)
# 2. Remove extra columns from Test (that weren't in Train)
# 3. Reorder to match Train
X_test_final = X_test_encoded.reindex(columns=X_encoded.columns, fill_value=0)

print(f"Train Shape: {X_encoded.shape}")
print(f"Test Shape:  {X_test_final.shape}")

# Deviation Check
assert list(X_encoded.columns) == list(X_test_final.columns), "Columns do not match!"

# ==========================================
# 4. Train Model (Must Retrain!)
# ==========================================
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("Training model...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# ==========================================
# 5. Predict on Test
# ==========================================
predictions = model.predict(X_test_final)

# Create Submission
submission = pd.DataFrame({
    'id': df_test_raw['id'],
    'price': predictions
})

print("\nSample Predictions:")
print(submission.head())

# Optional: View Deviation
mae = mean_absolute_error(y_val, model.predict(X_val))
print(f"\nValidation MAE: {mae:,.0f}")
