import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# 1. Load Data
data_path = r"C:\Users\samar\car_price\old-_car_price_regression\data\train.csv"
df = pd.read_csv(data_path)

print(f"Original Shape: {df.shape}")

# 2. Feature Engineering & Cleaning

# Function to extract horsepower
def extract_hp(text):
    match = re.search(r'(\d+\.?\d*)HP', str(text))
    return float(match.group(1)) if match else None

# Function to extract displacement (Liters)
def extract_liters(text):
    match = re.search(r'(\d+\.?\d*)L', str(text))
    return float(match.group(1)) if match else None

# Function to extract cylinders
def extract_cylinders(text):
    match = re.search(r'(\d+)\s+Cylinder', str(text)) 
    if match:
        return int(match.group(1))
    # Fallback for V6, V8 notation
    if 'V6' in str(text): return 6
    if 'V8' in str(text): return 8
    return None

# Apply Extraction
df['hp'] = df['engine'].apply(extract_hp)
df['liters'] = df['engine'].apply(extract_liters)
df['cylinders'] = df['engine'].apply(extract_cylinders)

# Transmission: Extract speed
def extract_speed(text):
    match = re.search(r'(\d+)-Speed', str(text))
    return int(match.group(1)) if match else 0 # 0 for unknown or CVT/Manual unspecified

df['trans_speed'] = df['transmission'].apply(extract_speed)

# Handling Missing Values (Simple Strategy)
# Numerical: Fill with Median
for col in ['hp', 'liters', 'cylinders', 'trans_speed']:
    df[col] = df[col].fillna(df[col].median())

# Binary/Categorical Mapping
df['accident_clean'] = df['accident'].apply(lambda x: 1 if x == 'At least 1 accident or damage reported' else 0)

# 3. Drop Unused & High Cardinality Columns
# Dropping 'id', 'clean_title' (low info), 'model' (high card), original text columns
drop_cols = ['id', 'clean_title', 'model', 'engine', 'transmission', 'ext_col', 'int_col', 'accident']
df_processed = df.drop(columns=drop_cols)

# 4. Encoding Categoricals (One-Hot)
# 'brand', 'fuel_type'
df_processed = pd.get_dummies(df_processed, columns=['brand', 'fuel_type'], drop_first=True)

print(f"Processed Shape: {df_processed.shape}")

# 5. Train/Test Split
target = 'price'
X = df_processed.drop(columns=[target])
y = df_processed[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Model Training (XGBoost)
print("Training XGBoost Model...")
model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)

# 7. Evaluation
predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n--- Model Performance ---")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"R2:   {r2:.4f}")

# Feature Importance
print("\n--- Top 10 Feature Importances ---")
importances = pd.Series(model.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(10))

# ==========================================
# 1. Define Preprocessing Function
# ==========================================
def preprocess_data(df, is_train=True):
    df = df.copy()
    
    # Feature Extraction
    df['hp'] = df['engine'].apply(lambda x: float(re.search(r'(\d+\.?\d*)HP', str(x)).group(1)) if re.search(r'(\d+\.?\d*)HP', str(x)) else None)
    df['liters'] = df['engine'].apply(lambda x: float(re.search(r'(\d+\.?\d*)L', str(x)).group(1)) if re.search(r'(\d+\.?\d*)L', str(x)) else None)
    df['cylinders'] = df['engine'].apply(lambda x: int(re.search(r'(\d+)\s+Cylinder', str(x)).group(1)) if re.search(r'(\d+)\s+Cylinder', str(x)) else None)
    df['trans_speed'] = df['transmission'].apply(lambda x: int(re.search(r'(\d+)-Speed', str(x)).group(1)) if re.search(r'(\d+)-Speed', str(x)) else 0)

    # Fill Missing (using hardcoded values to avoid data leakage between train/test if simpler)
    # Ideally, compute medians from Train and apply to Test. Using rough operational defaults here:
    df['hp'] = df['hp'].fillna(df['hp'].median())
    df['liters'] = df['liters'].fillna(df['liters'].median())
    df['cylinders'] = df['cylinders'].fillna(6) # Median approx
    df['trans_speed'] = df['trans_speed'].fillna(0) # 0 for unknown
    
    # Accident
    df['accident_clean'] = df['accident'].apply(lambda x: 1 if x == 'At least 1 accident or damage reported' else 0)

    # Drop unused
    drop_cols = ['id', 'clean_title', 'model', 'engine', 'transmission', 'ext_col', 'int_col', 'accident']
    # Only drop if they exist
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    # Target encoding or Drop price if in test (it won't be, but good practice)
    if 'price' in df.columns and not is_train:
        df = df.drop(columns=['price'])
        
    return df

# ==========================================
# 2. Process Both Dataframes
# ==========================================
# Assuming df_train is already loaded
df_test = pd.read_csv(r"C:\Users\samar\car_price\old-_car_price_regression\data\test.csv")

# Apply cleaning
train_clean = preprocess_data(df_train, is_train=True)
test_clean = preprocess_data(df_test, is_train=False)

# One-Hot Encoding
train_encoded = pd.get_dummies(train_clean, columns=['brand', 'fuel_type'], drop_first=True)
test_encoded = pd.get_dummies(test_clean, columns=['brand', 'fuel_type'], drop_first=True)

# ==========================================
# 3. Align Columns (CRITICAL STEP)
# ==========================================
# Get columns from trained X (excluding target)
target = 'price'
train_cols = train_encoded.drop(columns=[target]).columns

# Reindex test to match train columns, filling missing with 0
X_real_test = test_encoded.reindex(columns=train_cols, fill_value=0)

# ==========================================
# 4. Predict
# ==========================================
final_predictions = model.predict(X_real_test)

# Verify
print(f"Predictions shape: {final_predictions.shape}")
print(final_predictions[:5])

# Save if needed
submission = pd.DataFrame({'id': df_test['id'], 'price': final_predictions})
# submission.to_csv('submission.csv', index=False)