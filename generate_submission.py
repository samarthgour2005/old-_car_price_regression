import pandas as pd
import pickle

# 1. Load the Model Package
print("Loading model...")
with open('car_price_model_package.pkl', 'rb') as f:
    package = pickle.load(f)

model = package['model']
feature_names = package['feature_names'] # The column names the model expects

# 2. Get the Test Data (from df_all_encoded which you should have in memory)
# If not, recreate it or assuming you are running this in the same notebook session:
# We need to ensure we select exact same columns in same order
X_test_submission = test_final[feature_names] 

# 3. Predict
print("Predicting...")
predictions = model.predict(X_test_submission)

# 4. Create Submission DataFrame
submission = pd.DataFrame({
    'id': test_raw['id'], # Make sure to use the original IDs
    'price': predictions
})

# 5. Save
output_path = 'submission.csv'
submission.to_csv(output_path, index=False)
print(f"Saved submission to {output_path}")
print(submission.head())
