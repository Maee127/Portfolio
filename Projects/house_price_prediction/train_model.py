# train_model.py - Complete training script with model saving

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor, VotingRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBMeta
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

# Suppress LightGBM warnings
import lightgbm as lgb
#lgb.logging.set_verbosity(-1)

print("="*60)
print("🏠 TRAINING CALIFORNIA HOUSING PRICE MODEL")
print("="*60)

# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================
print("\n📂 Loading data...")
from sklearn.datasets import fetch_california_housing
california = fetch_california_housing(as_frame=True)
X = california.data
y = california.target * 100000  # Convert to dollars

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

# ============================================================
# STEP 2: DEFINE BEST MODELS
# ============================================================
print("\n🔧 Defining best models...")

best_ltgbm = LGBMRegressor(
    n_estimators=500, 
    learning_rate=0.1, 
    random_state=42,
    verbose=-1,           
    silent=True
)

best_xgb = XGBRegressor(
    booster='gbtree', 
    n_estimators=500, 
    learning_rate=0.12, 
    subsample=0.8, 
    colsample_bytree=0.8, 
    alpha=0.01, 
    max_depth=6, 
    random_state=42,
    verbosity=0
)

best_gb = GradientBoostingRegressor(
    n_estimators=500, 
    learning_rate=0.1, 
    max_depth=7,       
    subsample=0.8,
    random_state=42,
    verbose=0
)

# ============================================================
# STEP 3: CREATE FINAL STACKING MODEL
# ============================================================
print("\n🏗️ Creating Stacking Ensemble...")

final_model = StackingRegressor(
    estimators=[
        ('xgb', best_xgb),
        ('gb', best_gb),
        ('lb', best_ltgbm)
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5
)

# ============================================================
# STEP 4: TRAIN THE MODEL
# ============================================================
print("\n📊 Training final model...")
final_model.fit(X_train_scaled, y_train)

# Evaluate
train_score = final_model.score(X_train_scaled, y_train)
test_score = final_model.score(X_test_scaled, y_test)

print(f"\n📈 Model Performance:")
print(f"   Training R²: {train_score:.4f}")
print(f"   Test R²:     {test_score:.4f}")

# ============================================================
# STEP 5: SAVE MODEL AND SCALER
# ============================================================
print("\n💾 Saving model and scaler...")

# Save your final stacking model
joblib.dump(final_model, 'best_model.pkl')

# Save your fitted scaler
joblib.dump(scaler, 'scaler.pkl')

print("✅ Models saved successfully!")
print(f"   - best_model.pkl (Stacking Ensemble, R²={test_score:.4f})")
print(f"   - scaler.pkl (StandardScaler)")

# ============================================================
# STEP 6: VERIFY SAVED MODELS LOAD CORRECTLY
# ============================================================
print("\n🔍 Verifying saved models...")

# Test loading
loaded_model = joblib.load('best_model.pkl')
loaded_scaler = joblib.load('scaler.pkl')

# Make a test prediction
sample = X_test_scaled[0:1]
prediction = loaded_model.predict(sample)

print(f"✅ Models loaded successfully!")
print(f"   Sample prediction: ${prediction[0]:,.0f}")
print(f"   Actual value:      ${y_test.iloc[0]:,.0f}")

print("\n" + "="*60)
print("✅ Training complete! Ready for deployment.")
print("="*60)