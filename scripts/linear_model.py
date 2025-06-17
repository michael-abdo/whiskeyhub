import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
print("📊 LINEAR REGRESSION MODEL FOR WHISKEY RATINGS")
print("=" * 50)
print("\n📂 Loading data...")
df = pd.read_csv("../results/full_joined.csv")

# Check available features
print("\n🔍 Checking available features...")
print(f"Total columns: {len(df.columns)}")
print(f"Sample columns: {list(df.columns)[:20]}...")

# Define potential features based on what might be in the whiskey data
potential_features = ['proof', 'price', 'age', 'viscosity', 'complexity', 
                     'profiness', 'viscocity', 'finish_duration']

# Find which features are actually available
available_features = []
for feature in potential_features:
    matching_cols = [col for col in df.columns if feature.lower() in col.lower()]
    if matching_cols:
        available_features.append(matching_cols[0])

print(f"\n✅ Available features: {available_features}")

# Prepare data for modeling
# First, aggregate ratings by whiskey (if multiple ratings per whiskey)
if 'rating' in df.columns and 'whiskey_id' in df.columns:
    # Group by whiskey and take mean rating
    whiskey_data = df.groupby('whiskey_id').agg({
        'rating': 'mean',
        **{feat: 'first' for feat in available_features if feat in df.columns}
    }).reset_index()
    
    print(f"\n📊 Aggregated to {len(whiskey_data)} unique whiskeys")
else:
    print("\n⚠️ No 'rating' column found. Cannot proceed with modeling.")
    exit()

# Clean data - remove rows with missing values in features or rating
features_to_use = [f for f in available_features if f in whiskey_data.columns]
print(f"\n🔧 Using features: {features_to_use}")

if len(features_to_use) == 0:
    print("\n❌ No valid features found. Cannot build model.")
    exit()

# Prepare feature matrix and target
whiskey_data_clean = whiskey_data.dropna(subset=features_to_use + ['rating'])
print(f"\n🧹 Clean data: {len(whiskey_data_clean)} samples after removing missing values")

X = whiskey_data_clean[features_to_use]
y = whiskey_data_clean['rating']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n📊 Train set: {len(X_train)} samples")
print(f"📊 Test set: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
print("\n🚀 Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Evaluate model
print("\n📈 MODEL EVALUATION:")
print("-" * 30)

# Training metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
train_mae = mean_absolute_error(y_train, y_pred_train)
train_r2 = r2_score(y_train, y_pred_train)

print(f"Training Set:")
print(f"  - RMSE: {train_rmse:.3f}")
print(f"  - MAE: {train_mae:.3f}")
print(f"  - R²: {train_r2:.3f}")

# Test metrics
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nTest Set:")
print(f"  - RMSE: {test_rmse:.3f}")
print(f"  - MAE: {test_mae:.3f}")
print(f"  - R²: {test_r2:.3f}")

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                           scoring='neg_mean_squared_error')
cv_rmse = np.sqrt(-cv_scores.mean())
print(f"\n5-Fold Cross-Validation RMSE: {cv_rmse:.3f} (+/- {np.sqrt(cv_scores.std()):.3f})")

# Feature importance
print("\n🎯 FEATURE IMPORTANCE:")
print("-" * 30)
feature_importance = pd.DataFrame({
    'feature': features_to_use,
    'coefficient': model.coef_,
    'abs_coefficient': np.abs(model.coef_)
}).sort_values('abs_coefficient', ascending=False)

for _, row in feature_importance.iterrows():
    print(f"{row['feature']:20s}: {row['coefficient']:+.3f}")

# Create visualizations
print("\n📊 Creating visualizations...")

# 1. Actual vs Predicted scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title('Linear Regression: Actual vs Predicted Ratings')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/linear_model_predictions.png')
plt.close()

# 2. Feature importance bar plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['abs_coefficient'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance in Linear Regression Model')
plt.tight_layout()
plt.savefig('../results/linear_model_feature_importance.png')
plt.close()

# 3. Residuals plot
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Rating')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../results/linear_model_residuals.png')
plt.close()

# Save results
results = {
    'model_type': 'Linear Regression',
    'features_used': features_to_use,
    'n_train_samples': len(X_train),
    'n_test_samples': len(X_test),
    'train_rmse': train_rmse,
    'train_mae': train_mae,
    'train_r2': train_r2,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2,
    'cv_rmse': cv_rmse,
    'feature_coefficients': dict(zip(features_to_use, model.coef_))
}

# Save results to file
with open('../results/linear_model_results.txt', 'w') as f:
    f.write("LINEAR REGRESSION MODEL RESULTS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model: {results['model_type']}\n")
    f.write(f"Features: {', '.join(results['features_used'])}\n")
    f.write(f"Training samples: {results['n_train_samples']}\n")
    f.write(f"Test samples: {results['n_test_samples']}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"Train RMSE: {results['train_rmse']:.3f}\n")
    f.write(f"Train R²: {results['train_r2']:.3f}\n")
    f.write(f"Test RMSE: {results['test_rmse']:.3f}\n")
    f.write(f"Test R²: {results['test_r2']:.3f}\n")
    f.write(f"CV RMSE: {results['cv_rmse']:.3f}\n\n")
    f.write("Feature Coefficients:\n")
    for feat, coef in results['feature_coefficients'].items():
        f.write(f"  {feat}: {coef:.3f}\n")

print("\n✅ Results saved to:")
print("  - linear_model_results.txt")
print("  - linear_model_predictions.png")
print("  - linear_model_feature_importance.png")
print("  - linear_model_residuals.png")