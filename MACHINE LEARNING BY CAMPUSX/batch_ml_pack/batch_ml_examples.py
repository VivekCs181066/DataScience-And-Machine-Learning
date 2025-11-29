# batch_ml_examples.py
# Batch Machine Learning examples with scikit-learn
# Run: python batch_ml_examples.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os

np.random.seed(42)

OUTDIR = "outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------
# Example 1 — Batch Regression
# ---------------------------
# Synthetic house-price style data
n = 2000
X = np.random.rand(n, 3)
# features: [sqft_norm, bedrooms_norm, location_score]
sqft = 400 + X[:,0]*2600                # 400—3000 sqft
bedrooms = (X[:,1]*4 + 1).round()       # 1—5 bedrooms
location = X[:,2]*10                    # 0—10 location score
price = 30000 + sqft*150 + bedrooms*50000 + location*20000 + np.random.randn(n)*20000
df_reg = pd.DataFrame({
    "sqft": sqft,
    "bedrooms": bedrooms,
    "location_score": location,
    "price": price
})

Xr = df_reg[["sqft","bedrooms","location_score"]].values
yr = df_reg["price"].values
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(Xr_train, yr_train)  # <--- Batch training in one go
yr_pred = reg.predict(Xr_test)

mse = mean_squared_error(yr_test, yr_pred)
r2 = r2_score(yr_test, yr_pred)

print("\n=== Example 1: Batch Regression ===")
print("MSE:", round(mse, 2))
print("R^2:", round(r2, 4))
dump(reg, os.path.join(OUTDIR, "regression_model.joblib"))

# ------------------------------
# Example 2 — Batch Classification
# ------------------------------
# Synthetic churn-like data
m = 3000
Xp = np.random.randn(m, 4)
# features: [tenure_months, monthly_spend, support_calls, engagement_score]
tenure = np.abs(Xp[:,0])*36
spend = np.abs(Xp[:,1])*50 + 20
calls = np.abs(Xp[:,2])*3
engage = np.clip(np.abs(Xp[:,3])*5, 0, 10)
# True rule: high spend + low engagement -> more churn
logit = -0.03*tenure + 0.02*spend + 0.15*calls - 0.25*engage
prob = 1/(1+np.exp(-logit))
y = (prob > 0.5).astype(int)

df_cls = pd.DataFrame({
    "tenure": tenure,
    "spend": spend,
    "calls": calls,
    "engagement": engage,
    "churn": y
})

Xc = df_cls[["tenure","spend","calls","engagement"]].values
yc = df_cls["churn"].values

# scale helpful for LogisticRegression
scaler = StandardScaler()
Xc_scaled = scaler.fit_transform(Xc)

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc_scaled, yc, test_size=0.25, random_state=42)

clf = LogisticRegression(max_iter=200)
clf.fit(Xc_train, yc_train)  # <--- Batch training in one go
yc_pred = clf.predict(Xc_test)

acc = accuracy_score(yc_test, yc_pred)
prec = precision_score(yc_test, yc_pred)
rec = recall_score(yc_test, yc_pred)
cm = confusion_matrix(yc_test, yc_pred)

print("\n=== Example 2: Batch Classification ===")
print("Accuracy:", round(acc, 4))
print("Precision:", round(prec, 4))
print("Recall:", round(rec, 4))
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", classification_report(yc_test, yc_pred))

dump(clf, os.path.join(OUTDIR, "churn_model.joblib"))
dump(scaler, os.path.join(OUTDIR, "churn_scaler.joblib"))

# --------------------------------
# Example 3 — Periodic Retraining
# --------------------------------
# Simulate "Month 2" new data and retrain from scratch (batch)
m2 = 1500
Xp2 = np.random.randn(m2, 4)
tenure2 = np.abs(Xp2[:,0])*36
spend2 = np.abs(Xp2[:,1])*50 + 20
calls2 = np.abs(Xp2[:,2])*3
engage2 = np.clip(np.abs(Xp2[:,3])*5, 0, 10)
logit2 = -0.03*tenure2 + 0.02*spend2 + 0.15*calls2 - 0.25*engage2 + 0.05  # slight drift
prob2 = 1/(1+np.exp(-logit2))
y2 = (prob2 > 0.5).astype(int)

df_cls2 = pd.DataFrame({
    "tenure": tenure2, "spend": spend2, "calls": calls2, "engagement": engage2, "churn": y2
})

# Combine old + new; retrain from scratch (batch)
df_all = pd.concat([df_cls, df_cls2], ignore_index=True)
X_all = df_all[["tenure","spend","calls","engagement"]].values
y_all = df_all["churn"].values
scaler2 = StandardScaler()
X_all_scaled = scaler2.fit_transform(X_all)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all_scaled, y_all, test_size=0.25, random_state=1)

clf2 = LogisticRegression(max_iter=200)
clf2.fit(X_train2, y_train2)  # <--- batch retrain
y_pred2 = clf2.predict(X_test2)

print("\n=== Example 3: Periodic Retraining (Batch) ===")
print("Accuracy after retrain:", round(accuracy_score(y_test2, y_pred2), 4))

dump(clf2, os.path.join(OUTDIR, "churn_model_retrained.joblib"))
dump(scaler2, os.path.join(OUTDIR, "churn_scaler_retrained.joblib"))

# --------------------------------
# Example 4 — Batch Scoring (Bulk)
# --------------------------------
# Simulate a CSV of new applicants to score
to_score = pd.DataFrame({
    "tenure":[3, 12, 24, 6, 48],
    "spend":[25, 60, 40, 120, 45],
    "calls":[0, 1, 2, 5, 1],
    "engagement":[9.2, 3.1, 6.0, 1.5, 8.0]
})
csv_path = os.path.join(OUTDIR, "new_applicants.csv")
to_score.to_csv(csv_path, index=False)

# Load latest retrained model + scaler, run batch predictions
scaler_loaded = load(os.path.join(OUTDIR, "churn_scaler_retrained.joblib"))
model_loaded = load(os.path.join(OUTDIR, "churn_model_retrained.joblib"))

X_new = scaler_loaded.transform(to_score.values)
preds = model_loaded.predict(X_new)
result = to_score.copy()
result["predicted_churn"] = preds
out_pred_path = os.path.join(OUTDIR, "new_applicants_scored.csv")
result.to_csv(out_pred_path, index=False)

print("\n=== Example 4: Batch Scoring ===")
print("Wrote predictions to:", out_pred_path)
