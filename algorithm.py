# ml_trading_model.py

import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# -----------------------------
# Step 1: Load Data
# -----------------------------
try:
    df = pd.read_csv("data.csv")  # Replace with your file path
except FileNotFoundError:
    print("⚠️ data.csv not found. Generating synthetic data.")
    np.random.seed(0)
    n = 500
    df = pd.DataFrame({
        "open": np.random.uniform(100, 200, n),
        "high": np.random.uniform(200, 250, n),
        "low": np.random.uniform(90, 150, n),
        "close": np.random.uniform(100, 220, n),
        "volume": np.random.randint(1000, 10000, n)
    })

# -----------------------------
# Step 2: Feature Engineering
# -----------------------------
df["garman_klass_vol"] = (np.log(df["high"]) - np.log(df["low"]))**2 - \
                         (2 * np.log(2) - 1) * (np.log(df["close"]) - np.log(df["open"]))**2

# MACD
macd = ta.macd(df["close"])
df = pd.concat([df, macd], axis=1)

# RSI
df["rsi"] = ta.rsi(df["close"])

# Bollinger Bands
bb = ta.bbands(df["close"])
df["bollinger_hband"] = bb["BBU_20_2.0"]
df["bollinger_lband"] = bb["BBL_20_2.0"]

# ATR
df["atr"] = ta.atr(df["high"], df["low"], df["close"])

# -----------------------------
# Step 3: Target Variable
# -----------------------------
# Predict if price will go UP the next day
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# -----------------------------
# Step 4: Clean & Split
# -----------------------------
df.dropna(inplace=True)

X = df.drop(columns=["target", "close", "open", "high", "low", "volume"])
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False)

# -----------------------------
# Step 5: Train Model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

print("⏳ Training model...")
start = time.time()
model.fit(X_train, y_train)
end = time.time()
print(f"✅ Model trained in {end - start:.2f} seconds.\n")

# -----------------------------
# Step 6: Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("🔍 Classification Report:\n")
print(classification_report(y_test, y_pred))

print("📉 Confusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------
# Step 7: Feature Importance
# -----------------------------
importances = model.feature_importances_
feat_names = X.columns

fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
fi_df.sort_values("Importance", ascending=True, inplace=True)

plt.figure(figsize=(8, 6))
plt.barh(fi_df["Feature"], fi_df["Importance"], color="teal")
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
