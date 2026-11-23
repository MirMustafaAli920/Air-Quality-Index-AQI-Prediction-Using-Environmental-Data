import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

POLLUTANTS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]

# CPCB breakpoints: concentration points and the AQI value at each point
# (PM2.5, PM10, NO2, SO2, O3 in ug/m3 | CO in mg/m3)
INDEX_POINTS = [0, 50, 100, 200, 300, 400, 500]
BREAKPOINTS = {
    "PM2.5": [0, 30, 60, 90, 120, 250, 380],
    "PM10":  [0, 50, 100, 250, 350, 430, 510],
    "NO2":   [0, 40, 80, 180, 280, 400, 520],
    "SO2":   [0, 40, 80, 380, 800, 1600, 2100],
    "CO":    [0, 1, 2, 10, 17, 34, 50],
    "O3":    [0, 50, 100, 168, 208, 748, 1000],
}


# ---------- Step 1: formula-based AQI ----------
def sub_indices(df):
    """AQI sub-index of each pollutant (linear interpolation between breakpoints)."""
    return pd.DataFrame({p: np.interp(df[p], BREAKPOINTS[p], INDEX_POINTS) for p in POLLUTANTS},
                        index=df.index).where(df[POLLUTANTS].notna())


def calculate_aqi(df):
    """Overall AQI = the highest sub-index (needs >=3 pollutants, incl. PM2.5 or PM10)."""
    sub = sub_indices(df)
    aqi = sub.max(axis=1)
    valid = (sub.notna().sum(axis=1) >= 3) & (df["PM2.5"].notna() | df["PM10"].notna())
    return aqi.where(valid)


def aqi_category(aqi):
    if aqi <= 50:  return "Good😊👌12"
    if aqi <= 100: return "OK😐"
    if aqi <= 200: return "Normal😶‍🌫️"
    if aqi <= 400: return "deadly☠️"
    return "💀"


# ---------- Load data ----------
def load_data():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        df = pd.read_csv(sys.argv[1])
        print(f"Loaded {len(df)} rows from {sys.argv[1]}")
        return df[POLLUTANTS]
    if len(sys.argv) > 1:
        print(f"File not found: {sys.argv[1]}")
    print("Using synthetic demo data")
    rng = np.random.default_rng(42)
    n = 3000
    return pd.DataFrame({
        "PM2.5": rng.uniform(5, 250, n), "PM10": rng.uniform(10, 400, n),
        "NO2": rng.uniform(5, 100, n),   "SO2": rng.uniform(2, 60, n),
        "CO": rng.uniform(0.2, 5, n),    "O3": rng.uniform(10, 120, n),
    })


data = load_data()
data["AQI"] = calculate_aqi(data)
data = data.dropna(subset=["AQI"]).copy()
data[POLLUTANTS] = data[POLLUTANTS].fillna(data[POLLUTANTS].median())
print(f"Rows with a calculable AQI: {len(data)}")

# ---------- Step 2: train ML model ----------
X, y = data[POLLUTANTS], data["AQI"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("\n--- ML model vs formula AQI (test set) ---")
print(f"MAE : {mean_absolute_error(y_test, pred):.2f}")
print(f"R2  : {r2_score(y_test, pred):.3f}")

# ---------- Step 3: enter your own values ----------
UNITS = {"PM2.5": "ug/m3", "PM10": "ug/m3", "NO2": "ug/m3",
         "SO2": "ug/m3", "CO": "mg/m3", "O3": "ug/m3"}
medians = data[POLLUTANTS].median()


def ask(name):
    while True:
        text = input(f"{name} ({UNITS[name]}) [Enter to skip]: ").strip()
        if text == "":
            return np.nan
        try:
            value = float(text)
            if value < 0:
                raise ValueError
            return value
        except ValueError:
            print("  Please enter a positive number.")


print("\n=== AQI Calculator ===")
print("Enter pollutant values (press Enter to skip one).")

while True:
    sample = pd.DataFrame([{p: ask(p) for p in POLLUTANTS}])

    print("\n--- Result ---")
    formula_aqi = calculate_aqi(sample).iloc[0]
    if np.isnan(formula_aqi):
        print("Formula AQI : not enough data (give at least 3 pollutants incl. PM2.5 or PM10)")
    else:
        print(f"Formula AQI : {formula_aqi:.0f} ({aqi_category(formula_aqi)})")
        sub = sub_indices(sample).iloc[0].dropna()
        print("\nSub-index of each pollutant (AQI = the highest one):")
        for name, value in sub.items():
            mark = "  <-- main pollutant" if name == sub.idxmax() else ""
            print(f"  {name:6s} {value:5.0f}{mark}")
        if sample["CO"].iloc[0] > 50:
            print("\n  Warning: CO should be in mg/m3. Values above 50 are usually ug/m3 or ppm - divide by 1000 if ug/m3.")

    ml_aqi = model.predict(sample.fillna(medians))[0]
    print(f"ML AQI      : {ml_aqi:.0f} ({aqi_category(ml_aqi)})")

    if input("\nCalculate another? (y/n): ").strip().lower() != "y":
        break
