import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import zscore

temps_raw = [
    22,
    23,
    np.nan,
    np.nan,
    np.nan,
    25,
    26,
    58,
    24,
    23,
    22,
    21,
    23,
    24,
    25,
]
df = pd.DataFrame({"Hour": list(range(1, 16)), "Temperature": temps_raw})

non_null_temps = df["Temperature"].dropna()
mean_before = df["Temperature"].mean()
std_before = df["Temperature"].std()
skew_before = stats.skew(non_null_temps)
kurt_before = stats.kurtosis(non_null_temps)

print("=== Stats BEFORE cleaning ===")
print(f"Mean: {mean_before:.2f}, Std: {std_before:.2f}")
print(f"Skewness: {skew_before:.2f}, Kurtosis: {kurt_before:.2f}")

df["Temp_interp"] = df["Temperature"].interpolate(method="linear")
print(
    f"\nAfter interpolation, missing values: {df['Temp_interp'].isnull().sum()}"
)

z_scores_interp = np.abs(zscore(df["Temp_interp"]))
outlier_mask = z_scores_interp > 2

flagged_hours = df[outlier_mask]["Hour"].tolist()
flagged_values = df[outlier_mask]["Temp_interp"].tolist()

print(
    f"Z-score(>2) outlier detected at Hour: {flagged_hours} (value: {flagged_values})"
)

clean_median = df["Temp_interp"].median()
df["Temp_clean"] = df["Temp_interp"].copy()
df.loc[outlier_mask, "Temp_clean"] = clean_median
print(f"Replaced with column median: {clean_median}")

df["Temp_Zscore"] = zscore(df["Temp_clean"])

comparison_df = df[
    ["Hour", "Temperature", "Temp_clean", "Temp_Zscore"]
].rename(columns={"Temperature": "Temp_raw"})

print("\n=== Before/After Comparison Table ===")
print(comparison_df.to_string(index=False))

# Stats AFTER cleaning
mean_after = df["Temp_clean"].mean()
std_after = df["Temp_clean"].std()

print("\n=== Stats AFTER cleaning ===")
print(f"Mean: {mean_after:.2f}, Std: {std_after:.2f}")