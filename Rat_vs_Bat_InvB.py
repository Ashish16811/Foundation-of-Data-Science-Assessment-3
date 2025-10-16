"""
================================================================================
INVESTIGATION B
Question: Do bat behaviors change following seasonal changes?
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

sns.set_style("whitegrid")
sns.set_palette("Set2")
plt.rcParams.update({"figure.figsize": (14, 6), "font.size": 10})

def section(title):
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)

def normalize_season(df):
    """Create a lowercase, trimmed season column 'season_norm' if 'season' exists."""
    if "season" in df.columns:
        df["season_norm"] = (
            df["season"]
            .astype("string")   # robust to NaN / mixed types
            .str.strip()
            .str.lower()
        )

def safe_ttest_ind(a, b):
    """Return Welch t-test; if any group empty, return NaNs."""
    a, b = a.dropna(), b.dropna()
    if len(a) == 0 or len(b) == 0:
        return np.nan, np.nan
    return stats.ttest_ind(a, b, equal_var=False)

# LOAD & PREPARE DATA
section("DATA LOADING AND PREPARATION")

try:
    df1 = pd.read_csv("dataset1.csv")
    df2 = pd.read_csv("dataset2.csv")
    print("Successfully loaded both datasets")
except FileNotFoundError:
    raise SystemExit("Place dataset1.csv and dataset2.csv in the same folder.")

# Fill numeric missing values with median
for df in [df1, df2]:
    num_cols = df.select_dtypes(include=np.number).columns
    for c in num_cols:
        if df[c].isnull().sum() > 0:
            df[c].fillna(df[c].median(), inplace=True)

# Ensure 'season' column exists and normalise to 'season_norm'
if "season" not in df1.columns:
    raise SystemExit("'season' column missing from dataset1.csv")
normalize_season(df1)
if "season" in df2.columns:
    normalize_season(df2)

print("\nDataset summary:")
print(df1.groupby("season_norm").size())

# DESCRIPTIVE STATISTICS BY SEASON
section("DESCRIPTIVE STATISTICS BY SEASON")

season_summary = df1.groupby("season_norm").agg({
    "bat_landing_to_food": ["mean", "median", "std"],
    "risk": "mean",
    "reward": "mean"
}).round(2)

print(season_summary)

# INFERENTIAL TESTS (SEASONAL COMPARISONS)
section("STEP 3 – INFERENTIAL TESTS")

# Vigilance comparison (t-test)
winter = df1.loc[df1["season_norm"].eq("winter"), "bat_landing_to_food"]
spring = df1.loc[df1["season_norm"].eq("spring"), "bat_landing_to_food"]

t_vig, p_vig = safe_ttest_ind(winter, spring)
print(f"Vigilance t-test: t={t_vig:.3f}, p={p_vig:.4f}")

# Risk-taking comparison (Chi-square)
ct_risk = pd.crosstab(df1["season_norm"], df1["risk"])
if ct_risk.shape[0] >= 2 and ct_risk.shape[1] >= 2:
    chi_risk, p_risk, _, _ = stats.chi2_contingency(ct_risk)
else:
    chi_risk, p_risk = np.nan, np.nan
print(f"Risk-taking χ²={chi_risk:.3f}, p={p_risk:.4f}")

# Foraging success comparison (Chi-square)
ct_reward = pd.crosstab(df1["season_norm"], df1["reward"])
if ct_reward.shape[0] >= 2 and ct_reward.shape[1] >= 2:
    chi_reward, p_reward, _, _ = stats.chi2_contingency(ct_reward)
else:
    chi_reward, p_reward = np.nan, np.nan
print(f"Foraging success χ²={chi_reward:.3f}, p={p_reward:.4f}")

# Rat encounters (dataset2)
if "season_norm" in df2.columns:
    rat_w = df2.loc[df2["season_norm"].eq("winter"), "rat_arrival_number"]
    rat_s = df2.loc[df2["season_norm"].eq("spring"), "rat_arrival_number"]
    t_rat, p_rat = safe_ttest_ind(rat_w, rat_s)
    print(f"Rat encounters t-test: t={t_rat:.3f}, p={p_rat:.4f}")
else:
    print("season missing in dataset2.")
    t_rat = p_rat = np.nan

# VISUALISATIONS
section("VISUALISATIONS")

fig, ax = plt.subplots(2, 2, figsize=(14, 10))
axes = ax.ravel()

# 1. Vigilance by season
sns.boxplot(x="season_norm", y="bat_landing_to_food", data=df1,
            palette="Set3", showfliers=False, ax=axes[0])
axes[0].set_title("Vigilance by Season")
axes[0].set_xlabel("Season")
axes[0].set_ylabel("Time to Food (seconds)")

# 2. Risk behaviour by season
sns.barplot(x="season_norm", y="risk", data=df1, ci="sd", ax=axes[1])
axes[1].set_title("Risk-taking Rate by Season")
axes[1].set_xlabel("Season")
axes[1].set_ylabel("Proportion (Risk=1)")

# 3. Foraging success
sns.barplot(x="season_norm", y="reward", data=df1, ci="sd", ax=axes[2])
axes[2].set_title("Foraging Success Rate by Season")
axes[2].set_xlabel("Season")
axes[2].set_ylabel("Proportion (Reward=1)")

# 4. Rat activity
if "season_norm" in df2.columns:
    sns.boxplot(x="season_norm", y="rat_arrival_number", data=df2, palette="coolwarm", ax=axes[3])
    axes[3].set_title("Rat Arrivals per 30-min Observation")
    axes[3].set_xlabel("Season")
    axes[3].set_ylabel("Count")
else:
    axes[3].axis("off")
    axes[3].set_title("No 'season' in dataset2")

plt.tight_layout()
plt.savefig("investigationB_seasonal_comparison.png", dpi=300)
print("Saved figure: investigationB_seasonal_comparison.png")
plt.show()

# SUMMARY
higher_vig = "winter" if winter.mean() > spring.mean() else "spring"
print(f"Vigilance: Higher in {higher_vig}")
print("Risk-taking:", "Significant difference" if (pd.notna(p_risk) and p_risk < 0.05) else "No significant difference")
print("Foraging success:", "Significant difference" if (pd.notna(p_reward) and p_reward < 0.05) else "No significant difference")
if "season_norm" in df2.columns:
    print("Rat encounters:", "Significant seasonal variation" if (pd.notna(p_rat) and p_rat < 0.05) else "No significant difference")

any_sig = any([
    pd.notna(p_vig) and p_vig < 0.05,
    pd.notna(p_risk) and p_risk < 0.05,
    pd.notna(p_reward) and p_reward < 0.05
])
print(f"\nOverall: {'Season significantly affects bat behaviour' if any_sig else 'No strong seasonal effect detected'}")
