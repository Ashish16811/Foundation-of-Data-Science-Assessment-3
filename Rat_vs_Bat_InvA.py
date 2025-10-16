"""
================================================================================
INVESTIGATION A
Question: Do bats perceive rats as potential predators?
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
sns.set_palette("husl")
plt.rcParams.update({"figure.figsize": (14, 6), "font.size": 10})

def section(title):
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)

def load_dataset(path1="dataset1.csv", path2="dataset2.csv"):
    try:
        d1, d2 = pd.read_csv(path1), pd.read_csv(path2)
        print(f"Loaded {path1} ({d1.shape}) and {path2} ({d2.shape})")
        return d1, d2
    except FileNotFoundError:
        raise SystemExit(" Dataset files not found.")

def summarize_missing(df, name):
    miss = df.isnull().sum()
    return pd.DataFrame({"Missing": miss, "%": (miss / len(df) * 100).round(2)})

def fill_missing_median(df):
    for col in df.select_dtypes(include=np.number):
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

def detect_outliers_iqr(df, col):
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    return len(outliers), (lower, upper)

def chi_square_test(series, alpha=0.05):
    obs = series.value_counts().sort_index()
    chi, p = stats.chisquare(obs, [len(series)/2]*2)
    return chi, p, p < alpha

def one_sample_ttest(series, mu0=5, alpha=0.05, side="greater"):
    t, p = stats.ttest_1samp(series, mu0)
    p_side = p/2 if side == "greater" else p
    return t, p_side, (p_side < alpha and t > 0)

def pearson_test(x, y, alpha=0.05):
    r, p = stats.pearsonr(x, y)
    return r, p, (p < alpha)

def independent_ttest(a, b, alpha=0.05):
    t, p = stats.ttest_ind(a, b)
    return t, p, p < alpha


# Data Loading & Inspection
section("DATA LOADING AND EXPLORATION")
df1, df2 = load_dataset()

print("\nDataset 1 columns:", list(df1.columns))
print("Dataset 2 columns:", list(df2.columns))

# Data Quality Assessment
section("DATA QUALITY ASSESSMENT")
print("Missing values summary:\n")
print("Dataset 1\n", summarize_missing(df1, "df1"))
print("\nDataset 2\n", summarize_missing(df2, "df2"))

fill_missing_median(df1)
fill_missing_median(df2)

for col in ["bat_landing_to_food", "seconds_after_rat_arrival", "hours_after_sunset"]:
    if col in df1:
        n, (lo, hi) = detect_outliers_iqr(df1, col)
        print(f"{col}: {n} outliers; valid range [{lo:.2f}, {hi:.2f}]")

# Descriptive Statistics
section("DESCRIPTIVE STATISTICS")

# Vigilance
vig = df1["bat_landing_to_food"].describe()
print(f"Mean ={vig['mean']:.2f}s, Median ={vig['50%']:.2f}s, Std ={vig['std']:.2f}s")

# Risk Behaviour
risk = df1["risk"].value_counts(normalize=True) * 100
avoid, take = risk.get(0, 0), risk.get(1, 0)
print(f"Risk-avoid ={avoid:.1f}%, Risk-take ={take:.1f}%")

# Foraging Success
reward = df1["reward"].value_counts(normalize=True) * 100
print(f"Foraging success ={reward.get(1,0):.1f}%")

# Risk vs Reward Cross-tab
print(pd.crosstab(df1["risk"], df1["reward"], normalize="index")*100)

# Behavioural Habit
print("Habit distribution:\n", df1["habit"].value_counts(normalize=True)*100)

# Rat Impact
r_corr = df2["rat_arrival_number"].corr(df2["bat_landing_number"])
print(f"Correlation (rats vs bats) = {r_corr:.3f}")


# Inferential Statistics
section("INFERENTIAL TESTS")

# χ² Test
chi, p_chi, sig_chi = chi_square_test(df1["risk"])
print(f"Chi²={chi:.3f}, p={p_chi:.4f}, significant={sig_chi}")

# One-sample t-test for vigilance
t_vig, p_vig, sig_vig = one_sample_ttest(df1["bat_landing_to_food"])
print(f"T={t_vig:.3f}, p={p_vig:.4f}, significant={sig_vig}")

# Pearson Correlation
r, p_r, sig_r = pearson_test(df2["rat_arrival_number"], df2["bat_landing_number"])
print(f"r={r:.3f}, p={p_r:.4f}, significant={sig_r}")

# Independent t-test
a = df1[df1["risk"]==0]["bat_landing_to_food"]
b = df1[df1["risk"]==1]["bat_landing_to_food"]
t_ind, p_ind, sig_ind = independent_ttest(a, b)
print(f"T={t_ind:.3f}, p={p_ind:.4f}, significant={sig_ind}")

# Visualisations
section("VISUALISATIONS")

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Helper for axis trimming
def cap(series, p=0.99):
    return series.clip(upper=series.quantile(p))

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.ravel()

# 1. Vigilance distribution (trimmed)
x = cap(df1["bat_landing_to_food"].dropna())
sns.histplot(x, kde=True, bins=30, color="skyblue", ax=axes[0])
axes[0].set_title("Distribution of Vigilance")
axes[0].set_xlabel("Time to Food (seconds)")
axes[0].set_ylabel("Count")
axes[0].axvline(x.mean(), color="red", ls="--", lw=1.2, label=f"Mean={x.mean():.1f}")
axes[0].axvline(x.median(), color="green", ls=":", lw=1.2, label=f"Median={x.median():.1f}")
axes[0].legend()

# 2. Risk behaviour
sns.countplot(x="risk", data=df1, ax=axes[1], palette=["#2ecc71","#e74c3c"])
axes[1].set_title("Risk Behaviour")
axes[1].set_xlabel("Risk Type")
axes[1].set_ylabel("Count")
for c in axes[1].containers:
    axes[1].bar_label(c, fmt="%.0f", label_type="edge", fontsize=9)

# 3. Risk vs Reward
ct = pd.crosstab(df1["risk"], df1["reward"], normalize="index") * 100
ct.plot(kind="bar", ax=axes[2], color=["#f39c12", "#27ae60"], edgecolor="black")
axes[2].set_title("Risk vs Reward (%)")
axes[2].set_xlabel("Risk Behaviour (0=Avoid,1=Take)")
axes[2].set_ylabel("% within Risk Group")
axes[2].legend(title="Reward", loc="upper right")

# 4. Vigilance by Risk (boxplot)
sns.boxplot(x="risk", y="bat_landing_to_food", data=df1, ax=axes[3], showfliers=False,
            palette=["#2ecc71","#e74c3c"])
axes[3].set_title("Vigilance by Risk (Outliers Hidden)")
axes[3].set_xlabel("Risk Behaviour")
axes[3].set_ylabel("Time to Food (seconds)")
axes[3].set_ylim(0, df1["bat_landing_to_food"].quantile(0.95)*1.1)

# 5. Rat Presence vs Bat Activity (scatter)
sns.regplot(x="rat_arrival_number", y="bat_landing_number", data=df2, ax=axes[4],
            scatter_kws={"alpha":0.5, "s":40}, line_kws={"color":"red"})
axes[4].set_title("Rat Presence vs Bat Activity")
axes[4].set_xlabel("Rat Arrivals (per 30 min)")
axes[4].set_ylabel("Bat Landings (per 30 min)")

# 6. Habit Distribution (Top 8 + Other)
hab = df1["habit"].astype(str).value_counts()
if len(hab) > 8:
    hab = pd.concat([hab.head(8), pd.Series({"Other": hab[8:].sum()})])
hab.sort_values().plot.barh(ax=axes[5], color="#3498db", edgecolor="black")
axes[5].set_title("Habit Distribution (Top 8 + Other)")
axes[5].set_xlabel("Count")
axes[5].set_ylabel("Habit Type")

plt.tight_layout()
plt.savefig("investigationA_clean.png", dpi=300, bbox_inches="tight")
print("Cleaned visualisations saved as 'investigationA_clean.png'")
plt.show()


print(f"Vigilance ={sig_vig}, Risk-avoid ={sig_chi}, Rat-Bat neg corr ={sig_r}")
evidence = sum([sig_vig, sig_chi, sig_r])
if evidence >= 2:
    print("Bats perceive rats as predation risk")
elif evidence == 1:
    print("Some risk perception signals")
else:
    print("No predator perception detected")
