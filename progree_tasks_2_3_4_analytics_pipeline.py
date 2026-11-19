import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

INTER_DIR = "artifacts/file_generation/ttl=63d/intermediate"
OUT_DIR = "artifacts/file_generation/ttl=63d/output"
os.makedirs(INTER_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# ==============================================================================
# 1. GENERATE REALISTIC RAW ENTERPRISE DATASET (5,200 clean tx + 85 dups = 5,285 raw)
# ==============================================================================
n_customers = 1200
n_tx = 5200

regions = [
    ("Dhaka Metro", "Bangladesh", 23.8103, 90.4125, 0.28),
    ("Chattogram Hub", "Bangladesh", 22.3569, 91.7832, 0.14),
    ("Singapore APAC", "Singapore", 1.3521, 103.8198, 0.16),
    ("London EMEA", "United Kingdom", 51.5072, -0.1276, 0.12),
    ("New York North", "United States", 40.7128, -74.0060, 0.12),
    ("Dubai MENA", "UAE", 25.2048, 55.2708, 0.08),
    ("Toronto Hub", "Canada", 43.6532, -79.3832, 0.05),
    ("Sydney ANZ", "Australia", -33.8688, 151.2093, 0.05),
]

categories = [
    ("Enterprise Cloud Suite", 420.0, 65.0),
    ("Data Analytics Pro", 310.0, 48.0),
    ("CyberSecurity Shield", 275.0, 42.0),
    ("FinTech API Gateway", 195.0, 32.0),
    ("IoT Edge Sensors", 145.0, 24.0),
]

channels = ["Organic Search", "LinkedIn B2B Ads", "Partner Referral", "Email Campaign"]
channel_probs = [0.34, 0.29, 0.21, 0.16]

cust_ids = [f"CUST-{10000 + i}" for i in range(n_customers)]
cust_region_idx = np.random.choice(len(regions), size=n_customers, p=[r[4] for r in regions])
cust_channel = np.random.choice(channels, size=n_customers, p=channel_probs)
cust_ab_group = np.random.choice(["Control_A", "Variant_B"], size=n_customers, p=[0.5, 0.5])

# Channel-modulated loyalty propensity (Partner Referral & LinkedIn B2B Ads yield stronger enterprise retention)
ch_scale = np.array([1.45 if ch == "Partner Referral" else (1.30 if ch == "LinkedIn B2B Ads" else (0.95 if ch == "Organic Search" else 0.80)) for ch in cust_channel])
cust_propensity = np.random.gamma(shape=2.4, scale=ch_scale, size=n_customers)
cust_weights = cust_propensity / cust_propensity.sum()

# Generate transactions across 104 full weeks (2024-11-04 to 2026-11-01)
start_date = pd.Timestamp("2024-11-04")
n_hist_weeks = 104

# Allocate weekly transactions following growth + annual + semi-annual seasonality
w_idx = np.arange(n_hist_weeks)
weekly_weights = (
    38.0
    + 0.24 * w_idx
    + 6.5 * np.sin(2 * np.pi * w_idx / 52.0)
    + 4.2 * np.cos(2 * np.pi * w_idx / 52.0)
    + 3.1 * np.sin(4 * np.pi * w_idx / 52.0)
    + np.random.normal(0, 1.4, size=n_hist_weeks)
)
weekly_weights = np.clip(weekly_weights, 20, None)
weekly_probs = weekly_weights / weekly_weights.sum()

tx_week_indices = np.random.choice(n_hist_weeks, size=n_tx, p=weekly_probs)
tx_cust_indices = np.random.choice(n_customers, size=n_tx, p=cust_weights)

rows = []
for i in range(n_tx):
    c_idx = tx_cust_indices[i]
    c_id = cust_ids[c_idx]
    reg_name, country, lat, lon, _ = regions[cust_region_idx[c_idx]]
    ch = cust_channel[c_idx]
    ab = cust_ab_group[c_idx]
    wk = int(tx_week_indices[i])
    dt = start_date + pd.Timedelta(days=wk * 7 + int(np.random.randint(0, 7)))
    
    cat_idx = np.random.choice(len(categories), p=[0.26, 0.28, 0.20, 0.15, 0.11])
    cat_name, base_price, price_sd = categories[cat_idx]
    
    ab_boost = 1.15 if ab == "Variant_B" else 1.0
    ch_boost = 1.14 if ch in ["Partner Referral", "LinkedIn B2B Ads"] else 1.0
    season_boost = 1.0 + 0.06 * np.sin(2 * np.pi * wk / 52.0) + 0.0012 * wk
    
    unit_price = max(40.0, np.random.normal(base_price * season_boost, price_sd * 0.55))
    quantity = max(1, int(np.round(np.random.normal(3.2 * ab_boost * ch_boost, 0.85))))
    discount_pct = np.round(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.45, 0.25, 0.15, 0.10, 0.05]), 2)
    shipping_cost = np.round(max(5.0, np.random.normal(18.5, 4.0) + quantity * 1.8), 2)
    csat_score = np.round(np.clip(np.random.normal(4.22 if ab == "Variant_B" else 3.84, 0.50), 1.0, 5.0), 1)
    
    gross_rev = unit_price * quantity
    net_rev = np.round(gross_rev * (1.0 - discount_pct) + shipping_cost, 2)
    
    rows.append({
        "transaction_id": f"TX-{500000 + i}",
        "customer_id": c_id,
        "transaction_date": dt.strftime("%Y-%m-%d"),
        "region": reg_name,
        "country": country,
        "latitude": lat,
        "longitude": lon,
        "acquisition_channel": ch,
        "experiment_group": ab,
        "product_category": cat_name,
        "quantity": quantity,
        "unit_price": np.round(unit_price, 2),
        "discount_pct": discount_pct,
        "shipping_cost": shipping_cost,
        "customer_satisfaction": csat_score,
        "net_revenue": net_rev
    })

df_raw = pd.DataFrame(rows)

# Inject realistic data abnormalities for Task 2:
missing_up_idx = np.random.choice(df_raw.index, size=234, replace=False)
missing_disc_idx = np.random.choice(df_raw.index, size=322, replace=False)
missing_ship_idx = np.random.choice(df_raw.index, size=198, replace=False)
missing_csat_idx = np.random.choice(df_raw.index, size=286, replace=False)

df_raw.loc[missing_up_idx, "unit_price"] = np.nan
df_raw.loc[missing_disc_idx, "discount_pct"] = np.nan
df_raw.loc[missing_ship_idx, "shipping_cost"] = np.nan
df_raw.loc[missing_csat_idx, "customer_satisfaction"] = np.nan

outlier_hi_idx = np.random.choice(df_raw.index, size=78, replace=False)
df_raw.loc[outlier_hi_idx, "unit_price"] = df_raw.loc[outlier_hi_idx, "unit_price"].fillna(350.0) * np.random.uniform(5.5, 9.5, size=78)
df_raw.loc[outlier_hi_idx, "net_revenue"] = np.round(df_raw.loc[outlier_hi_idx, "unit_price"] * df_raw.loc[outlier_hi_idx, "quantity"], 2)

corrupt_qty_idx = np.random.choice(list(set(df_raw.index) - set(outlier_hi_idx)), size=18, replace=False)
df_raw.loc[corrupt_qty_idx, "quantity"] = -np.abs(df_raw.loc[corrupt_qty_idx, "quantity"])

dup_rows = df_raw.sample(n=85, random_state=42)
df_raw = pd.concat([df_raw, dup_rows], ignore_index=True)

raw_csv_path = os.path.join(INTER_DIR, "progree_raw_enterprise_transactions.csv")
df_raw.to_csv(raw_csv_path, index=False)

# ==============================================================================
# 2. TASK 2: EXPLORATORY DATA ANALYSIS & INTEGRITY CLEANING PIPELINE
# ==============================================================================
initial_rows = len(df_raw)
missing_before = df_raw.isnull().sum().to_dict()
duplicates_count = int(df_raw.duplicated().sum())
neg_qty_count = int((df_raw["quantity"] <= 0).sum())

df_clean = df_raw.drop_duplicates().copy()
df_clean["quantity"] = df_clean["quantity"].abs().replace(0, 1)

df_clean["discount_pct"] = df_clean["discount_pct"].fillna(0.0)
df_clean["unit_price"] = df_clean.groupby("product_category")["unit_price"].transform(lambda x: x.fillna(x.median()))
df_clean["shipping_cost"] = df_clean.groupby("region")["shipping_cost"].transform(lambda x: x.fillna(x.median()))
df_clean["customer_satisfaction"] = df_clean.groupby("experiment_group")["customer_satisfaction"].transform(lambda x: np.round(x.fillna(x.mean()), 1))

outlier_counts_by_cat = {}
pre_clean_revenues = df_clean["net_revenue"].copy()

for cat in df_clean["product_category"].unique():
    mask = df_clean["product_category"] == cat
    s = df_clean.loc[mask, "unit_price"]
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower, upper = max(10.0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
    n_out = int(((s < lower) | (s > upper)).sum())
    outlier_counts_by_cat[cat] = {
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2),
        "iqr": round(float(iqr), 2),
        "upper_fence": round(float(upper), 2),
        "outliers_capped": n_out
    }
    df_clean.loc[mask, "unit_price"] = np.clip(s, lower, upper)

df_clean["net_revenue"] = np.round(
    df_clean["unit_price"] * df_clean["quantity"] * (1.0 - df_clean["discount_pct"]) + df_clean["shipping_cost"], 2
)

clean_csv_path = os.path.join(OUT_DIR, "task2_cleaned_enterprise_data.csv")
df_clean.to_csv(clean_csv_path, index=False)

# Plot Task 2 EDA Figure
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9.5
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), dpi=220, constrained_layout=True)

anom_labels = ["Missing\nDiscount", "Missing\nCSAT", "Missing\nUnit Price", "Missing\nShipping", "Duplicate\nRows", "IQR Price\nOutliers", "Negative\nQuantity"]
anom_counts = [
    missing_before["discount_pct"],
    missing_before["customer_satisfaction"],
    missing_before["unit_price"],
    missing_before["shipping_cost"],
    duplicates_count,
    sum(v["outliers_capped"] for v in outlier_counts_by_cat.values()),
    neg_qty_count
]
bars = axes[0, 0].bar(anom_labels, anom_counts, color=["#1e3a8a", "#1d4ed8", "#2563eb", "#3b82f6", "#d97706", "#dc2626", "#7c3aed"], width=0.6)
axes[0, 0].set_title("A. Data Quality Audit: Anomalies Remediated", fontweight="bold", fontsize=10)
axes[0, 0].set_ylabel("Record Count")
axes[0, 0].tick_params(axis='x', labelsize=8)
axes[0, 0].set_ylim(0, max(anom_counts) * 1.22)
for b in bars:
    axes[0, 0].text(b.get_x() + b.get_width()/2, b.get_height() + 6, f"{int(b.get_height())}", ha='center', va='bottom', fontsize=8, fontweight='bold')

sns.kdeplot(np.clip(pre_clean_revenues, 0, 6000), ax=axes[0, 1], color="#dc2626", fill=True, alpha=0.25, label="Raw (Uncapped Skew)")
sns.kdeplot(df_clean["net_revenue"], ax=axes[0, 1], color="#0f766e", fill=True, alpha=0.35, label="Cleaned (Winsorized)")
axes[0, 1].set_title("B. Net Order Value Distribution (Raw vs. Cleaned)", fontweight="bold", fontsize=10)
axes[0, 1].set_xlabel("Net Order Revenue (USD)")
axes[0, 1].set_ylabel("Density")
axes[0, 1].set_xlim(0, 4200)
axes[0, 1].legend(loc="upper right", fontsize=8, frameon=True)

cat_order = ["Enterprise Cloud Suite", "Data Analytics Pro", "CyberSecurity Shield", "FinTech API Gateway", "IoT Edge Sensors"]
short_cats = ["Cloud Suite", "Analytics Pro", "CyberSecurity", "FinTech API", "IoT Edge"]
cat_data = [df_clean.loc[df_clean["product_category"] == c, "net_revenue"].values for c in cat_order]
bp = axes[1, 0].boxplot(cat_data, tick_labels=short_cats, patch_artist=True, showfliers=False)
palette_box = ["#1e3a8a", "#0f766e", "#2563eb", "#d97706", "#64748b"]
for patch, color in zip(bp['boxes'], palette_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
axes[1, 0].set_title("C. Order Value Spread by Product Category", fontweight="bold", fontsize=10)
axes[1, 0].set_ylabel("Net Revenue (USD)")
axes[1, 0].tick_params(axis='x', labelsize=8, rotation=12)

num_cols = ["quantity", "unit_price", "discount_pct", "shipping_cost", "customer_satisfaction", "net_revenue"]
corr_mat = df_clean[num_cols].corr()
short_num = ["Qty", "Unit $", "Disc %", "Ship $", "CSAT", "Net Rev"]
im = axes[1, 1].imshow(corr_mat, cmap="coolwarm", vmin=-0.3, vmax=1.0, aspect="auto")
axes[1, 1].set_xticks(range(len(short_num)))
axes[1, 1].set_yticks(range(len(short_num)))
axes[1, 1].set_xticklabels(short_num, fontsize=8)
axes[1, 1].set_yticklabels(short_num, fontsize=8)
for r in range(len(short_num)):
    for c in range(len(short_num)):
        val = corr_mat.iloc[r, c]
        axes[1, 1].text(c, r, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.55 else "#1f2937", fontsize=7.5, fontweight="bold")
axes[1, 1].set_title("D. Cleaned Parameter Correlation Matrix", fontweight="bold", fontsize=10)

fig.savefig(os.path.join(INTER_DIR, "fig_task2_eda.png"), dpi=220)
plt.close(fig)

# ==============================================================================
# 3. TASK 3: CUSTOMER SEGMENTATION (RFM), COHORT & GEOSPATIAL ENGINE
# ==============================================================================
df_clean["transaction_date_dt"] = pd.to_datetime(df_clean["transaction_date"])
snapshot_date = pd.Timestamp("2026-11-05")

rfm = df_clean.groupby("customer_id").agg(
    recency_days=("transaction_date_dt", lambda x: (snapshot_date - x.max()).days),
    frequency_orders=("transaction_id", "count"),
    monetary_total=("net_revenue", lambda x: np.round(x.sum(), 2)),
    avg_order_value=("net_revenue", lambda x: np.round(x.mean(), 2)),
    avg_csat=("customer_satisfaction", lambda x: np.round(x.mean(), 2)),
    region=("region", "first"),
    country=("country", "first"),
    latitude=("latitude", "first"),
    longitude=("longitude", "first"),
    acquisition_channel=("acquisition_channel", "first"),
    experiment_group=("experiment_group", "first"),
    first_purchase_date=("transaction_date_dt", "min"),
    last_purchase_date=("transaction_date_dt", "max")
).reset_index()

rfm["R_Score"] = pd.qcut(rfm["recency_days"].rank(method="first"), q=5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_Score"] = pd.qcut(rfm["frequency_orders"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_Score"] = pd.qcut(rfm["monetary_total"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["RFM_Score"] = rfm["R_Score"] * 100 + rfm["F_Score"] * 10 + rfm["M_Score"]
rfm["RFM_Index"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

def classify_rfm(row):
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 4:
        return "Loyal Customers"
    elif r >= 4 and f >= 2:
        return "Potential Loyalists"
    elif r <= 2 and (f >= 3 or m >= 4):
        return "At Risk (High Value)"
    else:
        return "Hibernating / Churned"

rfm["customer_segment"] = rfm.apply(classify_rfm, axis=1)
rfm["first_purchase_date"] = rfm["first_purchase_date"].dt.strftime("%Y-%m-%d")
rfm["last_purchase_date"] = rfm["last_purchase_date"].dt.strftime("%Y-%m-%d")

rfm_csv_path = os.path.join(OUT_DIR, "task3_rfm_customer_segments_tableau.csv")
rfm.to_csv(rfm_csv_path, index=False)

# Cohort Retention Matrix
df_clean["tx_quarter"] = df_clean["transaction_date_dt"].dt.to_period("Q")
df_clean["cohort_quarter"] = df_clean.groupby("customer_id")["transaction_date_dt"].transform("min").dt.to_period("Q")
df_clean["quarter_index"] = df_clean["tx_quarter"].astype("int64") - df_clean["cohort_quarter"].astype("int64")

cohort_counts = df_clean.groupby(["cohort_quarter", "quarter_index"])["customer_id"].nunique().unstack(fill_value=0)
cohort_sizes = cohort_counts.iloc[:, 0]
cohort_retention = cohort_counts.divide(cohort_sizes, axis=0).round(3) * 100.0
cohort_retention.index = cohort_retention.index.astype(str)
cohort_retention.columns = [f"Q+{c}" for c in cohort_retention.columns]
cohort_csv_path = os.path.join(OUT_DIR, "task3_cohort_retention_matrix.csv")
cohort_retention.reset_index().rename(columns={"index": "cohort_quarter"}).to_csv(cohort_csv_path, index=False)

# Plot Task 3 RFM & Cohort Figure
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), dpi=220, constrained_layout=True)

seg_order = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk (High Value)", "Hibernating / Churned"]
seg_summary = rfm.groupby("customer_segment").agg(
    customers=("customer_id", "count"),
    total_revenue=("monetary_total", "sum"),
    avg_monetary=("monetary_total", "mean"),
    avg_recency=("recency_days", "mean"),
    avg_frequency=("frequency_orders", "mean")
).reindex(seg_order)

colors_seg = ["#0f766e", "#1e3a8a", "#2563eb", "#d97706", "#64748b"]
y_pos = np.arange(len(seg_order))
rev_shares = (seg_summary["total_revenue"] / seg_summary["total_revenue"].sum() * 100).values
bars_s = axes[0].barh(y_pos, rev_shares, color=colors_seg, height=0.6)
axes[0].set_yticks(y_pos)
axes[0].set_yticklabels(seg_order, fontsize=8.5, fontweight="bold")
axes[0].invert_yaxis()
axes[0].set_xlabel("Share of Total Enterprise Revenue (%)")
axes[0].set_xlim(0, max(rev_shares) * 1.25)
axes[0].set_title("A. Revenue Contribution by RFM Segment", fontweight="bold", fontsize=9.5)
for b, cust_cnt in zip(bars_s, seg_summary["customers"].values):
    axes[0].text(b.get_width() + 0.8, b.get_y() + b.get_height()/2, f"{b.get_width():.1f}% (n={cust_cnt})", va="center", fontsize=8, fontweight="bold")

ret_sub = cohort_retention.iloc[:6, :6]
im2 = axes[1].imshow(ret_sub.values, cmap="YlGnBu", vmin=15, vmax=100, aspect="auto")
axes[1].set_xticks(range(ret_sub.shape[1]))
axes[1].set_yticks(range(ret_sub.shape[0]))
axes[1].set_xticklabels(ret_sub.columns, fontsize=8)
axes[1].set_yticklabels(ret_sub.index, fontsize=8)
axes[1].set_title("B. Quarterly Cohort Retention Heatmap (%)", fontweight="bold", fontsize=9.5)
axes[1].set_xlabel("Quarters Since Acquisition")
for r in range(ret_sub.shape[0]):
    for c in range(ret_sub.shape[1]):
        val = ret_sub.iloc[r, c]
        if val > 0:
            axes[1].text(c, r, f"{val:.0f}%", ha="center", va="center", color="white" if val > 55 else "#1f2937", fontsize=7.5, fontweight="bold")

fig.savefig(os.path.join(INTER_DIR, "fig_task3_rfm_cohort.png"), dpi=220)
plt.close(fig)

# ==============================================================================
# 4. TASK 4: STATISTICAL HYPOTHESIS TESTS & TIME-SERIES DEMAND FORECASTING
# ==============================================================================
rev_a = df_clean.loc[df_clean["experiment_group"] == "Control_A", "net_revenue"].values
rev_b = df_clean.loc[df_clean["experiment_group"] == "Variant_B", "net_revenue"].values
t_stat, p_val_ab = stats.ttest_ind(rev_b, rev_a, equal_var=False)
u_stat, p_val_mw = stats.mannwhitneyu(rev_b, rev_a, alternative="two-sided")
diff_mean = rev_b.mean() - rev_a.mean()
se_diff = np.sqrt(rev_b.var(ddof=1)/len(rev_b) + rev_a.var(ddof=1)/len(rev_a))
ci_low_ab, ci_high_ab = diff_mean - 1.96 * se_diff, diff_mean + 1.96 * se_diff
pooled_sd = np.sqrt((rev_b.var(ddof=1) + rev_a.var(ddof=1)) / 2)
cohens_d = diff_mean / pooled_sd

cat_groups = [df_clean.loc[df_clean["product_category"] == c, "net_revenue"].values for c in cat_order]
f_stat, p_val_anova = stats.f_oneway(*cat_groups)
ss_between = sum(len(g) * (g.mean() - df_clean["net_revenue"].mean())**2 for g in cat_groups)
ss_total = ((df_clean["net_revenue"] - df_clean["net_revenue"].mean())**2).sum()
eta_sq = ss_between / ss_total

rfm["is_high_value"] = np.where(rfm["customer_segment"].isin(["Champions", "Loyal Customers"]), "High_Value_Tier", "Standard_Tier")
ct = pd.crosstab(rfm["acquisition_channel"], rfm["is_high_value"])
chi2_stat, p_val_chi2, dof_chi2, _ = stats.chi2_contingency(ct)
n_cust_total = len(rfm)
cramers_v = np.sqrt(chi2_stat / (n_cust_total * (min(ct.shape) - 1)))

# Weekly Time-Series Aggregation (exactly 104 historical weeks)
df_clean["week_idx"] = ((df_clean["transaction_date_dt"] - start_date).dt.days // 7).clip(0, n_hist_weeks - 1)
weekly_ts = df_clean.groupby("week_idx").agg(
    weekly_revenue=("net_revenue", "sum"),
    weekly_orders=("transaction_id", "count"),
    active_customers=("customer_id", "nunique"),
    avg_discount=("discount_pct", "mean")
).reset_index().sort_values("week_idx")
weekly_ts["week_start"] = [start_date + pd.Timedelta(weeks=int(w)) for w in weekly_ts["week_idx"]]

n_weeks = len(weekly_ts)
t_idx = np.arange(n_weeks)

# Multivariate + Harmonic Time-Series Forecasting Model (SARIMAX-equivalent structure)
# First fit structural demand volume + seasonal revenue trajectory
X_ts = np.column_stack([
    np.ones(n_weeks),
    t_idx,
    np.sin(2 * np.pi * t_idx / 52.0),
    np.cos(2 * np.pi * t_idx / 52.0),
    np.sin(4 * np.pi * t_idx / 52.0),
    np.cos(4 * np.pi * t_idx / 52.0),
    weekly_ts["weekly_orders"].values,
])
y_ts = weekly_ts["weekly_revenue"].values
beta_mv, _, _, _ = np.linalg.lstsq(X_ts, y_ts, rcond=None)
y_fitted = X_ts @ beta_mv
residuals = y_ts - y_fitted

ss_res = np.sum((y_ts - y_fitted)**2)
ss_tot_ts = np.sum((y_ts - y_ts.mean())**2)
r2_score = 1.0 - ss_res / ss_tot_ts
rmse_score = np.sqrt(np.mean((y_ts - y_fitted)**2))
mae_score = np.mean(np.abs(y_ts - y_fitted))
mape_score = np.mean(np.abs((y_ts - y_fitted) / y_ts)) * 100.0
sigma_err = np.std(residuals, ddof=7)

# Project weekly_orders & revenue 16 weeks ahead
horizon = 16
t_future = np.arange(n_weeks, n_weeks + horizon)
last_date = weekly_ts["week_start"].iloc[-1]
future_dates = [last_date + pd.Timedelta(weeks=int(w+1)) for w in range(horizon)]

# Fit harmonic trend on weekly_orders to project exogenous volume for future 16 weeks
X_ord_hist = np.column_stack([
    np.ones(n_weeks),
    t_idx,
    np.sin(2 * np.pi * t_idx / 52.0),
    np.cos(2 * np.pi * t_idx / 52.0),
    np.sin(4 * np.pi * t_idx / 52.0),
    np.cos(4 * np.pi * t_idx / 52.0),
])
beta_ord, _, _, _ = np.linalg.lstsq(X_ord_hist, weekly_ts["weekly_orders"].values, rcond=None)
X_ord_fut = np.column_stack([
    np.ones(horizon),
    t_future,
    np.sin(2 * np.pi * t_future / 52.0),
    np.cos(2 * np.pi * t_future / 52.0),
    np.sin(4 * np.pi * t_future / 52.0),
    np.cos(4 * np.pi * t_future / 52.0),
])
proj_orders = X_ord_fut @ beta_ord

X_future = np.column_stack([
    np.ones(horizon),
    t_future,
    np.sin(2 * np.pi * t_future / 52.0),
    np.cos(2 * np.pi * t_future / 52.0),
    np.sin(4 * np.pi * t_future / 52.0),
    np.cos(4 * np.pi * t_future / 52.0),
    proj_orders,
])
y_forecast = X_future @ beta_mv

h_steps = np.arange(1, horizon + 1)
se_forecast = sigma_err * np.sqrt(1.25 + 0.10 * h_steps)
ci_80_low = y_forecast - 1.2816 * se_forecast
ci_80_high = y_forecast + 1.2816 * se_forecast
ci_95_low = y_forecast - 1.9600 * se_forecast
ci_95_high = y_forecast + 1.9600 * se_forecast

forecast_rows = []
for i in range(n_weeks):
    forecast_rows.append({
        "week_start": weekly_ts["week_start"].iloc[i].strftime("%Y-%m-%d"),
        "period_type": "Historical",
        "actual_revenue": round(float(y_ts[i]), 2),
        "predicted_revenue": round(float(y_fitted[i]), 2),
        "ci_80_lower": round(float(y_fitted[i] - 1.2816 * sigma_err), 2),
        "ci_80_upper": round(float(y_fitted[i] + 1.2816 * sigma_err), 2),
        "ci_95_lower": round(float(y_fitted[i] - 1.9600 * sigma_err), 2),
        "ci_95_upper": round(float(y_fitted[i] + 1.9600 * sigma_err), 2),
    })
for j in range(horizon):
    forecast_rows.append({
        "week_start": future_dates[j].strftime("%Y-%m-%d"),
        "period_type": "Forecast_16W",
        "actual_revenue": np.nan,
        "predicted_revenue": round(float(y_forecast[j]), 2),
        "ci_80_lower": round(float(ci_80_low[j]), 2),
        "ci_80_upper": round(float(ci_80_high[j]), 2),
        "ci_95_lower": round(float(ci_95_low[j]), 2),
        "ci_95_upper": round(float(ci_95_high[j]), 2),
    })

df_forecast = pd.DataFrame(forecast_rows)
forecast_csv_path = os.path.join(OUT_DIR, "task4_demand_forecast_results.csv")
df_forecast.to_csv(forecast_csv_path, index=False)

# Plot Task 4 Hypothesis Testing & Forecasting Figure
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), dpi=220, constrained_layout=True)

ab_means = [rev_a.mean(), rev_b.mean()]
ab_cis = [1.96 * rev_a.std(ddof=1)/np.sqrt(len(rev_a)), 1.96 * rev_b.std(ddof=1)/np.sqrt(len(rev_b))]
bars_ab = axes[0].bar(["Control A\n(Standard Flow)", "Variant B\n(AI Express Flow)"], ab_means, yerr=ab_cis, capsize=6, color=["#64748b", "#0f766e"], width=0.48)
axes[0].set_title(f"A. Checkout A/B Test (p = {p_val_ab:.2e}, Lift +{diff_mean/rev_a.mean()*100:.1f}%)", fontweight="bold", fontsize=9.2)
axes[0].set_ylabel("Mean Net Order Value (USD)")
axes[0].set_ylim(0, max(ab_means) * 1.28)
for b in bars_ab:
    axes[0].text(b.get_x() + b.get_width()/2, b.get_height() + 35, f"${b.get_height():,.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

axes[1].plot(weekly_ts["week_start"].iloc[-40:], y_ts[-40:] / 1000.0, color="#1e3a8a", lw=1.6, label="Actual Weekly Rev")
axes[1].plot(weekly_ts["week_start"].iloc[-40:], y_fitted[-40:] / 1000.0, color="#3b82f6", lw=1.2, ls="--", label="Model Fit")
axes[1].plot(future_dates, y_forecast / 1000.0, color="#d97706", lw=2.0, label="16-Wk Forecast")
axes[1].fill_between(future_dates, ci_95_low / 1000.0, ci_95_high / 1000.0, color="#f59e0b", alpha=0.25, label="95% Conf. Band")
axes[1].set_title(f"B. 16-Week Revenue Forecast (R²={r2_score:.2f}, MAPE={mape_score:.1f}%)", fontweight="bold", fontsize=9.2)
axes[1].set_ylabel("Weekly Revenue ($K USD)")
axes[1].tick_params(axis="x", labelsize=7.5, rotation=20)
axes[1].legend(loc="upper left", fontsize=7.5, frameon=True)

fig.savefig(os.path.join(INTER_DIR, "fig_task4_stats_forecast.png"), dpi=220)
plt.close(fig)

reg_summary = df_clean.groupby(["region", "country", "latitude", "longitude"]).agg(
    transactions=("transaction_id", "count"),
    customers=("customer_id", "nunique"),
    total_revenue=("net_revenue", lambda x: round(float(x.sum()), 2)),
    avg_order_value=("net_revenue", lambda x: round(float(x.mean()), 2)),
    avg_csat=("customer_satisfaction", lambda x: round(float(x.mean()), 2))
).reset_index().sort_values("total_revenue", ascending=False)

summary_metrics = {
    "task2": {
        "raw_rows": initial_rows,
        "clean_rows": len(df_clean),
        "unique_customers": int(df_clean["customer_id"].nunique()),
        "duplicates_removed": duplicates_count,
        "neg_qty_fixed": neg_qty_count,
        "missing_imputed": {k: int(v) for k, v in missing_before.items() if v > 0},
        "total_missing_imputed": int(sum(missing_before.values())),
        "outliers_by_category": outlier_counts_by_cat,
        "total_outliers_capped": int(sum(v["outliers_capped"] for v in outlier_counts_by_cat.values())),
        "raw_mean_rev": round(float(pre_clean_revenues.mean()), 2),
        "raw_sd_rev": round(float(pre_clean_revenues.std()), 2),
        "clean_mean_rev": round(float(df_clean["net_revenue"].mean()), 2),
        "clean_median_rev": round(float(df_clean["net_revenue"].median()), 2),
        "clean_sd_rev": round(float(df_clean["net_revenue"].std()), 2),
        "total_clean_revenue": round(float(df_clean["net_revenue"].sum()), 2),
    },
    "task3": {
        "segments": [
            {
                "segment": seg,
                "customers": int(seg_summary.loc[seg, "customers"]),
                "customer_share_pct": round(float(seg_summary.loc[seg, "customers"] / len(rfm) * 100), 1),
                "total_revenue": round(float(seg_summary.loc[seg, "total_revenue"]), 2),
                "revenue_share_pct": round(float(seg_summary.loc[seg, "total_revenue"] / seg_summary["total_revenue"].sum() * 100), 1),
                "avg_monetary": round(float(seg_summary.loc[seg, "avg_monetary"]), 2),
                "avg_recency_days": round(float(seg_summary.loc[seg, "avg_recency"]), 1),
                "avg_frequency": round(float(seg_summary.loc[seg, "avg_frequency"]), 2),
            }
            for seg in seg_order
        ],
        "regions": reg_summary.to_dict(orient="records"),
        "cohorts": cohort_retention.iloc[:6, :6].reset_index().to_dict(orient="records")
    },
    "task4": {
        "ab_test": {
            "control_n": int(len(rev_a)),
            "variant_n": int(len(rev_b)),
            "control_mean": round(float(rev_a.mean()), 2),
            "variant_mean": round(float(rev_b.mean()), 2),
            "abs_lift": round(float(diff_mean), 2),
            "pct_lift": round(float(diff_mean / rev_a.mean() * 100), 2),
            "t_stat": round(float(t_stat), 4),
            "p_value": float(p_val_ab),
            "u_stat": round(float(u_stat), 1),
            "p_value_mw": float(p_val_mw),
            "cohens_d": round(float(cohens_d), 3),
            "ci_95_low": round(float(ci_low_ab), 2),
            "ci_95_high": round(float(ci_high_ab), 2)
        },
        "anova": {
            "f_stat": round(float(f_stat), 2),
            "p_value": float(p_val_anova),
            "eta_squared": round(float(eta_sq), 4),
            "category_means": {
                c: round(float(df_clean.loc[df_clean["product_category"] == c, "net_revenue"].mean()), 2)
                for c in cat_order
            }
        },
        "chi_square": {
            "chi2_stat": round(float(chi2_stat), 2),
            "p_value": float(p_val_chi2),
            "dof": int(dof_chi2),
            "cramers_v": round(float(cramers_v), 4),
            "high_value_rate_by_channel": {
                ch: round(float((ct.loc[ch, "High_Value_Tier"] / ct.loc[ch].sum()) * 100), 1)
                for ch in channels
            }
        },
        "forecast": {
            "historical_weeks": int(n_weeks),
            "forecast_weeks": int(horizon),
            "r2": round(float(r2_score), 4),
            "rmse": round(float(rmse_score), 2),
            "mae": round(float(mae_score), 2),
            "mape_pct": round(float(mape_score), 2),
            "projected_16w_total": round(float(y_forecast.sum()), 2),
            "projected_16w_ci95_low": round(float(ci_95_low.sum()), 2),
            "projected_16w_ci95_high": round(float(ci_95_high.sum()), 2),
            "series_tail_and_forecast": df_forecast.tail(44).to_dict(orient="records")
        }
    }
}

with open(os.path.join(INTER_DIR, "summary_metrics.json"), "w") as f:
    json.dump(summary_metrics, f, indent=2)

# Copy standalone Python script to output so user can upload to GitHub
with open(__file__, "r") as src_f:
    code_text = src_f.read()
with open(os.path.join(OUT_DIR, "progree_tasks_2_3_4_analytics_pipeline.py"), "w") as out_f:
    out_f.write(code_text)

# Build clean Jupyter Notebook (.ipynb) in output for GitHub / Colab / Kaggle upload
notebook_obj = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Progree Data Analytics Internship — Tasks 2, 3 & 4 Complete Notebook\n",
                "**Author:** Data Analytics Intern (Progree)  \n",
                "**Environment:** Python 3 / Pandas / NumPy / SciPy / Matplotlib / Seaborn\n\n",
                "This notebook executes the end-to-end analytical pipeline across:\n",
                "- **Task 2:** Exploratory Data Analysis (EDA) & Data Integrity Cleaning Pipeline\n",
                "- **Task 3:** RFM Customer Segmentation, Cohort Retention & Geospatial Export for Tableau / Power BI\n",
                "- **Task 4:** Inferential Statistical Hypothesis Testing (A/B Test, One-Way ANOVA, Chi-Square) & 16-Week Demand Forecasting"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Cleaned Dataset Rows: {summary_metrics['task2']['clean_rows']} | Unique Customers: {summary_metrics['task2']['unique_customers']}\n",
                        f"Total Cleaned Net Revenue: ${summary_metrics['task2']['total_clean_revenue']:,.2f}\n",
                        f"A/B Test Lift: +{summary_metrics['task4']['ab_test']['pct_lift']}% (t = {summary_metrics['task4']['ab_test']['t_stat']}, p = {summary_metrics['task4']['ab_test']['p_value']:.2e})\n",
                        f"ANOVA Across 5 Categories: F = {summary_metrics['task4']['anova']['f_stat']}, p = {summary_metrics['task4']['anova']['p_value']:.2e}\n",
                        f"Chi-Square Channel vs RFM Tier: Chi2 = {summary_metrics['task4']['chi_square']['chi2_stat']}, p = {summary_metrics['task4']['chi_square']['p_value']:.2e}\n",
                        f"16-Week Demand Forecast: R2 = {summary_metrics['task4']['forecast']['r2']}, MAPE = {summary_metrics['task4']['forecast']['mape_pct']}%, Projected 16W Revenue = ${summary_metrics['task4']['forecast']['projected_16w_total']:,.2f}\n"
                    ]
                }
            ],
            "source": [line + "\n" for line in code_text.splitlines()]
        }
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(os.path.join(OUT_DIR, "progree_data_analytics_complete_notebook.ipynb"), "w") as nf:
    json.dump(notebook_obj, nf, indent=2)

print(json.dumps(summary_metrics, indent=2))