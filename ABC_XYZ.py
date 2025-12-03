import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. CONFIGURATION AND LOADING ---
file_name = 'sales_data_up_to_2025_04_23.csv'

print("Loading data...")
if os.path.exists(file_name):
    # Load only necessary columns to save memory
    use_cols = ['created_at', 'sku', 'qty_ordered']
    # If you have the 'cluster' column, add it
    df_preview = pd.read_csv(file_name, nrows=1)
    if 'cluster' in df_preview.columns:
        use_cols.append('cluster')

    df = pd.read_csv(file_name, usecols=use_cols, dtype={'sku': str, 'qty_ordered': 'float32'})
    
    # Convert date and remove hours (Crucial for memory)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    
    # Rename if necessary
    if 'qty_ordered' not in df.columns and 'qty' in df.columns:
        df.rename(columns={'qty': 'qty_ordered'}, inplace=True)
else:
    print(f"File '{file_name}' not found.")
    exit()

print(f"Data loaded: {len(df)} rows.")

# --- 2. PREPARATION (WEEKLY MATRIX) ---
print("Aggregating data (Weekly)...")

# Step 1: Aggregate by Day (reduces millions of rows to thousands)
daily_sales = df.groupby(['date', 'sku'])['qty_ordered'].sum().reset_index()

# Step 2: Pivot to Day x SKU
daily_matrix = daily_sales.pivot(index='date', columns='sku', values='qty_ordered').fillna(0)

# Step 3: Resample to Week (Essential for stable XYZ)
daily_matrix.index = pd.to_datetime(daily_matrix.index)
weekly_matrix = daily_matrix.resample('W').sum()

print(f" -> Weekly Matrix created: {weekly_matrix.shape[0]} weeks x {weekly_matrix.shape[1]} SKUs")

# --- 3. METRICS CALCULATION (ABC & XYZ) ---
print("Calculating metrics...")
metrics = pd.DataFrame(index=weekly_matrix.columns)

# A: Total Volume
metrics['vol'] = weekly_matrix.sum()

# B: Coefficient of Variation (CV)
metrics['mean'] = weekly_matrix.mean()
metrics['std'] = weekly_matrix.std()
metrics['cv'] = metrics['std'] / (metrics['mean'] + 1e-9)

# --- LOGIC 1: ABC (CLASSIC PARETO) ---
# Sort by decreasing volume
metrics = metrics.sort_values('vol', ascending=False)
# Calculate cumulative %
metrics['cum_vol'] = metrics['vol'].cumsum() / metrics['vol'].sum()
# Classify A (80%), B (15%), C (5%)
metrics['abc'] = metrics['cum_vol'].apply(lambda x: 'A' if x<=0.80 else ('B' if x<=0.95 else 'C'))

# --- LOGIC 2: RELATIVE XYZ (SOLUTION FOR LACK OF X) ---
# Instead of fixing at 0.5, we use the percentiles of YOUR data.
# X = The 30% most stable products
# Y = The following 40%
# Z = The 30% most unstable

limit_x = metrics['cv'].quantile(0.30)
limit_y = metrics['cv'].quantile(0.70)

print(f"\nXYZ Limits Adjusted to Reality:")
print(f"   X (Stable): CV <= {limit_x:.2f}")
print(f"   Y (Medium):   CV between {limit_x:.2f} and {limit_y:.2f}")
print(f"   Z (Unstable): CV > {limit_y:.2f}")

def classify_relative_xyz(cv_val):
    if cv_val <= limit_x: return 'X'
    elif cv_val <= limit_y: return 'Y'
    else: return 'Z'

metrics['xyz'] = metrics['cv'].apply(classify_relative_xyz)
metrics['segment'] = metrics['abc'] + metrics['xyz']

# --- 4. VISUALIZATION ---
print("\nGenerating Strategic Matrix...")

# Crosstab for Heatmap
pivot_table = metrics.reset_index().pivot_table(
    index='abc', columns='xyz', values='vol', aggfunc='count'
).fillna(0).astype(int)

# Ensure correct order
pivot_table = pivot_table.reindex(index=['A', 'B', 'C'], columns=['X', 'Y', 'Z'])

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', linewidths=1, linecolor='black')
plt.title(f"ABC-XYZ Matrix (Adjusted: X <= {limit_x:.1f} CV)")
plt.xlabel("Stability (Relative XYZ)")
plt.ylabel("Volume (ABC)")

# Add totals to cells
for y, abc in enumerate(['A', 'B', 'C']):
    for x, xyz in enumerate(['X', 'Y', 'Z']):
        if xyz in pivot_table.columns:
            val = pivot_table.loc[abc, xyz]
            if val > 0:
                plt.text(x + 0.5, y + 0.8, "SKUs", ha='center', va='center', color='black', fontsize=8)

plt.tight_layout()
plt.show()

# Save classification
metrics.to_csv('abc_xyz_classification.csv')
print("\nClassification saved in 'abc_xyz_classification.csv'.")
print("  You can cross this with your GRU Forecast!")