import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# --- CONFIGURATION ---
ABC_XYZ_FILE = 'abc_xyz_classification.csv'
DATA_FILE = 'full_data_with_clusters.csv'  # Your source file
TARGET_SEGMENTS = ['AX', 'AY', 'AZ', 'BX', 'BY']

# ==============================================================================
# 1. LOAD ABC/XYZ CLASSIFICATION (To know what to filter)
# ==============================================================================
print("1. Identifying target products (ABC/XYZ)...")

if os.path.exists(ABC_XYZ_FILE):
    abc_df = pd.read_csv(ABC_XYZ_FILE, dtype={'sku': str})
    
    # Filter only interested segments
    target_df = abc_df[abc_df['segment'].isin(TARGET_SEGMENTS)]
    target_skus = target_df['sku'].unique()
    
    print(f"   Total classified SKUs: {len(abc_df)}")
    print(f"   Target SKUs ({', '.join(TARGET_SEGMENTS)}): {len(target_skus)}")
else:
    print("ERROR: File 'abc_xyz_classification.csv' not found.")
    exit()

# ==============================================================================
# 2. LOAD SALES DATA (From CSV instead of Oracle)
# ==============================================================================
print("\n2. Loading sales data from local file...")

if os.path.exists(DATA_FILE):
    # Load raw data
    # dtype={'sku': str} ensures numeric SKUs don't lose leading zeros
    df = pd.read_csv(DATA_FILE, dtype={'sku': str})
    
    # Normalize column names
    df.columns = [col.lower() for col in df.columns]
    
    # Convert dates (day/month/year format as per your example)
    df['created_at'] = pd.to_datetime(df['created_at'], dayfirst=True)
    df['date'] = df['created_at'].dt.date
    df['qty_ordered'] = pd.to_numeric(df['qty_ordered'])
    
    # --- CRITICAL FILTERING ---
    # Keep only transactions of Target SKUs
    full_data = df[df['sku'].isin(target_skus)].copy()
    
    print(f"   Loaded data: {len(df)} rows.")
    print(f"   After filtering Targets: {len(full_data)} rows.")
else:
    print(f"ERROR: File '{DATA_FILE}' not found.")
    exit()

# ==============================================================================
# 3. FEATURE EXTRACTION (HAAR WAVELETS)
# ==============================================================================

if not full_data.empty:
    print("\n" + "="*70)
    print("HIERARCHICAL CLUSTERING WITH HAAR WAVELETS (TARGET SKUs)")
    print("="*70)

    # 3.1 Pivot (Time Series)
    print("Creating time series matrix...")
    daily_sales = full_data.groupby(['sku', 'date'])['qty_ordered'].sum().reset_index()
    ts_pivot = daily_sales.pivot(index='sku', columns='date', values='qty_ordered').fillna(0)
    print(f"   -> Matrix: {ts_pivot.shape[0]} SKUs x {ts_pivot.shape[1]} Days")

    # 3.2 Haar Transform
    print("Extracting Features (Haar Wavelet Energy)...")

    def extract_haar_features(series):
        data = series.values
        # Haar Decomposition Level 2
        coeffs = pywt.wavedec(data, 'haar', level=2)
        
        features = {
            'energy_approx': np.sum(np.square(coeffs[0])),    # Trend
            'energy_detail': np.sum(np.square(coeffs[1])) + np.sum(np.square(coeffs[2])), # Variation
            'sparsity': (data == 0).mean(), # % Days without sales
            'mean_vol': np.mean(data)
        }
        return pd.Series(features)

    sku_features = ts_pivot.apply(extract_haar_features, axis=1).reset_index()

    # Log Transform
    cols_to_log = ['energy_approx', 'energy_detail', 'mean_vol']
    sku_features[cols_to_log] = np.log1p(sku_features[cols_to_log])
    
    scaler = StandardScaler()
    feature_cols = ['energy_approx', 'energy_detail', 'sparsity']
    X_scaled = scaler.fit_transform(sku_features[feature_cols])

    # ==============================================================================
    # 4. HIERARCHICAL CLUSTERING (Dendrogram)
    # ==============================================================================
    print("\nStep 1: Dendrogram Analysis...")
    
    # Smart sampling if too large
    if len(X_scaled) > 5000:
        print(f"   (Using sample of 2000 SKUs for the plot)")
        idx_sample = np.random.choice(len(X_scaled), 2000, replace=False)
        X_plot = X_scaled[idx_sample]
    else:
        X_plot = X_scaled

    plt.figure(figsize=(12, 5))
    plt.title("Hierarchical Dendrogram (Targets AX..BY)")
    linkage_matrix = linkage(X_plot, method='ward', metric='euclidean')
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90., show_contracted=True)
    plt.ylabel("Distance")
    plt.show()

    # --- FINAL CLUSTERING ---
    n_clusters = 4  
    
    print(f"\nStep 2: Calculating {n_clusters} final clusters...")
    
    try:
        # Uses Euclidean metric on the Haar features we created
        h_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        sku_features['cluster'] = h_clustering.fit_predict(X_scaled)
        print("Hierarchical Clustering completed.")
        
    except MemoryError:
        print("Insufficient memory. Using K-Means.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        sku_features['cluster'] = kmeans.fit_predict(X_scaled)

    # ==============================================================================
    # 5. VISUALIZATION AND OUTPUT
    # ==============================================================================
    print("\nGenerating cluster profiles...")

    # Scatter Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sku_features, x='mean_vol', y='energy_detail', hue='cluster', palette='viridis', alpha=0.6)
    plt.title('New Clusters (Targets): Volume vs Instability')
    plt.xlabel('Mean Volume (Log)')
    plt.ylabel('Noise Energy (Log)')
    plt.show()

    # Time Pattern Plot
    ts_pivot['cluster'] = sku_features.set_index('sku')['cluster']
    cluster_patterns = ts_pivot.groupby('cluster').mean().T
    cluster_patterns.index = pd.to_datetime(cluster_patterns.index)

    plt.figure(figsize=(14, 6))
    for c in cluster_patterns.columns:
        plt.plot(cluster_patterns.index, cluster_patterns[c], label=f'Cluster {c}', linewidth=2)
    plt.title('Average Behavior by Cluster')
    plt.ylabel('Qty Sold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save
    # Add original segment for reference
    sku_features = sku_features.merge(target_df[['sku', 'segment']], on='sku', how='left')
    
    # Merge with original filtered data
    final_data = full_data.merge(sku_features[['sku', 'cluster']], on='sku', how='left')
    
    # Remove old 'cluster' column if it exists, to avoid conflict
    if 'cluster_x' in final_data.columns:
        final_data.rename(columns={'cluster_y': 'cluster'}, inplace=True)
        final_data.drop(columns=['cluster_x'], inplace=True)
        
    final_data.to_csv('target_data_clustered.csv', index=False)
    
    print("\nResult saved in: 'target_data_clustered.csv'")

else:
    print("No data to process.")