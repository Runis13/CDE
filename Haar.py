import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pywt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

# --- CONFIGURATION ---
ABC_XYZ_FILE = 'abc_xyz_classification.csv'
DATA_FILE = 'sales_data_up_to_2025_04_23.csv'
TARGET_SEGMENTS = ['AX', 'AY', 'AZ', 'BX', 'BY']

# ==============================================================================
# 1. LOAD CLASSIFICATION (ABC/XYZ)
# ==============================================================================
print("1. Identifying target products (ABC/XYZ)...")

if os.path.exists(ABC_XYZ_FILE):
    abc_df = pd.read_csv(ABC_XYZ_FILE, dtype={'sku': str})
    target_df = abc_df[abc_df['segment'].isin(TARGET_SEGMENTS)]
    target_skus = target_df['sku'].unique()
    print(f"   Total Targets ({', '.join(TARGET_SEGMENTS)}): {len(target_skus)}")
else:
    print("ERROR: File 'abc_xyz_classification.csv' not found.")
    exit()

# ==============================================================================
# 2. LOAD SALES DATA
# ==============================================================================
print("\n2. Loading sales data from local file...")

if os.path.exists(DATA_FILE):
    df = pd.read_csv(DATA_FILE, dtype={'sku': str})
    df.columns = [col.lower() for col in df.columns]
    df['created_at'] = pd.to_datetime(df['created_at'], dayfirst=True, format='mixed')
    df['date'] = df['created_at'].dt.date
    df['qty_ordered'] = pd.to_numeric(df['qty_ordered'])
    
    full_data = df[df['sku'].isin(target_skus)].copy()
    print(f"   Transactions loaded for Targets: {len(full_data)}")
else:
    print(f"ERROR: File '{DATA_FILE}' not found.")
    exit()

# ==============================================================================
# 3. FEATURE EXTRACTION (HAAR WAVELETS)
# ==============================================================================

if not full_data.empty:
    print("\n" + "="*70)
    print("HIERARCHICAL CLUSTERING WITH QUALITY ANALYSIS")
    print("="*70)

    # 3.1 Pivot
    daily_sales = full_data.groupby(['sku', 'date'])['qty_ordered'].sum().reset_index()
    ts_pivot = daily_sales.pivot(index='sku', columns='date', values='qty_ordered').fillna(0)

    # 3.2 Haar Transform
    print("Extracting Features (Haar Wavelet Energy)...")

    def extract_haar_features(series):
        data = series.values
        coeffs = pywt.wavedec(data, 'haar', level=2)
        features = {
            'energy_approx': np.sum(np.square(coeffs[0])),    # Trend
            'energy_detail': np.sum(np.square(coeffs[1])) + np.sum(np.square(coeffs[2])), # Variation
            'sparsity': (data == 0).mean(), # % Zeros
            'mean_vol': np.mean(data)
        }
        return pd.Series(features)

    sku_features = ts_pivot.apply(extract_haar_features, axis=1).reset_index()

    cols_to_log = ['energy_approx', 'energy_detail', 'mean_vol']
    sku_features[cols_to_log] = np.log1p(sku_features[cols_to_log])
    
    scaler = StandardScaler()
    feature_cols = ['energy_approx', 'energy_detail', 'sparsity']
    X_scaled = scaler.fit_transform(sku_features[feature_cols])

    # ==============================================================================
    # 4. HIERARCHICAL CLUSTERING & ANALYSIS
    # ==============================================================================
    # Justification of the Method
    print("\nCLUSTERING METHOD JUSTIFICATION:")
    print("   * Algorithm: Agglomerative Clustering (Hierarchical)")
    print("   * Linkage Metric: 'Ward'")
    print("     -> Ward's method minimizes the sum of squared differences within each cluster.")
    print("     -> Attempts to create compact and spherical clusters, ideal for client/product segmentation.")
    print("   * Distance Measure: Euclidean (Compatible with Ward)")

    # Dendrogram
    print("\nDrawing Dendrogram...")
    if len(X_scaled) > 5000:
        idx_sample = np.random.choice(len(X_scaled), 2000, replace=False)
        X_plot = X_scaled[idx_sample]
    else:
        X_plot = X_scaled

    plt.figure(figsize=(12, 5))
    plt.title("Hierarchical Dendrogram (Ward Method)")
    linkage_matrix = linkage(X_plot, method='ward', metric='euclidean')
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90., show_contracted=True)
    plt.ylabel("Ward Distance (Sum of Squares)")
    plt.show()

    # Final Clustering
    n_clusters = 3
    print(f"\nCalculating {n_clusters} clusters...")
    
    try:
        h_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        labels = h_clustering.fit_predict(X_scaled)
        sku_features['cluster'] = labels
    except MemoryError:
        print("Insufficient memory. Using K-Means.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        sku_features['cluster'] = kmeans.fit_predict(X_scaled)

    # ==============================================================================
    # 5. CLUSTER QUALITY ANALYSIS
    # ==============================================================================
    print("\nSTATISTICAL ANALYSIS OF GROUPS:")
    print("-" * 60)
    
    # 1. Count
    counts = sku_features['cluster'].value_counts().sort_index()
    print("1. SKU Distribution by Cluster:")
    print(counts)
    
    # 2. Quality Metrics (Silhouette)
    # Silhouette measures how similar an object is to its own cluster compared to other clusters.
    # The value ranges from -1 to 1. The closer to 1, the better.
    if len(X_scaled) < 16000: # Silhouette is heavy for giant datasets
        sil_score = silhouette_score(X_scaled, sku_features['cluster'])
        print(f"\n2. Global Silhouette Score: {sil_score:.3f}")
        print("   (> 0.5 indicates strong structure; > 0.25 indicates reasonable structure)")
    
    # Davies-Bouldin (Lower is better)
    db_score = davies_bouldin_score(X_scaled, sku_features['cluster'])
    print(f"   Davies-Bouldin Score: {db_score:.3f} (Lower value means better separation)")

    # 3. Centroids (What defines each group?)
    print("\n3. Average Profile of each Cluster (Real Values - No Log):")
    # Convert logs back to real scale for human interpretation
    summary_df = sku_features.groupby('cluster')[['mean_vol', 'energy_detail', 'sparsity']].mean()
    summary_df['mean_vol_real'] = np.expm1(summary_df['mean_vol']) # Reverse Log
    summary_df['noise_real'] = np.expm1(summary_df['energy_detail'])
    
    print(summary_df[['mean_vol_real', 'noise_real', 'sparsity']].round(4))
    print("-" * 60)
    print("LEGEND:")
    print(" * Mean Vol Real: Average daily sales volume")
    print(" * Noise Real: Fluctuation energy (Haar Detail) - Measures instability")
    print(" * Sparsity: % of days the product did NOT sell (0.0 = sold always, 0.9 = rare)")
    print("-" * 60)

    # Plots (Kept for visual validation)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sku_features, x='mean_vol', y='energy_detail', hue='cluster', palette='viridis', alpha=0.6)
    plt.title('Clusters: Volume vs Instability')
    plt.xlabel('Average Volume (Log)')
    plt.ylabel('Noise Energy (Log)')
    plt.show()

    # Save
    sku_features = sku_features.merge(target_df[['sku', 'segment']], on='sku', how='left')
    final_data = full_data.merge(sku_features[['sku', 'cluster']], on='sku', how='left')
    
    if 'cluster_x' in final_data.columns:
        final_data.rename(columns={'cluster_y': 'cluster'}, inplace=True)
        final_data.drop(columns=['cluster_x'], inplace=True)
        
    final_data.to_csv('target_data_clustered.csv', index=False)
    print("\nResult saved in: 'target_data_clustered.csv'")

else:
    print("No data to process.")