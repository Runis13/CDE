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

# --- CONFIGURAÇÃO ---
ABC_XYZ_FILE = 'abc_xyz_classification.csv'
DATA_FILE = 'full_data_with_clusters.csv'  # O teu ficheiro fonte
TARGET_SEGMENTS = ['AX', 'AY', 'AZ', 'BX', 'BY']

# ==============================================================================
# 1. CARREGAR CLASSIFICAÇÃO ABC/XYZ (Para saber quem filtrar)
# ==============================================================================
print("🔍 1. A identificar produtos alvo (ABC/XYZ)...")

if os.path.exists(ABC_XYZ_FILE):
    abc_df = pd.read_csv(ABC_XYZ_FILE, dtype={'sku': str})
    
    # Filtrar apenas os segmentos de interesse
    target_df = abc_df[abc_df['segment'].isin(TARGET_SEGMENTS)]
    target_skus = target_df['sku'].unique()
    
    print(f"   ✅ Total SKUs classificados: {len(abc_df)}")
    print(f"   🎯 SKUs Alvo ({', '.join(TARGET_SEGMENTS)}): {len(target_skus)}")
else:
    print("❌ ERRO: Ficheiro 'abc_xyz_classification.csv' não encontrado.")
    exit()

# ==============================================================================
# 2. CARREGAR DADOS DE VENDAS (Do CSV em vez da Oracle)
# ==============================================================================
print("\n📂 2. A carregar dados de vendas do ficheiro local...")

if os.path.exists(DATA_FILE):
    # Carregar dados brutos
    # dtype={'sku': str} garante que SKUs numéricos não perdem zeros à esquerda
    df = pd.read_csv(DATA_FILE, dtype={'sku': str})
    
    # Normalizar nomes das colunas
    df.columns = [col.lower() for col in df.columns]
    
    # Converter datas (formato dia/mês/ano conforme o teu exemplo)
    df['created_at'] = pd.to_datetime(df['created_at'], dayfirst=True)
    df['date'] = df['created_at'].dt.date
    df['qty_ordered'] = pd.to_numeric(df['qty_ordered'])
    
    # --- FILTRAGEM CRÍTICA ---
    # Manter apenas as transações dos SKUs Alvo
    full_data = df[df['sku'].isin(target_skus)].copy()
    
    print(f"   ✅ Dados carregados: {len(df)} linhas.")
    print(f"   ✂️  Após filtrar Targets: {len(full_data)} linhas.")
else:
    print(f"❌ ERRO: Ficheiro '{DATA_FILE}' não encontrado.")
    exit()

# ==============================================================================
# 3. FEATURE EXTRACTION (HAAR WAVELETS)
# ==============================================================================

if not full_data.empty:
    print("\n" + "="*70)
    print("CLUSTERING HIERÁRQUICO COM HAAR WAVELETS (TARGET SKUs)")
    print("="*70)

    # 3.1 Pivot (Séries Temporais)
    print("🔄 Criando matriz de séries temporais...")
    daily_sales = full_data.groupby(['sku', 'date'])['qty_ordered'].sum().reset_index()
    ts_pivot = daily_sales.pivot(index='sku', columns='date', values='qty_ordered').fillna(0)
    print(f"   -> Matriz: {ts_pivot.shape[0]} SKUs x {ts_pivot.shape[1]} Dias")

    # 3.2 Haar Transform
    print("🌊 Extraindo Features (Haar Wavelet Energy)...")

    def extract_haar_features(series):
        data = series.values
        # Decomposição Haar Nível 2
        coeffs = pywt.wavedec(data, 'haar', level=2)
        
        features = {
            'energy_approx': np.sum(np.square(coeffs[0])),    # Tendência
            'energy_detail': np.sum(np.square(coeffs[1])) + np.sum(np.square(coeffs[2])), # Variação
            'sparsity': (data == 0).mean(), # % Dias sem vendas
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
    # 4. HIERARCHICAL CLUSTERING (Dendrograma)
    # ==============================================================================
    print("\n🌳 Passo 1: Análise de Dendrograma...")
    
    # Amostragem inteligente se for muito grande
    if len(X_scaled) > 5000:
        print(f"   (A usar amostra de 2000 SKUs para o gráfico)")
        idx_sample = np.random.choice(len(X_scaled), 2000, replace=False)
        X_plot = X_scaled[idx_sample]
    else:
        X_plot = X_scaled

    plt.figure(figsize=(12, 5))
    plt.title("Dendrograma Hierárquico (Targets AX..BY)")
    linkage_matrix = linkage(X_plot, method='ward', metric='euclidean')
    dendrogram(linkage_matrix, truncate_mode='lastp', p=30, leaf_rotation=90., show_contracted=True)
    plt.ylabel("Distância")
    plt.show()

    # --- CLUSTERING FINAL ---
    n_clusters = 4  
    
    print(f"\n🚀 Passo 2: A calcular {n_clusters} clusters finais...")
    
    try:
        # Usa a métrica Euclidiana nas features de Haar que criámos
        h_clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
        sku_features['cluster'] = h_clustering.fit_predict(X_scaled)
        print("✅ Clustering Hierárquico concluído.")
        
    except MemoryError:
        print("⚠️ Memória insuficiente. A usar K-Means.")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        sku_features['cluster'] = kmeans.fit_predict(X_scaled)

    # ==============================================================================
    # 5. VISUALIZAÇÃO E OUTPUT
    # ==============================================================================
    print("\n📊 A gerar perfis dos clusters...")

    # Gráfico de Dispersão
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sku_features, x='mean_vol', y='energy_detail', hue='cluster', palette='viridis', alpha=0.6)
    plt.title('Novos Clusters (Targets): Volume vs Instabilidade')
    plt.xlabel('Volume Médio (Log)')
    plt.ylabel('Energia de Ruído (Log)')
    plt.show()

    # Gráfico de Padrão Temporal
    ts_pivot['cluster'] = sku_features.set_index('sku')['cluster']
    cluster_patterns = ts_pivot.groupby('cluster').mean().T
    cluster_patterns.index = pd.to_datetime(cluster_patterns.index)

    plt.figure(figsize=(14, 6))
    for c in cluster_patterns.columns:
        plt.plot(cluster_patterns.index, cluster_patterns[c], label=f'Cluster {c}', linewidth=2)
    plt.title('Comportamento Médio por Cluster')
    plt.ylabel('Qty Vendida')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Guardar
    # Adicionar o segmento original para referência
    sku_features = sku_features.merge(target_df[['sku', 'segment']], on='sku', how='left')
    
    # Merge com os dados originais filtrados
    final_data = full_data.merge(sku_features[['sku', 'cluster']], on='sku', how='left')
    
    # Removemos a coluna 'cluster' antiga se existir, para não haver conflito
    if 'cluster_x' in final_data.columns:
        final_data.rename(columns={'cluster_y': 'cluster'}, inplace=True)
        final_data.drop(columns=['cluster_x'], inplace=True)
        
    final_data.to_csv('target_data_clustered.csv', index=False)
    
    print("\n💾 Resultado guardado em: 'target_data_clustered.csv'")

else:
    print("❌ Não há dados para processar.")