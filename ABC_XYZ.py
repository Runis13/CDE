import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. CONFIGURAÇÃO E LEITURA ---
file_name = 'full_data_with_clusters.csv'

print("📂 A carregar dados...")
if os.path.exists(file_name):
    # Carregar apenas colunas necessárias para poupar memória
    use_cols = ['created_at', 'sku', 'qty_ordered']
    # Se tiveres a coluna 'cluster', adiciona-a
    df_preview = pd.read_csv(file_name, nrows=1)
    if 'cluster' in df_preview.columns:
        use_cols.append('cluster')

    df = pd.read_csv(file_name, usecols=use_cols, dtype={'sku': str, 'qty_ordered': 'float32'})
    
    # Converter data e remover horas (Crucial para memória)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['date'] = df['created_at'].dt.date
    
    # Renomear se necessário
    if 'qty_ordered' not in df.columns and 'qty' in df.columns:
        df.rename(columns={'qty': 'qty_ordered'}, inplace=True)
else:
    print(f"❌ Ficheiro '{file_name}' não encontrado.")
    exit()

print(f"✅ Dados carregados: {len(df)} linhas.")

# --- 2. PREPARAÇÃO (MATRIZ SEMANAL) ---
print("🔄 A agregar dados (Semanal)...")

# Passo 1: Agregar por Dia (reduz milhões de linhas para milhares)
daily_sales = df.groupby(['date', 'sku'])['qty_ordered'].sum().reset_index()

# Passo 2: Pivot para Dia x SKU
daily_matrix = daily_sales.pivot(index='date', columns='sku', values='qty_ordered').fillna(0)

# Passo 3: Reamostrar para Semana (Essencial para XYZ estável)
daily_matrix.index = pd.to_datetime(daily_matrix.index)
weekly_matrix = daily_matrix.resample('W').sum()

print(f"   -> Matriz Semanal criada: {weekly_matrix.shape[0]} semanas x {weekly_matrix.shape[1]} SKUs")

# --- 3. CÁLCULO DE MÉTRICAS (ABC & XYZ) ---
print("📊 A calcular métricas...")
metrics = pd.DataFrame(index=weekly_matrix.columns)

# A: Volume Total
metrics['vol'] = weekly_matrix.sum()

# B: Coeficiente de Variação (CV)
metrics['mean'] = weekly_matrix.mean()
metrics['std'] = weekly_matrix.std()
metrics['cv'] = metrics['std'] / (metrics['mean'] + 1e-9)

# --- LÓGICA 1: ABC (PARETO CLÁSSICO) ---
# Ordenar por volume decrescente
metrics = metrics.sort_values('vol', ascending=False)
# Calcular % acumulada
metrics['cum_vol'] = metrics['vol'].cumsum() / metrics['vol'].sum()
# Classificar A (80%), B (15%), C (5%)
metrics['abc'] = metrics['cum_vol'].apply(lambda x: 'A' if x<=0.80 else ('B' if x<=0.95 else 'C'))

# --- LÓGICA 2: XYZ RELATIVO (SOLUÇÃO PARA A FALTA DE X) ---
# Em vez de fixar em 0.5, usamos os percentis dos TEUS dados.
# X = Os 30% produtos mais estáveis
# Y = Os 40% seguintes
# Z = Os 30% mais instáveis

limit_x = metrics['cv'].quantile(0.30)
limit_y = metrics['cv'].quantile(0.70)

print(f"\n⚖️  Limites XYZ Ajustados à Realidade:")
print(f"   X (Estável): CV <= {limit_x:.2f}")
print(f"   Y (Médio):   CV entre {limit_x:.2f} e {limit_y:.2f}")
print(f"   Z (Instável): CV > {limit_y:.2f}")

def classify_relative_xyz(cv_val):
    if cv_val <= limit_x: return 'X'
    elif cv_val <= limit_y: return 'Y'
    else: return 'Z'

metrics['xyz'] = metrics['cv'].apply(classify_relative_xyz)
metrics['segment'] = metrics['abc'] + metrics['xyz']

# --- 4. VISUALIZAÇÃO ---
print("\n📈 A gerar Matriz Estratégica...")

# Tabela cruzada para o Heatmap
pivot_table = metrics.reset_index().pivot_table(
    index='abc', columns='xyz', values='vol', aggfunc='count'
).fillna(0).astype(int)

# Garantir ordem correta
pivot_table = pivot_table.reindex(index=['A', 'B', 'C'], columns=['X', 'Y', 'Z'])

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlGnBu', linewidths=1, linecolor='black')
plt.title(f"Matriz ABC-XYZ (Ajustada: X <= {limit_x:.1f} CV)")
plt.xlabel("Estabilidade (XYZ Relativo)")
plt.ylabel("Volume (ABC)")

# Adicionar totais nas células
for y, abc in enumerate(['A', 'B', 'C']):
    for x, xyz in enumerate(['X', 'Y', 'Z']):
        if xyz in pivot_table.columns:
            val = pivot_table.loc[abc, xyz]
            if val > 0:
                plt.text(x + 0.5, y + 0.8, "SKUs", ha='center', va='center', color='black', fontsize=8)

plt.tight_layout()
plt.show()

# Guardar classificação
metrics.to_csv('abc_xyz_classification.csv')
print("\n✅ Classificação guardada em 'abc_xyz_classification.csv'.")
print("   Podes cruzar isto com o teu Forecast GRU!")