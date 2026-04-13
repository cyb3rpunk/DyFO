import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

# Evita o uso de interface gráfica interativa bloqueante
import matplotlib
matplotlib.use('Agg')

OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_radar_chart():
    """
    Gera o gráfico de radar (Spyder Chart) comparando a robustez dos modelos:
    TGN, ROLAND e GAT-Static nas métricas principais.
    Dados extraídos do doc/EXPERIMENT_LOG.md, v0.9 Block Bootstrap.
    """
    print("Gerando Radar Chart...")
    
    # Métricas e resultados obtidos no teste
    metrics = ['R² (Correl)', 'Spearman \u03c1', 'F1-Score', '1 - MAE', 'Sharpe (GMVP)']
    
    # Para o MAE, convertemos para uma métrica "maior é melhor". 
    # MAE TGN=0.053 -> 1-MAE=0.947 / ROLAND=0.090 -> 0.910 / GAT=0.078 -> 0.922 
    # Normalizando o Sharpe Ratio pelo máximo (aproximado 3.0) para caber no range [0-1]
    # Sharpe TGN=2.437 -> 0.812 / ROLAND=1.493 -> 0.497 / GAT=2.354 -> 0.784
    
    tgn_vals =    [0.789, 0.939, 0.766, 1 - 0.053, 2.437 / 3.0]
    gat_vals =    [0.562, 0.891, 0.564, 1 - 0.078, 2.354 / 3.0]
    roland_vals = [0.354, 0.724, 0.447, 1 - 0.090, 1.493 / 3.0]

    # Fechando o polígono repetindo o primeiro valor
    tgn_vals += tgn_vals[:1]
    gat_vals += gat_vals[:1]
    roland_vals += roland_vals[:1]
    
    angles = [n / float(len(metrics)) * 2 * np.pi for n in range(len(metrics))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Ajustar eixo X
    plt.xticks(angles[:-1], metrics, color='black', size=12, fontweight='bold')
    
    # Ajustar eixo Y e grid
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.05)

    # Cores no padrão publicável
    c_tgn = '#1f77b4' # Azul
    c_gat = '#ff7f0e' # Laranja
    c_roland = '#2ca02c' # Verde

    # Plotando TGN
    ax.plot(angles, tgn_vals, linewidth=2, linestyle='solid', label='DyFO (TGN)', color=c_tgn)
    ax.fill(angles, tgn_vals, alpha=0.2, color=c_tgn)

    # Plotando GAT-Static
    ax.plot(angles, gat_vals, linewidth=2, linestyle='dashed', label='GAT-Static', color=c_gat)
    ax.fill(angles, gat_vals, alpha=0.1, color=c_gat)

    # Plotando ROLAND
    ax.plot(angles, roland_vals, linewidth=2, linestyle='dotted', label='ROLAND', color=c_roland)
    ax.fill(angles, roland_vals, alpha=0.1, color=c_roland)

    # Adicionando a legenda
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), prop={'size': 12})
    
    plt.title('Robustez do Modelo (Ablation B16)', size=16, y=1.1, fontweight='bold')
    
    # Salvando imagem
    output_path = os.path.join(OUTPUT_DIR, "radar_chart_b16.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radar Chart salvo em {output_path}")


def generate_mock_attention_heatmap():
    """
    Gera um Heatmap de Atenção simulado para o fluxo do TGN.
    Isso serve de teste visual enquanto o hook de pesos de atenção não roda em um dataset completo (Março 2020).
    A implementação do TGNEncoder já grava `last_alpha` no forward, e o script final usará valores dinâmicos.
    """
    print("Gerando Attention Heatmap (Exemplo)...")
    
    # 20 nós simulando tickers do S&P 500 para março de 2020 (Covid Crash)
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "GS", "MA", "JNJ", 
               "UNH", "PG", "KO", "XOM", "CVX", "CAT", "BA", "NEE", "DUK", "PFE"]
    
    n = len(tickers)
    
    # Matriz de atenção N x N pseudo-realista para demonstração
    # Geralmente a diagonal tem grande peso, e setores similares têm peso
    np.random.seed(42)
    attention_matrix = np.random.rand(n, n) * 0.2
    
    # Enfatizando self-loops
    np.fill_diagonal(attention_matrix, np.random.rand(n) * 0.5 + 0.5)
    
    # Enfatizando o crash tech (as techs prestando atenção umas nas outras)
    tech_indices = [0, 1, 2, 3, 4, 5]
    for i in tech_indices:
        for j in tech_indices:
            if i != j:
                attention_matrix[i, j] += 0.3
    
    # Enfatizando o fator energia/finanças (JPM, GS, XOM, CVX) e como BA agiu
    attention_matrix[6, 7] += 0.4
    attention_matrix[13, 14] += 0.4
    attention_matrix[16, :] += 0.2 # Boeing broadcast attention
    
    # Normaliza as linhas baseada em softmax aproximado (como o TGN faria localmente)
    row_sums = attention_matrix.sum(axis=1)
    attention_matrix = attention_matrix / row_sums[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_matrix, xticklabels=tickers, yticklabels=tickers, cmap="YlOrRd",
                annot=False, cbar_kws={'label': 'Attention Weight ($a_{ij}$)'})
    
    plt.title('TGN Temporal Attention Heatmap (Simulated Snapshot: Mar 2020)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Target Node ($j$)", fontsize=12)
    plt.ylabel("Source Node ($i$)", fontsize=12)
    
    output_path = os.path.join(OUTPUT_DIR, "attention_heatmap_mock.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap salvo em {output_path}")


if __name__ == "__main__":
    generate_radar_chart()
    generate_mock_attention_heatmap()
    
    print("\nVisualizações concluídas. Verifique a pasta 'results/plots/'.")
    print("Para um heatmap da atenção real sobre o dataset, é necessário carregar o TGN best_model_*.pt e injetar os eventos correspondentes à época de Março de 2020.")
