import sys
import os
import glob
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Ajustar caminho do projeto
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dyfo.config import DyFOConfig
from dyfo.core.dyfo_module import DyFOModule

# Tickers seguem a ordem do módulo original
TICKERS_30 = [
    "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "JPM", "GS", "MA", "BRK-B",
    "JNJ", "UNH", "LLY", "AMZN", "TSLA", "HD", "PG", "KO", "XOM", "CVX",
    "CAT", "BA", "RTX", "META", "GOOGL", "DIS", "LIN", "APD", "NEE", "DUK", "PLD"
]

def find_latest_model():
    """Encontra o .pt mais recente do TGN salvo."""
    paths = glob.glob("results/link_pred_tgn_*/best_model.pt")
    if not paths:
        raise FileNotFoundError("Nenhum best_model.pt do TGN encontrado em results/")
    latest_model = max(paths, key=os.path.getmtime)
    return latest_model

def plot_real_heatmap_from_model(model_path):
    print(f"Carregando modelo: {model_path}")
    
    config = DyFOConfig()
    model = DyFOModule(config, num_nodes=len(TICKERS_30))
    
    # Carrega os pesos salvos do experimento
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    # Como não estamos refazendo o loop completo do DataLoader aqui (que levaria minutos),
    # vamos utilizar o estado dos pesos de atenção no exato momento pós-carregamento. 
    # Para capturar dados REAIS e PRECISOS de uma data específica (ex: 2020-03-20),
    # idealmente interceptaríamos o DataLoader no arquivo de teste iterativo. 
    # Aqui renderizarei os pesos gravados no último estado salvo da rede.
    
    encoder = model.tgn
    
    if not hasattr(encoder, 'last_alpha'):
        print("Erro: O modelo carregado não salvou last_alpha. Rode o treino novamente usando a definição de TGNEncoder atualizada.")
        return

    src = encoder.last_src.numpy()
    tgt = encoder.last_tgt.numpy()
    alpha = encoder.last_alpha.numpy()
    
    # Inicializa matriz N x N
    attn_matrix = np.zeros((len(TICKERS_30), len(TICKERS_30)))
    
    for s, t, a in zip(src, tgt, alpha):
        attn_matrix[s, t] = a
        
    # Normalização por linha (softmax-like) 
    row_sums = attn_matrix.sum(axis=1, keepdims=True)
    # Evita divisão por zero
    row_sums[row_sums == 0] = 1.0
    attn_matrix = attn_matrix / row_sums

    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, xticklabels=TICKERS_30, yticklabels=TICKERS_30, 
                cmap="YlOrRd", cbar_kws={'label': 'Attention Weight ($a_{ij}$)'})
    
    plt.title('Real TGN Attention Heatmap (Latest Snapshot)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Target Node ($j$)", fontsize=12)
    plt.ylabel("Source Node ($i$)", fontsize=12)
    
    out_path = "results/plots/attention_heatmap_real.png"
    os.makedirs("results/plots", exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap real salvo em: {out_path}")

if __name__ == "__main__":
    try:
        model_path = find_latest_model()
        plot_real_heatmap_from_model(model_path)
    except Exception as e:
        print(f"Falha ao gerar o heatmap real: {str(e)}")
