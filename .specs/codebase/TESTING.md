# 05 — Protocolo de Avaliação Experimental (Rev 2)

> Documento de referência para todos os scripts de bootstrap eval.
> Atualizado em: 2026-04-19 — Rev 2 inclui TGAT, métricas financeiras e 10k-bootstrap.

---

## Scripts de Avaliação

| Script                                 | Propósito                                            | Status    |
|----------------------------------------|------------------------------------------------------|-----------|
| `run_bootstrap_eval_temporal_kg_rev2.py`| Runner principal: TGAT/Baselines + Scale + Ablation | ✅ Ativo   |
| `compute_mdd_turnover_full.py`         | Cálculo pós-hoc de métricas financeiras de risco     | ✅ Ativo   |
| `run_bootstrap_eval_v5.py`              | H4 legacy (TGN 30 ativos)                          | ⚠️ Legado |

---

## Variantes de Modelo

| Variante      | Classe               | LR    | Patience | Tipo                          |
|---------------|----------------------|-------|----------|-------------------------------|
| `tgat`        | `TGATEncoder`        | 1e-3  | 5        | Temporal (Stateless Attention) |
| `tgn`         | `TGNEncoder`         | 1e-3  | 5        | Temporal (Memória GRU)        |
| `temporal_kg` | `TemporalKGEncoder`  | 1e-3  | 5        | Temporal (KG interpretável)   |
| `roland`      | `ROLANDLikeEncoder`  | 1e-3  | 5        | Snapshot + EMA               |
| `gat_static`  | `GATStaticEncoder`   | 1e-3  | 5        | GAT estático                 |

---

## Universos de Ativos (`--n_tickers`)

| Tamanho | Esparsificação CORR         | Pares  | Status           |
|---------|-----------------------------|--------|------------------|
| 30      | Threshold `|ρ| > 0.3`        | 435    | Estável          |
| 50      | Threshold `|ρ| > 0.3`        | 1.225  | **Config. Ótima**|
| 100     | **TMFG** (Stress Test)      | 4.950  | Experimental     |

---

## Parâmetros de Protocolo

| Parâmetro      | Padrão  | Justificativa                             |
|----------------|---------|-------------------------------------------|
| `n_tickers`    | 50      | Configuração ótima de escala              |
| `n_bootstrap`  | 10000   | Robustez estatística máxima               |
| `block_size`   | 10      | Captura dependências de 2 semanas         |
| `metrics`      | Sharpe, MDD, Turnover | Avaliação financeira completa |

---

## Testes Estatísticos

| Teste                      | Nível        | Objetivo                                      |
|----------------------------|--------------|-----------------------------------------------|
| **Wilcoxon signed-rank**   | Window-level | Comparar medianas de Sharpe entre modelos     |
| **Diebold-Mariano**        | Time-series  | Comparar erros de predição ao longo do tempo  |
| **Holm-Bonferroni**        | Protocolo    | Controlar FWER em testes de múltiplas variantes|
| **Block Bootstrap (10k)**  | Janela       | Estimar IC 95% para métricas de risco (MDD)   |

---

## Exemplos de Uso

```bash
# Avaliação completa de TGAT com 50 ativos
python scripts/run_bootstrap_eval_temporal_kg_rev2.py \
  --variants tgat roland \
  --n_tickers 50 \
  --n_bootstrap 10000

# Cálculo de metrics de risco detalhadas
python scripts/compute_mdd_turnover_full.py \
  --model_path results/tgat_best.pt
```
