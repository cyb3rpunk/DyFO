# 05 — Protocolo de Avaliação Experimental (Rev 2)

> Documento de referência para todos os scripts de bootstrap eval.
> Atualizado em: 2026-04-16 — Rev 2 inclui `--variants`, `--n_tickers`, `--ablation`.

---

## Scripts de Avaliação

| Script | Propósito | Status |
|--------|-----------|--------|
| `run_bootstrap_eval_v5.py` | H4 original: TGN ≥ ROLAND ≥ 70% janelas | ✅ Congelado — não editar |
| `run_bootstrap_eval_ra_htgn.py` | BL-17: RA-HTGN vs TGN/ROLAND/GAT | ✅ Congelado |
| `run_bootstrap_eval_temporal_kg.py` | BL-18 rev 0 (windows sobrepostas — deprecado) | ⚠️ Deprecado |
| `run_bootstrap_eval_temporal_kg_rev1.py` | BL-18 rev 1 (windows não-sobrepostas, LR correto) | ✅ Estável |
| `run_bootstrap_eval_temporal_kg_rev2.py` | BL-18 rev 2 (`--variants`, `--n_tickers`, `--ablation`) | 🆕 Atual |

**Regra:** apenas `run_bootstrap_eval_temporal_kg_rev2.py` deve ser usado em novos experimentos.

---

## Variantes de Modelo

| Variante | Classe | LR | Patience | Tipo |
|----------|--------|----|----------|------|
| `tgn` | `TGNWrapper` → `TGNEncoder` | 1e-3 | 5 | Temporal (memória GRU) |
| `ra_htgn` | `RAHTGNEncoder` | 1e-3 | 5 | Temporal (relação-aware) |
| `temporal_kg` | `TemporalKGEncoder` | 1e-3 | 5 | Temporal (KG interpretável) |
| `roland` | `ROLANDLikeEncoder` | 1e-3 | 5 | Snapshot mensal + EMA |
| `gat_static` | `GATStaticEncoder` | 1e-3 | 5 | GAT estático (sem memória) |

---

## Universos de Ativos (`--n_tickers`)

Definidos em `dyfo/core/ticker_registry.py`.

| Tamanho | Lista | Esparsificação CORR | Pares |
|---------|-------|---------------------|-------|
| 30 | `TICKERS_30` | Threshold `|ρ| > 0.3` | 435 |
| 50 | `TICKERS_50` | Threshold `|ρ| > 0.3` | 1.225 |
| 100 | `TICKERS_100` | **TMFG** (51-200, ver spec/02_graph_spec.md) | 4.950 |

> ⚠️ **TMFG para 100 ativos:** a implementação atual não possui TMFG — usa threshold simples como fallback.
> Resultados com `--n_tickers 100` devem ser reportados com essa ressalva.

---

## Modo de Ablação de Tipos de Aresta (`--ablation`)

Isola a contribuição de cada tipo de aresta no grafo financeiro treinando
o mesmo `--ablation_variant` com apenas um subconjunto de arestas ativo.

### Subconjuntos

| Modo | Subconjuntos treinados |
|------|------------------------|
| `basic` | `CORR_only`, `SECT_only`, `FACT_only` |
| `full` | Todos os 7 subconjuntos (3 individuais + 3 pares + `all_edges`) |

### Como funciona o mascaramento

1. **Arestas do grafo:** a `_MaskedGraph` retorna apenas `edge_index` e `edge_type_ids`
   para os tipos de aresta ativos.
2. **Eventos da stream:** quando `CORR` está desativado, eventos `CORRELATION_UPDATE`
   são removidos de `events_by_date`.
3. **Rótulos de correlação:** quando `CORR` está desativado, `corr_labels_by_date`
   é zerado — o modelo treina sem sinal de correlação.

### Interpretação do ranking

```
Ablation ranking (Sharpe ↓):
  1. all_edges      mean_sharpe=+0.82   ← combinação completa é melhor
  2. CORR+FACT      mean_sharpe=+0.71
  3. CORR_only      mean_sharpe=+0.65   ← CORR é o tipo mais informativo
  4. FACT_only      mean_sharpe=+0.41
  5. SECT_only      mean_sharpe=+0.18   ← SECT contribui pouco isolado
  6. CORR+SECT      mean_sharpe=+0.12
  7. SECT+FACT      mean_sharpe=-0.03
```

Um tipo de aresta é **necessário** para o desempenho se sua remoção causa
queda significativa no Sharpe médio vs `all_edges`.

---

## Protocolo Walk-Forward (Rev 1+)

```
Regra de ouro: step_days >= test_days (janelas não-sobrepostas)
```

| Parâmetro | Padrão | Justificativa |
|-----------|--------|---------------|
| `train_days` | 500 | ~2 anos de histórico |
| `val_days` | 125 | ~6 meses |
| `test_days` | 125 | ~6 meses |
| `step_days` | 125 | = test_days → sem sobreposição |
| `n_bootstrap` | 2000 | Block bootstrap por janela |
| `block_size` | 5 | Captura autocorrelação semanal |

### Por que step_days < test_days é problemático?

O TGN mantém memória GRU entre os dias de um mesmo treinamento. Quando
`step_days < test_days`, o período de teste de uma janela sobrepõe com
o período de treino da janela seguinte. Isso significa que:

1. O TGN da janela k *treina* sobre dias que foram *testados* na janela k-1,
   com a memória "poluída" por padrões do período futuro.
2. GAT Static e ROLAND são stateless — não são afetados, criando uma
   comparação injusta.

---

## Testes Estatísticos

| Teste | Nível | Hipótese |
|-------|-------|----------|
| Binomial exacto | Confirmatorio | `win_rate > 0.50` (BL-18) ou `> 0.70` (H4 original) |
| Wilcoxon signed-rank (window-level) | Confirmatorio | Sharpe_esq > Sharpe_dir |
| Holm-Bonferroni | Correção múltipla | Aplicado sobre família confirmatória |
| Diebold-Mariano (pooled) | Exploratório | MAE_esq < MAE_dir (apenas sem sobreposição) |
| Block Bootstrap (por janela) | Exploratório | IC 95% do Sharpe diff por janela |

---

## Exemplos de Uso

```powershell
# Smoke test: 2 variantes, 2 janelas, 5 épocas
.venv\Scripts\python.exe scripts/run_bootstrap_eval_temporal_kg_rev2.py `
  --variants tgn roland --epochs 5 --max_windows 2

# 50 ativos, só TGN
.venv\Scripts\python.exe scripts/run_bootstrap_eval_temporal_kg_rev2.py `
  --variants tgn --n_tickers 50 --max_windows 3

# Ablação básica do TGN (CORR / SECT / FACT isolados)
.venv\Scripts\python.exe scripts/run_bootstrap_eval_temporal_kg_rev2.py `
  --ablation basic --ablation_variant tgn --max_windows 2 --epochs 5

# Ablação completa do ra_htgn (todos os subconjuntos)
.venv\Scripts\python.exe scripts/run_bootstrap_eval_temporal_kg_rev2.py `
  --ablation full --ablation_variant ra_htgn --max_windows 2 --epochs 5

# Treino standalone de uma variante
.venv\Scripts\python.exe scripts/train_link_prediction.py `
  --variant tgn --start 2021-01-01 --end 2023-12-31 --epochs 5
```
