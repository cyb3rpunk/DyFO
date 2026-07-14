# DyFO — Dados (DATA)

> Convenções de citação e proveniência: ver cabeçalho de `docs/OVERVIEW.md`.

---

## 1. Fontes de dados (3 adapters, todos gratuitos)

`dyfo/data/` contém 3 adapters, todos com retry + backoff exponencial:

### 1.1 yfinance (`dyfo/data/yfinance_adapter.py`)

Preços, fundamentals e eventos corporativos via Yahoo Finance (retry 5×, backoff base
3.0s — `:18-19`):

| Função | Linha | Produz |
|---|---|---|
| `download_prices` | `:86` | preços de fechamento ajustados (painel) |
| `download_ohlcv` | `:118` | OHLCV por ticker |
| `get_ticker_info` | `:140` | metadados (setor, market cap) |
| `get_earnings_dates` | `:162` | datas de earnings → eventos EARNINGS_REPORT |
| `get_corporate_actions` | `:203` | dividendos/splits → eventos CORP_ACTION |

### 1.2 FRED (`dyfo/data/fred_adapter.py`)

Séries macroeconômicas do Federal Reserve (retry 3× — `:18-19`); resolução da API key em
`_get_api_key` (`:22`): argumento → env var `FRED_API_KEY` → `.env` (o repo contém
`.env` com a chave e `.env.example`). `download_fred_series` (`:42`) baixa as séries;
`detect_macro_events` (`:94`) converte mudanças relevantes em eventos
FED_DECISION/MACRO_RELEASE.

As **8 séries** configuradas (`dyfo/config.py:92-104`): DFF (fed funds), VIXCLS (VIX),
BAMLC0A0CM (credit spread OAS), DGS10, DGS2 (yields 10y/2y), CPIAUCSL (CPI),
UNRATE (desemprego), MANEMP (proxy de PMI manufatureiro).

> **⚠️ Nota de segurança (registrada, não corrigida — fora do escopo desta documentação):**
> `DataConfig.fred_api_key` (`dyfo/config.py:91`) traz uma chave de API **hardcoded como
> default** no código versionado (string hex de 32 chars, não reproduzida aqui). O caminho
> correto (`.env`/env var) existe no adapter; o default hardcoded é um vazamento de
> credencial em potencial e deveria ser removido/rotacionado.

### 1.3 Ken French FF5 (`dyfo/data/ff_adapter.py`)

`download_ff5_factors` (`:35`) baixa o dataset diário Fama-French 5 fatores (2x3)
diretamente da Ken French Data Library (URL em `:25-28`), com cache local em `data/`
(`CACHE_DIR`, `:32`) — o cache `data/ff5_daily.csv` está presente no repo. Os fatores
alimentam as arestas FACT (`docs/ARCHITECTURE.md §4`).

## 2. Configuração de dados (`DataConfig`, `dyfo/config.py:80-120`)

- `benchmark_ticker = "SPY"` (`:86`) — usado para beta de nó (63d).
- Período: `start_date = "2018-01-01"`, `end_date = "2025-12-31"` (`:87-88`). Nota: os
  runs de link prediction usam recortes próprios (tipicamente 2018-01-01→2024-12-31; o
  probe usa 2016-01-01→2023-06-01).
- `gics_sectors`: **11 setores** GICS-like (`:106-120`) — one-hot de 11 dims no vetor de
  features de nó.

## 3. Universos de tickers (`dyfo/core/ticker_registry.py`, BL-20)

| Universo | Definição | Linhas | Sparsificação CORR |
|---|---|---|---|
| `TICKERS_30` | 30 large caps S&P 500, diversificação setorial | `:30-53` | threshold \|ρ\| > 0.3 |
| `TICKERS_50` | = 30 + 20 (universo **padrão/ótimo**, BL-22) | `:58-81` | threshold \|ρ\| > 0.3 |
| `TICKERS_100` | = 50 + 50 | `:86-111` | "tmfg" (ver ressalva) |

Asserts de tamanho em `:113-114`; estratégia por escala em `SPARSIFICATION_STRATEGY`
(`:117-121`); acesso via `get_tickers(n)` / `get_sparsification(n)` (`:124,155`).

**⚠️ Ressalva TMFG (BL-26 pendente, `ROADMAP.md:91`):** o registro rotula o universo 100
como "tmfg", mas o runner canônico **não implementa TMFG** — ao detectar
`sparsification == "tmfg"` ele emite warning explícito: *"The current implementation uses
threshold — results may differ from spec"*
(`scripts/run_bootstrap_eval_temporal_kg_rev3.py:747-753`). Todos os resultados de 100
tickers usam, na prática, threshold. Coerente com o colapso observado em escala 100
(tgat R²=−4.47, `docs/EXPERIMENTS.md §4.2`).

## 4. Correlações-alvo: DCC-GARCH (BL-03)

`dyfo/core/edge_features.py`:

- **Método canônico:** `compute_dcc_garch_correlations` (`:182-325`) — DCC-GARCH via o
  pacote `arch` (`requirements.txt`: `arch>=6.0`), janela `dcc_garch_window = 252`
  (`config.py:64`), com **fallback automático para Pearson rolante** se o ajuste GARCH
  falhar em >50% dos ativos. `correlation_method = "dcc_garch"` é o default
  (`config.py:63`) e é o método registrado nos artefatos
  (`results/link_pred_*/results.json` → `params.correlation_method`).
- **Alternativa:** `compute_rolling_correlations` (`:27-76`), Pearson com janela 63d
  (`config.py:65`).

As correlações produzem tanto as arestas CORR (esparsificadas em |ρ| ≥ 0.3,
`config.py:60`) quanto os **alvos de regressão** ρ(t+1) do pré-treino auto-supervisionado.
O cache de dados preparados é persistido em `results/prepared_data_cache_<hash>.pkl`
(9 caches presentes).

## 5. Grafo estático inicial e a lacuna SUPL (BL-10)

`GraphBuilder.build_initial_graph` (`dyfo/core/graph_builder.py:133-179`) monta as arestas
estáticas SECT/SUPL/FACT; CORR entra dinamicamente por eventos.

- SECT: binário mesmo-setor (`edge_features.py:333-354`), a partir de `get_ticker_info`.
- FACT: proximidade de loadings FF5 por OLS 252d (`edge_features.py:391-454`).
- **SUPL: código existe, dados não.** `load_supply_chain_edges` (`edge_features.py:362-383`)
  lê um CSV externo `source_ticker,target_ticker,strength`, mas `data/` contém apenas
  3 arquivos (`ff5_daily.csv`, `tickers_nasdaq100.txt`, `.gitkeep`) — **não há CSV de
  supply chain** — e o pipeline principal passa `supply_chain_edges=[]` explicitamente
  (`scripts/train_link_prediction.py:191`). Lacuna de **dados** (BL-10, `ROADMAP.md:49`),
  não de implementação.

## 6. Event stream (dados → eventos)

`dyfo/core/event_stream.py`: `EventType` (7 tipos, `:25-32`), `FinancialEvent`
(dataclass, `:56-74`), `EventStreamBuilder` (`:93+`) com construtores por fonte:
preços (`:100-158`), earnings (`:161-199`), corporate actions (`:202-235`), macro/FED
(`:238-285`, broadcast a todos os nós), correlação (`:288-333`). Cada tipo emite feature
bruta de dimensão 3 (`_EVENT_FEATURE_DIMS`, `:40-48`); detalhes de padding e da regra
FED em `docs/ARCHITECTURE.md §5`. CREDIT_DOWNGRADE está definido no enum mas **não tem
builder nem fonte de dados** — nunca é emitido.

**Integração π_t (M1→M2):** os 3 slots de `regime_prob` no vetor de nó existem
(`dyfo/core/node_features.py:173-179`) mas ficam **zero-filled** no pipeline real —
BL-09 pendente (`docs/OVERVIEW.md §1`). A migração futura dos dados para o dataset curado
do PORTA (ORION D-12/DI-5) é **planejada, não implementada**.

## 7. Riscos de proveniência de dados

- Dados de mercado **não são versionados** (apenas o cache FF5 e uma lista NASDAQ-100
  estão em `data/`); cada reexecução re-baixa yfinance/FRED, sujeitas a revisões
  retroativas do provedor (splits, ajustes de dividendos) — os alvos DCC-GARCH podem não
  ser bit-a-bit reprodutíveis entre datas de download.
- Os caches `results/prepared_data_cache_*.pkl` congelam os dados usados nos runs, mas
  estão em `results/`, que é **inteiramente ignorado pelo git** — ver `docs/TESTING.md §5`.
- `scripts/audit_data_sources.py` existe para auditoria de fontes
  `[NÃO VERIFICADO — conteúdo do script não lido nesta auditoria]`.
