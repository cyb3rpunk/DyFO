# DyFO — Arquitetura (ARCHITECTURE)

> Convenções de citação e proveniência: ver cabeçalho de `docs/OVERVIEW.md`.

---

## 1. Contratos de I/O do módulo

### Entrada (`.specs/codebase/ARCHITECTURE.md:11-18`, validado contra código)

```
G = (V, E)               # grafo financeiro heterogêneo (estático na init + CORR dinâmico)
stream de eventos e_i(t) # 7 tipos, dyfo/core/event_stream.py:25-32
π_t ∈ R^K                # probabilidades de regime do M1 (K=3) — SLOT RESERVADO
```

- **π_t (M1 → M2, BL-09 pendente):** os K=3 slots de `regime_prob` existem no vetor de
  features de nó (`dyfo/core/node_features.py:173-179`), mas o pipeline real nunca fornece
  `regime_probs` (`scripts/train_link_prediction.py:146-148` chama `build_daily_features`
  sem o argumento) → **zero-filled**. Integração real pendente (`STATE.md:103`;
  `ROADMAP.md:48`).

### Saída (M2 → M3)

```
e_t ∈ R^100   # embedding do grafo por dia útil
```

- `embedding_dim=100` (`dyfo/config.py:30`); `e_t` = readout global sobre `z_i(t)`
  (`dyfo/core/readout.py`; `dyfo/core/dyfo_module.py:151-188` no caminho TGN).
  O contrato com o M3/State Constructor é `[e_t | π_t | H(π_t) | α_t | x_t]`
  (`.specs/project/PROJECT.md:37-44`); do lado consumidor (ORION) existe apenas o slot
  desabilitado `graph_embedding=None` — ver `docs/OVERVIEW.md §2`.

### Invariantes (`.specs/codebase/ARCHITECTURE.md:24-31`)

- `e_t` uma vez por dia útil; TGAT **stateless** por padrão (sem memória recorrente);
  universo padrão **50 ações**; runners novos por variante (não editar
  `run_bootstrap_eval_v5.py` — `ROADMAP.md:69`).

## 2. Pipeline de 6 estágios

Conforme `.specs/codebase/ARCHITECTURE.md:35-80`, com mapeamento a código verificado:

| # | Estágio | O que faz | Código |
|---|---|---|---|
| 1 | Ingestion | ordena eventos por timestamp, agrupa por dia | `dyfo/core/event_stream.py:336-343` (`merge_and_sort`); agrupamento diário em `train_link_prediction.py:204-206` |
| 2 | Message Function | `m_i(t) = [s_i ‖ s_j ‖ φ(Δt) ‖ f_e ‖ edge_type_emb]` | TGN: `dyfo/core/tgn_encoder.py:48-79` (`MessageFunction`); TGAT: contexto `[evt_proj ‖ time_emb ‖ et_emb]` em `tgat_encoder.py:353-367` |
| 3 | Message Aggregation | média por nó (regra FED, ver §4) | TGN: `TGNEncoder(aggregation="mean")` (`dyfo_module.py:58-71`) |
| 4 | Temporal Attention / Memória | TGAT: multi-head attention sobre k eventos recentes (stateless); TGN: GRU memory | `tgat_encoder.py:119-179` (`_TemporalAttention`); `tgn_encoder.py` (GRU) |
| 5 | Graph Embedding (GAT) | `z_i(t) = GAT(h_i, N(i), edge_attr)` — 1 camada, 2 heads | `tgat_encoder.py:249-256` (GATConv `edge_dim=et_dim`), `:373-380` |
| 6 | Global Readout | `e_t = readout({z_i})`, dim 100 | `dyfo/core/readout.py:18-77` |

**Nuance de escopo (precisão sobre o spec):** `.specs/codebase/ARCHITECTURE.md:94` descreve
`dyfo_module.py` como orquestrador de "todos" os estágios, mas `DyFOModule` encapsula
especificamente o `TGNEncoder` (`dyfo/core/dyfo_module.py:24,58`). As demais variantes
(tgat, ra_htgn, gat_static, roland, temporal_kg) são encoders standalone instanciados pela
factory `build_encoder()` (`dyfo/core/model_variants.py:200-257`) e não passam por
`DyFOModule`; o decoder e o loop de treino são agnósticos à variante
(`model_variants.py:41-47`).

## 3. Configuração canônica (`dyfo/config.py`)

| Parâmetro | Valor | Linha |
|---|---|---|
| `embedding_dim` | **100** | `config.py:30` |
| `memory_dim` | **172** | `config.py:29` |
| `time_encoding_dim` | 100 | `config.py:31` |
| `edge_type_embedding_dim` | 16 | `config.py:32` |
| `num_attention_heads` | 2 | `config.py:33` |
| `num_neighbors` (k eventos/nó no buffer TGAT) | 10 | `config.py:34` |
| `num_gat_layers` | 1 | `config.py:35` |
| `node_feature_dim` | **20** = 1+1+1+11+1+1+K(3)+1 | `config.py:68` |
| `num_regimes` (K) | **3** | `config.py:69` |
| `model_variant` default | `"tgn"` (nota: o *primário documentado* é tgat/BL-21) | `config.py:24` |
| `correlation_method` | `"dcc_garch"` | `config.py:63` |
| `corr_sparsify_threshold` | 0.3 | `config.py:60` |
| `dcc_garch_window` / `rolling_corr_window` | 252 / 63 | `config.py:64-65` |
| `staleness_threshold_days` | 5 | `config.py:59` |

`VALID_MODEL_VARIANTS` (10): gat_static, ra_htgn, roland, temporal_kg, tgn, tgat,
persistence, ewma, zero, delta_ewma (`config.py:11-22`), com validação em `__post_init__`
(`config.py:71-77`). Catálogo completo em `docs/MODELS.md`.

Notas de consistência:
- Docstring de `node_features.py:3` diz "18-dim", mas a fórmula (`node_features.py:44-45`)
  e o código produzem **20** com K=3 — o valor funcional é 20, coerente com `config.py:68`.
- Docstring do TGAT (`tgat_encoder.py:29`) diz "k = 20 events per node", mas o valor real
  vem de `config.num_neighbors = 10` (`tgat_encoder.py:215`) — comentário desatualizado.

## 4. Grafo heterogêneo — 4 tipos de aresta

`edge_types = [CORR, SECT, SUPL, FACT]` (`config.py:39-41`); montagem em
`dyfo/core/graph_builder.py` (`FinancialGraph:26-103`; `build_initial_graph:133-179` monta
SECT/SUPL/FACT estáticos; CORR entra dinamicamente via eventos `CORRELATION_UPDATE`).

| Tipo | Semântica | Features | Código |
|---|---|---|---|
| CORR | correlação dinâmica (DCC-GARCH ou Pearson rolante), esparsificada \|ρ\|≥0.3 | `[ρ, Δρ, significance]` | `edge_features.py:27-76` (Pearson), `:182-325` (DCC-GARCH) |
| SECT | mesmo setor GICS (binário) | `[1.0]` | `edge_features.py:333-354` |
| SUPL | cadeia de suprimentos (CSV externo) | `[strength]` | `edge_features.py:362-383` |
| FACT | proximidade de loadings Fama-French 5 (OLS 252d, dist. L2 < 0.5) | `|Δβ₁..Δβ₅|` | `edge_features.py:391-454` |

**⚠️ SUPL está pendente (BL-10, `ROADMAP.md:49`):** o código de carga existe e funciona,
mas não há CSV de supply chain em `data/`, e o pipeline principal passa explicitamente
`supply_chain_edges=[]` (`scripts/train_link_prediction.py:191`). Na prática o grafo opera
com 3 tipos efetivos (CORR/SECT/FACT). É lacuna de **dados**, não de implementação.

**Relation-awareness (BL-27):** desde o fix edge_dim, o GATConv do TGAT recebe
`edge_attr = edge_type_emb` e diferencia vizinhos CORR/SECT/FACT
(`tgat_encoder.py:249-256,377-379`). Antes do fix, os tipos eram tratados como homogêneos,
causando diluição de atenção (diagnóstico em `.specs/quick/027-tgat-edge-dim-fix/TASK.md`).

## 5. Eventos — 7 tipos

`EventType` (`dyfo/core/event_stream.py:25-32`; espelhado em `config.py:44-54`):
PRICE_UPDATE, EARNINGS_REPORT, FED_DECISION, CREDIT_DOWNGRADE, CORP_ACTION,
CORRELATION_UPDATE, MACRO_RELEASE.

- Todos os tipos têm feature bruta de dimensão 3 (`event_stream.py:40-48`), e.g.
  PRICE_UPDATE `[Δret, vol_1d, volume_norm]`, CORRELATION_UPDATE `[ρ, Δρ, significance]`.
  No TGAT o vetor é padded/truncado para 20 antes da projeção
  (`tgat_encoder.py:306-315,230-233`) — na prática eventos reais preenchem 3 dims e o
  resto é zero-padding.
- Construtores: preços (`event_stream.py:100-158`), earnings (`:161-199`), corporate
  actions (`:202-235`), macro/FED (`:238-285` — broadcast para todos os N nós),
  correlação (`:288-333`).
- **Regra crítica FED_DECISION** (`.specs/codebase/ARCHITECTURE.md:111-120`): eventos FED
  atingem todos os nós no mesmo timestamp; o agregador deve ser `mean` (nunca `last`) para
  evitar atualização dependente de ordem — implementado via `aggregation="mean"` no
  `TGNEncoder` (`dyfo_module.py:70`).
- CREDIT_DOWNGRADE existe no enum e no config, mas **não há builder de eventos** para ele
  em `event_stream.py` (nenhuma fonte de dados de rating) — tipo definido porém nunca
  emitido no pipeline atual.

## 6. Features de nó (20 dims)

`NodeFeatureBuilder` (`dyfo/core/node_features.py:26-193`): retorno log 21d, vol 21d,
beta 63d vs benchmark, setor one-hot (11), market cap log-normalizado, drawdown corrente,
`regime_prob` (K=3, zero-filled — §1), volume normalizado 21d.

## 7. Readouts (M2 → e_t)

`dyfo/core/readout.py`: `MeanReadout` (default, `:18-30`), `WeightedReadout` (softmax de
pesos, e.g. market cap, `:33-46`), `AttentionReadout` (query aprendível, `:49-65`);
factory `get_readout` (`:68-77`). O caminho TGN usa `readout_strategy="mean"`
(`model_variants.py:149-151`).

## 8. Interface comum das variantes

`BaseGraphEncoder` (`dyfo/core/model_variants.py:41-127`) define o contrato de 4 métodos:

```
reset_state()             # zera estado temporal (início de época/janela)
advance_day(...)          # processa 1 dia de eventos, sem gradiente
get_node_embeddings(...)  # (N, embedding_dim), diferenciável
detach_state()            # TBPTT; no-op para variantes stateless
```

O loop de treino (`train_link_prediction.py:550-642`) usa exclusivamente essa interface,
com TBPTT via `detach_state()` após `backward()` (`:631-632`). Baselines estatísticos
(persistence/ewma/zero/delta_ewma) não instanciam encoder
(`train_link_prediction.py:394,455-467`) — ver `docs/MODELS.md §3`.
