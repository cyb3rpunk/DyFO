# 01 — Arquitetura do DyFO

> Contrato de I/O, pipeline de 6 estágios e invariantes arquiteturais.
> Este documento descreve a arquitetura estável do DyFO e as regras para evoluí-la sem
> quebrar o protocolo validado.

---

## Contrato de I/O do Módulo

### Entrada

```text
G = (V, E)                  # grafo financeiro heterogêneo (estático na init)
stream de eventos e_i(t)    # ver 03_event_spec.md
pi_t ∈ R^K                  # probabilidades de regime do M1 (K=3, zero-filled se M1 ausente)
```

### Saída

```text
e_t ∈ R^100                 # embedding do estado do grafo no passo t
```

### Invariantes

- `e_t` é produzido **uma vez por dia útil** (passo de decisão)
- O módulo é **stateless** por padrão (TGAT): não há memória recorrente persistente.
- A vizinhança temporal `φ(Δt)` captura a dinâmica histórica sem necessidade de GRU.
- O universo experimental padrão é de **50 ações** (otimizado).
- O protocolo de referência é o de [run_bootstrap_eval_v5.py](../scripts/run_bootstrap_eval_v5.py:1), mas novas variantes devem usar runners próprios.

---

## Pipeline de 6 Estágios

```text
Eventos do dia t (ordenados por timestamp)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 1: Ingestion                          │
│  Ordenar eventos por timestamp, enfileirar   │
└──────────────────────┬──────────────────────┘
                       │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 2: Message Function                   │
│  m_i(t) = [s_i(t⁻) ‖ s_j(t⁻) ‖ φ(Δt)        │
│            ‖ f_e(t) ‖ edge_type_emb]         │
└──────────────────────┬──────────────────────┘
                       │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 3: Message Aggregation                │
│  m̄_i(t) = mean({m_i(t) : eventos no batch}) │
│  (usar 'mean', nunca 'last' — ver §FED rule) │
└──────────────────────┬──────────────────────┘
                       │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 4: Temporal Attention (Stateless)     │
│  h_i(t) = MultiHeadAttention(φ(Δt), v_i(t))  │
│  Sem GRU/RNN (estabilidade walk-forward)     │
└──────────────────────┬──────────────────────┘
                       │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 5: Graph Embedding (GAT)              │
│  z_i(t) = GAT(h_i(t), N(i), edge_features)   │
│  1 camada, 2 heads, 10 vizinhos              │
└──────────────────────┬──────────────────────┘
                       │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 6: Global Readout                     │
│  e_t = mean({z_i(t) : i ∈ portfolio_K})      │
│  dim_out = 100                               │
└─────────────────────────────────────────────┘
```

---

## Mapa de Arquivos → Estágios

| Arquivo                     | Estágio(s) | Responsabilidade                    |
|-----------------------------|------------|-----------------------------------|
| `dyfo/core/event_stream.py` | 1          | Ingestion, ordenação de eventos   |
| `dyfo/core/tgat_encoder.py` | 2, 3, 4, 5 | Message function, Temp-Attn, GAT |
| `dyfo/core/node_features.py`| 5          | Features de nó `v_i(t)`           |
| `dyfo/core/edge_features.py`| 2          | Features de aresta por tipo       |
| `dyfo/core/readout.py`      | 6          | Global readout → `e_t`           |
| `dyfo/core/graph_builder.py`| init       | Constrói G=(V,E) estático        |
| `dyfo/core/dyfo_module.py`  | todos      | Orquestra o pipeline completo    |

---

## Dimensões críticas (não alterar sem justificativa)

| Dimensão                  | Valor | Motivo                                  |
|---------------------------|-------|-----------------------------------------|
| `embedding_dim`           | 100   | Padrão MATTS — `e_t` ocupa 100 dims     |
| `time_encoding_dim`       | 100   | Time2Vec, alinhado com `embedding_dim`  |
| `edge_type_embedding_dim` | 16    | Uma por tipo de aresta                  |
| `num_attention_heads`     | 2     | Baseline: 1 camada + 2 heads           |
| `num_neighbors`           | 10    | Vizinhança temporal                     |
| `num_gat_layers`          | 1     | Profundidade do grafo                   |

---

## Regra crítica: eventos FED_DECISION

Quando `FED_DECISION` é processado, **todos os N nós recebem evento no mesmo timestamp**.
Isso cria um batch com N mensagens simultâneas.

**Regra:** usar agregador `mean` (nunca `last`) para esses eventos.

Violação desta regra causa atualização não-determinística dependente de
ordem de processamento, introduzindo viés nos resultados.

---

## Regras de evolução arquitetural

1. **Não sobrescrever `tgat`:** a variante atual permanece disponível como referência primária.
2. **Novas variantes entram por `model_variant` novo.**
3. **Não editar o runner principal:** [run_bootstrap_eval_temporal_kg_rev2.py](../scripts/run_bootstrap_eval_temporal_kg_rev2.py:1) permanece como runner de referência.
4. **Sem expansão de universo:** manter 50 ativos como o padrão de escala ótimo.
