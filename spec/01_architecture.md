# 01 — Arquitetura do DyFO

> Contrato de I/O, pipeline de 6 estágios e invariantes arquiteturais.
> Este documento descreve a arquitetura estável do DyFO e as regras para evoluí-la sem
> quebrar o protocolo validado.

---

## Contrato de I/O do Módulo

### Entrada
```
G = (V, E)                  # grafo financeiro heterogêneo (estático na init)
stream de eventos e_i(t)    # ver 03_event_spec.md
pi_t ∈ R^K                  # probabilidades de regime do M1 (K=3, zero-filled se M1 ausente)
```

### Saída
```
e_t ∈ R^100                 # embedding do estado do grafo no passo t
```

### Invariantes
- `e_t` é produzido **uma vez por dia útil** (passo de decisão)
- A memória dos nós **persiste** entre passos (não é resetada por dia)
- A memória é **zerada apenas** no início de cada episódio de treinamento
- O módulo é **stateless entre episódios** (sem aprendizado online)
- O coupling com outros módulos é **exclusivamente via `e_t`**
- O universo experimental desta fase é **fixado em 30 ações**
- O protocolo de referência é o de [run_bootstrap_eval_v5.py](../scripts/run_bootstrap_eval_v5.py:1),
  mas novas variantes devem usar runners próprios

---

## Pipeline de 6 Estágios

```
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
│  m_i(t) = [s_i(t⁻) ‖ s_j(t⁻) ‖ φ(Δt)      │
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
│  Stage 4: Memory Update                      │
│  s_i(t) = GRU(m̄_i(t), s_i(t⁻))             │
│  hidden_size = 172, weight sharing           │
└──────────────────────┬──────────────────────┘
                       │
        ▼
┌─────────────────────────────────────────────┐
│  Stage 5: Graph Embedding (GAT)              │
│  h_i(t) = s_i(t) + v_i(t)                   │
│  z_i(t) = TGN-attn(h_i(t), N(i), φ)         │
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

| Arquivo | Estágio(s) | Responsabilidade |
|---------|-----------|-----------------|
| [dyfo/core/event_stream.py](../dyfo/core/event_stream.py) | 1 | Ingestion, ordenação de eventos |
| [dyfo/core/tgn_encoder.py](../dyfo/core/tgn_encoder.py) | 2, 3, 4, 5 | Message function, GRU, GAT |
| [dyfo/core/node_features.py](../dyfo/core/node_features.py) | 5 | Features de nó `v_i(t)` |
| [dyfo/core/edge_features.py](../dyfo/core/edge_features.py) | 2 | Features de aresta por tipo |
| [dyfo/core/readout.py](../dyfo/core/readout.py) | 6 | Global readout → `e_t` |
| [dyfo/core/graph_builder.py](../dyfo/core/graph_builder.py) | init | Constrói G=(V,E) estático |
| [dyfo/core/link_prediction.py](../dyfo/core/link_prediction.py) | — | Cabeça MLP para pré-treino |
| [dyfo/core/dyfo_module.py](../dyfo/core/dyfo_module.py) | todos | Orquestra o pipeline completo |
| [dyfo/config.py](../dyfo/config.py) | — | `DyFOConfig` + `DataConfig` |
| [dyfo/data/yfinance_adapter.py](../dyfo/data/yfinance_adapter.py) | — | Dados de mercado (preços, eventos) |
| [dyfo/data/fred_adapter.py](../dyfo/data/fred_adapter.py) | — | Dados macro (FRED) |
| [dyfo/data/ff_adapter.py](../dyfo/data/ff_adapter.py) | — | Fatores Fama-French 5 |

### Extensões planejadas nesta fase

| Arquivo | Papel |
|---------|------|
| [04_tgn_spec.md](04_tgn_spec.md) | fonte de verdade para `ra_htgn` e `temporal_kg` |
| `dyfo/core/relation_aware_tgn.py` | encoder BL-17 |
| `dyfo/core/relation_semantic_attention.py` | fusão inter-relação BL-17 |
| `dyfo/core/temporal_kg.py` | braço BL-18 |
| `scripts/run_bootstrap_eval_ra_htgn.py` | avaliação isolada BL-17 |
| `scripts/run_bootstrap_eval_temporal_kg.py` | avaliação isolada BL-18 |

---

## Dimensões críticas (não alterar sem justificativa)

| Dimensão | Valor | Motivo |
|---------|-------|--------|
| `memory_dim` | 172 | Alinhado com features de aresta LIWC do MATTS |
| `embedding_dim` | 100 | Padrão MATTS — `e_t` ocupa primeiros 100 dims de `s_t` |
| `time_encoding_dim` | 100 | Time2Vec, alinhado com `embedding_dim` |
| `edge_type_embedding_dim` | 16 | Uma por tipo de aresta |
| `num_attention_heads` | 2 | Ablation do paper original: 1 camada + 2 heads |
| `num_neighbors` | 10 | Vizinhança temporal TGN |
| `num_gat_layers` | 1 | Paper original: 1 camada c/ memória > 2 sem memória |

---

## Regra crítica: eventos FED_DECISION

Quando `FED_DECISION` é processado, **todos os N nós recebem evento no mesmo timestamp**.
Isso cria um batch com N mensagens simultâneas.

**Regra:** usar agregador `mean` (nunca `last`) para esses eventos.

Violação desta regra causa atualização de memória não-determinística dependente de
ordem de processamento, introduzindo viés nos resultados.

---

## Raw Message Store — mecanismo de backprop

Para treinar módulos de memória via backpropagation:

```
Batch t:
  1. Buscar raw messages armazenadas do batch (t-1)
  2. Computar mensagens → agregar → atualizar memória com as raw de (t-1)
  3. Usar memória atualizada → embeddings → calcular loss
  4. Armazenar raw messages do batch t para uso em (t+1)
  5. Gradiente flui: loss → embeddings → memória → raw messages de (t-1)
```

**batch_size padrão:** 200 eventos ≈ 1 ano de pregões diários.
Reduzir para 64-128 em datasets de crise (eventos esparsos).

---

## Regras de evolução arquitetural

1. **Não sobrescrever `tgn`:** a variante atual permanece disponível como baseline histórico.
2. **Novas variantes entram por `model_variant` novo:** no mínimo `ra_htgn` e `temporal_kg`.
3. **Não editar o runner validado:** [run_bootstrap_eval_v5.py](../scripts/run_bootstrap_eval_v5.py:1) permanece congelado.
4. **Compatibilidade do protocolo:** novos runners devem manter a lógica walk-forward do v5.
5. **Sem expansão de universo:** qualquer mudança que aumente o número de ativos está fora do escopo desta fase.
