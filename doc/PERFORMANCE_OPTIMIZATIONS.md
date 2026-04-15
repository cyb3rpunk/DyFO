# Performance Optimizations — BL-17 / BL-18 Training Pipeline

> Sessão de diagnóstico e otimização realizada em 2026-04-15.
> Todas as mudanças são de implementação apenas — nenhuma alteração de arquitetura,
> hiperparâmetros ou protocolo de avaliação.

---

## Contexto

Durante a execução de `run_bootstrap_eval_temporal_kg.py` (BL-18) foram identificados
três problemas distintos, resolvidos nesta ordem:

1. Bug de autograd em `temporal_kg.py` que impedia o treino
2. Ausência de suporte a GPU em todo o pipeline
3. Gargalos de CPU↔GPU em `temporal_kg` e `ra_htgn`

---

## 1. Bug: in-place operation no autograd (`temporal_kg`)

**Arquivo:** `dyfo/core/temporal_kg.py` — `TemporalKGCore.process_day`

### Causa

O loop de atualização de estado GRU usava atribuição indexada em-lugar:

```python
for node_idx, node_messages in messages_by_node.items():
    self.node_state[node_idx] = self.gru(..., self.node_state[node_idx]...).squeeze(0)
```

Cada `self.node_state[node_idx] = ...` incrementa o contador de versão interno do
tensor. Na iteração seguinte, o autograd detectava que o tensor usado como entrada do
GRU havia sido modificado desde que foi capturado no grafo computacional, lançando:

```
RuntimeError: one of the variables needed for gradient computation has been modified
by an inplace operation [...] is at version 31; expected version 30 instead.
```

### Correção

Separar leitura de escrita: computar todos os novos estados com os estados originais,
depois aplicar todas as atualizações em um único `index_put` não-in-place.

```python
# Acumular todos os novos estados
update_indices, update_states = [], []
for node_idx, node_messages in messages_by_node.items():
    mean_message = torch.stack(node_messages).mean(0)
    new_state = self.gru(mean_message.unsqueeze(0), self.node_state[node_idx].unsqueeze(0)).squeeze(0)
    update_indices.append(node_idx)
    update_states.append(new_state)

# Aplicar em um único index_put (cria novo tensor, sem bump de versão)
if update_indices:
    idx_t = torch.tensor(update_indices, dtype=torch.long, device=device)
    self._buffers["node_state"] = self.node_state.index_put((idx_t,), torch.stack(update_states))
```

O mesmo padrão foi aplicado em `detach_state`, que também passou a usar
`self.encoder._buffers["node_state"]` para consistência.

---

## 2. Suporte a GPU (`train_link_prediction.py`)

**Arquivo:** `scripts/train_link_prediction.py`

### Situação

O PyTorch instalado era `2.11.0+cpu` — sem suporte a CUDA. A RTX 3060 presente na
máquina não era utilizada. Após reinstalar com `torch==2.11.0+cu126`, foi adicionado
suporte explícito a dispositivo no script de treino.

### Mudanças

| Local | Mudança |
|---|---|
| Após `build_encoder` | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` |
| Encoder e decoder | `.to(device)` imediatamente após criação |
| Tensores estáticos do grafo | `edge_index`, `edge_type_ids`, `edge_timestamps` movidos para `device` |
| Loop de dias | `node_feat.to(device)` por iteração |
| Labels | `src`, `dst`, `targets` movidos para `device` após `build_regression_labels` |
| Perda BCE | `pos_weight=torch.tensor(pos_weight, device=device)` |

---

## 3. Vectorização do `process_day` (`temporal_kg`)

**Arquivo:** `dyfo/core/temporal_kg.py`

### Diagnóstico (profiler, 10 dias, CUDA)

| Hotspot | Calls | Tempo | % |
|---|---|---|---|
| `.item()` syncs GPU→CPU | 11.032 | 0.041s | 18% |
| `events_to_facts` / `event_to_fact` | 2.623 | 0.024s | 11% |
| `index_add_` | 20 | 0.022s | 10% |

### Problema principal

`last_update_time` era um `register_buffer` CUDA. No loop de preprocessamento de
fatos, `self.last_update_time[head_idx].item()` era chamado **uma vez por fato**
(~316 fatos/dia), forçando ~316 syncs GPU→CPU por dia de treino.

### Correção A — `last_update_time` como numpy

```python
# Antes
self.register_buffer("last_update_time", torch.zeros(num_nodes))
# ...
dts_list.append(fact.timestamp - float(self.last_update_time[head_idx].item()))

# Depois
self.last_update_time: np.ndarray = np.zeros(num_nodes, dtype=np.float64)
# ...
dts_list.append(fact.timestamp - float(self.last_update_time[head_idx]))  # numpy: sem sync
```

`last_update_time` é usado apenas para calcular `dt` em Python — nunca entra em
operações de gradiente. Mantê-lo em CPU como numpy é semanticamente correto.

### Correção B — Vectorização completa do forward

O loop `for fact in facts: _fact_message(fact)` foi substituído por:

1. **Preprocessamento Python** — extrai índices, atributos e `dt` em listas; zero ops CUDA
2. **`_fact_message_batch`** — constrói tensores batch e faz um único forward pass:
   ```python
   # lookup combinado: node_state | pseudo_entity_embeddings
   combined = torch.cat([self.node_state, self.pseudo_entity_embeddings.weight], dim=0)
   head_states  = self.node_state[head_indices]        # (F, D)
   tail_states  = combined[tail_combined_ids]           # (F, D)
   rel_states   = self.relation_embeddings(rel_ids)     # (F, D)
   dt_enc       = self.time_encoder(dts)                # (F, T)
   attr_repr    = self.attr_proj(cat([attrs, dt_enc]))  # (F, D)
   messages     = self.message_mlp(cat([head, rel, tail+attr]))  # (F, D)
   ```
3. **Scatter-mean** via `index_add_` + contagem de receptores
4. **GRU batch único** — `GRUCell` processa todos os K nós atualizados de uma vez

### Correção C — Explanations só fora do treino

```python
# _update_explanations é O(F) com score_fact por fato
if not self.training:
    self._update_explanations(facts, device=device)
```

### Correção D — Cap em `fact_history`

```python
if not self.training:
    self.fact_history.extend(facts)
    if len(self.fact_history) > 2000:
        self.fact_history = self.fact_history[-2000:]
```

### Resultado

| Métrica | Antes | Depois | Ganho |
|---|---|---|---|
| Tempo / 10 dias | 0.222s | 0.159s | −28% |
| ms/batch (loop completo) | — | 7.8ms | — |

---

## 4. Vectorização do `process_events` (`ra_htgn`)

**Arquivo:** `dyfo/core/relation_aware_tgn.py`

### Diagnóstico (profiler, 20 dias, CUDA)

| Hotspot | Calls | Tempo | % |
|---|---|---|---|
| `.item()` syncs GPU→CPU | **29.872** | **0.314s** | **32%** |
| `.to(device)` individuais | **5.210** | **0.093s** | **10%** |
| `linear` (NN real) | 181 | 0.070s | 7% |

### Problema A — Loop de atualização de timestamps

`process_events` atualizava `last_update_time` com um loop Python:

```python
for idx in range(source_nodes.shape[0]):
    src = source_nodes[idx].item()          # sync 1
    self.last_update_time[src] = max(
        self.last_update_time[src].item(),  # sync 2
        timestamps[idx].item(),             # sync 3
    )
    tgt = target_nodes[idx].item()          # sync 4
    if tgt >= 0:
        self.last_update_time[tgt] = max(
            self.last_update_time[tgt].item(),  # sync 5
            timestamps[idx].item(),             # sync 6
        )
```

~6 syncs GPU→CPU por evento × ~262 eventos/dia = **~1.572 syncs/dia**.

**Correção:** `scatter_reduce_` vectorizado — um kernel GPU por chamada:

```python
self.last_update_time.scatter_reduce_(
    0, source_nodes, timestamps, reduce="amax", include_self=True
)
real_targets = target_nodes[target_nodes >= 0]
if real_targets.numel() > 0:
    self.last_update_time.scatter_reduce_(
        0, real_targets, timestamps[target_nodes >= 0], reduce="amax", include_self=True
    )
```

### Problema B — Transfers individuais de `event.features`

```python
# Antes: N transfers individuais CPU→GPU
event_features = torch.stack([event.features.to(device) for event in events])

# Depois: stack em CPU, um transfer batch
event_features = torch.stack([event.features for event in events]).to(device)
```

### Resultado

| Métrica | Antes | Depois | Ganho |
|---|---|---|---|
| `.item()` calls / 20 dias | 29.872 | ~0 | −100% |
| `.to()` calls / 20 dias | 5.210 | 20 | −99.6% |
| ms/batch (loop completo) | 57.6ms | **16.2ms** | **−72%** |

---

## Resumo final — ms/batch por variante (loop completo: advance + embed + decode + backward)

| Variante | ms/batch | Gargalo restante |
|---|---|---|
| `tgn` | 46.2ms | `advance_day` (85%) — TGN original, não otimizado nesta sessão |
| `ra_htgn` | **16.2ms** | `build_intra_relation_messages` — computação neural legítima |
| `temporal_kg` | **7.8ms** | `decode_loss_back` (36%) — backward do decoder |
| `roland` | n/m | não perfilado nesta sessão |
| `gat_static` | n/m | não perfilado nesta sessão |

**Nenhuma métrica preditiva foi alterada.** As mudanças são puramente de eficiência
de execução: remoção de syncs GPU↔CPU desnecessários e vectorização de loops Python
que operavam fato-a-fato ou evento-a-evento.

---

## Arquivos modificados

| Arquivo | Mudanças |
|---|---|
| `dyfo/core/temporal_kg.py` | Fix autograd in-place; `last_update_time` → numpy; `process_day` vectorizado; explanations lazy; cap `fact_history` |
| `dyfo/core/relation_aware_tgn.py` | `scatter_reduce_` para timestamps; batch `.to(device)` |
| `scripts/train_link_prediction.py` | Suporte a GPU: `device` selection, `.to(device)` em encoder/decoder/tensores |
