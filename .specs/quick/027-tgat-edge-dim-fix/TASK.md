# QF-027: TGAT Edge-Dim Fix — Atenção Relacional no GATConv

**Status:** ✅ Implementado
**Data:** 2026-04-21
**Escopo:** Small (2 arquivos, 1 mudança cirúrgica)

---

## Problema

O teste de ablação TGAT (`paper_abllation_tgat_20260420_214339`) revelou que a variante
`all_edges` (CORR+SECT+FACT) tem **R² inferior** à variante `CORR+FACT`:

| Variante | R² (Test) | MSE (×10⁻³) |
|:---|:---:|:---:|
| CORR+FACT | **0.8867** | **2.113** |
| all_edges | 0.8825 | 2.194 |
| Delta | **-0.0042** | +0.081 |

## Causa Raiz

O `GATConv` do TGAT (Step 2: structural readout) era chamado **sem `edge_attr`**:

```python
# ANTES — homogêneo: não diferencia CORR de SECT de FACT
self.gat = GATConv(in_channels=..., out_channels=..., heads=..., concat=True)
gat_out = self.gat(gat_in, edge_index_dev)
```

A atenção do GATConv era calculada apenas por `node features`, tratando todos os vizinhos
como idênticos. Ao adicionar SECT (estático, binário) sobre CORR+FACT, o grau de cada nó
aumentava sem informação discriminativa, **diluindo a atenção** que deveria fluir para
vizinhos CORR e FACT.

## Correção

Habilitado `edge_dim` no GATConv + passagem de `edge_type_emb` como `edge_attr`:

```python
# DEPOIS — relation-aware: GATConv diferencia edge types via embedding
self.gat = GATConv(..., edge_dim=self._et_dim)  # et_dim=16
edge_type_emb_gat = self.edge_type_emb(edge_type_ids.to(device))  # (E, 16)
gat_out = self.gat(gat_in, edge_index_dev, edge_attr=edge_type_emb_gat)
```

A atenção agora é: `α_ij = softmax(a^T [W·h_i || W·h_j || W_e·e_ij])` — condicional ao
tipo de aresta. O modelo aprende automaticamente a dar peso baixo a SECT quando redundante.

## Arquivos Alterados

| Arquivo | Mudança |
|:---|:---|
| `dyfo/core/tgat_encoder.py` L249-255 | Adicionado `edge_dim=self._et_dim` no GATConv |
| `dyfo/core/tgat_encoder.py` L372-379 | Passagem de `edge_type_emb_gat` como `edge_attr` |

## Verificação

- [ ] Ablação `all_edges` com TGAT v2 (edge_dim) deve ter R² ≥ `CORR+FACT`
- [ ] Modelos treinados anteriormente (sem edge_dim) **não são compatíveis** com checkpoints novos
- [ ] Nenhum outro encoder (TGN, RA-HTGN, Temporal KG) é afetado

## Referências

- Diagnóstico completo: `artifacts/diagnostic_all_edges_r2.md`
- PyG GATConv: `edge_dim` suportado nativamente desde PyG 2.0
- Xu et al. 2020 (TGAT): edge features no attention mechanism
