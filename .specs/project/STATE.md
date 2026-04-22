# STATE - Decisions & Blockers

Status atual: **H4 CONFIRMADA** (p=0.0018, Block Bootstrap). Baseline `tgn` v0.9 congelado.
**TGAT v2** (BL-27: edge_dim fix) implementado — GATConv agora relation-aware.
Proxima prioridade: **BL-30 Re-ablação TGAT v2** (validar `all_edges ≥ CORR+FACT`).

---

## BL-27 - TGAT Edge-Dim Fix (Relation-Aware GATConv)

**Decisão (2026-04-21):** O diagnóstico do teste de ablação revelou que o GATConv do TGAT
tratava CORR, SECT e FACT como vizinhos homogêneos (sem `edge_attr`). Isso causava diluição
de atenção quando SECT era adicionado sobre CORR+FACT => R² caía 0.0042.

**Correção:** Adicionado `edge_dim=self._et_dim` no `GATConv.__init__` e passagem de
`edge_type_emb` como `edge_attr` no forward. 3 linhas de mudança em `tgat_encoder.py`.

**Impacto:** Checkpoints anteriores (sem edge_dim) **não são compatíveis** com o TGAT v2.
Todos os experimentos precisam ser re-rodados.

**Spec:** `.specs/quick/027-tgat-edge-dim-fix/TASK.md`

## BL-28 - Multi-Seed Ablation

**Decisão (2026-04-21):** Seed=42 fixa no treinamento impedia validação estatística dos
deltas de R² entre variantes de ablação. Agora suportado via `--seeds 42 123 456 789 2024`.

## BL-29 - Hyperparams Separados para temporal_kg/ra_htgn

**Decisão (2026-04-21):** `temporal_kg` e `ra_htgn` convergem em 20-30 épocas (vs 5-8 do TGAT).
Patience=5 cortava o treinamento antes do ponto ótimo. Agora: `TKG_PATIENCE=15`, `TKG_USE_COSINE=True`.

---


## BL-17 - Relation-Aware Heterogeneous TGN

Implementar o encoder `ra_htgn` conforme `spec/04_tgn_spec.md`.
Regra: **nao editar** `run_bootstrap_eval_v5.py` nem o caminho `model_variant="tgn"`.

### Sessao 1 - RelationSemanticAttention
- [x] Criar `dyfo/core/relation_semantic_attention.py`
  - Inputs: lista de `(N, d_rel)` tensors, um por grupo ativo
  - `alpha_i^r = softmax(W · h_i^r)` - shape `(N, 4)`
  - `m_i^fusion = sum_r alpha_i^r · h_i^r` - shape `(N, d_fused)`
  - Salvar `self.last_attn_weights = alpha.detach()` (criterio aceite BL-17 #4)
- Contexto minimo: bloco "Fusao inter-relacao" de `04_tgn_spec.md`

### Sessao 2 - MessageFunction + IntraRelation (parte 1 do encoder)
- [x] Criar `dyfo/core/relation_aware_tgn.py` - parte 1
  - 4 grupos semanticos com projecao + LayerNorm proprios:
    - `node_event`: PRICE_UPDATE, EARNINGS_REPORT, CREDIT_DOWNGRADE, CORP_ACTION
    - `system_event`: FED_DECISION, MACRO_RELEASE
    - `pair_relation`: CORRELATION_UPDATE
    - `static_relation`: embedding estrutural CORR/SECT/SUPL/FACT
  - `IntraRelationAggregator`: `mean` para todos os grupos (FED determinismo)
- Contexto minimo: `tgn_encoder.py` (interface) + grupos acima + `03_event_spec.md`

### Sessao 3 - GAT relation-aware + fusao (parte 2 do encoder)
- [x] Completar `dyfo/core/relation_aware_tgn.py` - parte 2
  - Integrar `RelationSemanticAttention` apos intra-aggregation
  - GRU compartilhado sobre `m_i^fusion`
  - GAT com edge features reais projetadas para `edge_feat_dim=16`:
    - CORR: `[rho, delta_rho, significance]` -> Linear(3->16)
    - FACT: `[d_beta_1..d_beta_5]` -> Linear(5->16)
    - SUPL: `[strength]` -> Linear(1->16)
    - SECT: `[1.0]` -> Linear(1->16)
  - Classe final: `RAHTGNEncoder(BaseGraphEncoder)`
- Contexto minimo: `TemporalGraphAttention` atual + tabela de edge features

### Sessao 4 - Registro da variante
- [x] Editar `dyfo/core/model_variants.py` - adicionar caso `ra_htgn` em `build_encoder`
- [x] Editar `dyfo/config.py` - validar `model_variant="ra_htgn"`
- Contexto minimo: trechos de `build_encoder` e `DyFOConfig`

### Sessao 5 - Runner de avaliacao
- [x] Criar `scripts/run_bootstrap_eval_ra_htgn.py`
  - Copia da logica do v5 + `ra_htgn` como variante adicional
  - Comparacoes: `ra_htgn` vs `tgn` / `roland` / `gat_static`
- Contexto minimo: `run_bootstrap_eval_v5.py` + nomes de variantes

### Checklist de aceite BL-17
- [ ] `ra_htgn` roda end-to-end nos 30 ativos
- [ ] `run_bootstrap_eval_v5.py` nao foi alterado
- [ ] `build_encoder("tgn", ...)` continua funcionando
- [ ] `encoder.last_attn_weights` disponivel apos `compute_embeddings`
- [ ] Runner novo reporta `ra_htgn` vs `tgn` / `roland` / `gat_static`

---

## BL-18 - Temporal KG Ablation

Aguardando BL-17 concluido. Ver `spec/04_tgn_spec.md` - secao BL-18.

- [ ] Criar `dyfo/core/temporal_kg.py`
- [ ] Criar `dyfo/core/temporal_kg_adapter.py`
- [ ] Criar `scripts/run_bootstrap_eval_temporal_kg.py`

---

## Pendentes de outras BLs

- [ ] **BL-09** Integracao RDM: substituir `zero-filled` pelo `regime_prob` real do M1
- [ ] **BL-10** SUPL edges: decidir integracao via FactSet ou descartar para artigo inicial
- [ ] **BL-12** Staleness proxy: documentado em `02_graph_spec.md`, nao implementado
- [ ] **BL-16** Visualizacoes para o paper: heatmap de atencao (base mock pronta, extracao real pendente)
- [ ] Otimizacao DCC-GARCH: cache de correlacoes em `run_multi_seed.py` (~10x speedup)
