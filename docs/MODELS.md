# DyFO — Catálogo de Modelos (MODELS)

> Convenções de citação e proveniência: ver cabeçalho de `docs/OVERVIEW.md`.

---

## 1. Visão geral

`VALID_MODEL_VARIANTS` define **10 variantes** (`dyfo/config.py:11-22`), divididas em dois
grupos com tratamento de código distinto:

- **6 encoders de grafo** — instanciados pela factory `build_encoder()`
  (`dyfo/core/model_variants.py:200-257`) e obrigados ao contrato `BaseGraphEncoder`
  (`model_variants.py:41-127`; ver `docs/ARCHITECTURE.md §8`): `tgat`, `tgn`, `ra_htgn`,
  `temporal_kg`, `roland`, `gat_static`.
- **4 baselines estatísticos** — não instanciam encoder nem decoder; suas predições são
  calculadas diretamente da série de correlações dentro do loop de avaliação
  (`scripts/train_link_prediction.py:394` — `is_baseline = model_variant in
  ["persistence","ewma","zero","delta_ewma"]`; lógica de predição `:644-681`).

Decoders compartilhados (auto-supervisão de link prediction):
`LinkPredictor` (classificação, `dyfo/core/link_prediction.py:52`) e
`CorrelationRegressor` (regressão de ρ contínuo, `link_prediction.py:217-287` —
inclui o flag `use_rho_conditioning`, default `False`; semântica completa e discrepância
de rotulagem em `docs/EXPERIMENTS.md §5`).

## 2. Encoders de grafo (6)

### 2.1 `tgat` — PRIMÁRIO (BL-21)

`dyfo/core/tgat_encoder.py` (388 linhas), classe `TGATEncoder(BaseGraphEncoder)`.
Encoder **stateless** no estilo TGAT (Xu et al. 2020): cada nó mantém um buffer dos k
eventos mais recentes (k = `config.num_neighbors` = 10, `config.py:34`; a docstring
`tgat_encoder.py:29` diz "k = 20" — comentário desatualizado); codificação temporal
Time2Vec sobre Δt; atenção temporal multi-head (2 heads) sobre o buffer; readout final
por GATConv de 1 camada sobre o grafo do dia. **Fix BL-27 (edge_dim):** o GATConv recebe
`edge_dim=self._et_dim` (`tgat_encoder.py:249-256`, comentário "edge-type-aware attention
(fixes homogeneous dilution)") e o embedding do tipo de aresta é passado como `edge_attr`
no forward (`:377-379`). **Consequência crítica:** checkpoints treinados antes do fix são
**INCOMPATÍVEIS** com o TGAT v2 — "todos os experimentos precisam ser re-rodados"
(`.specs/project/STATE.md:18-19`); a re-ablação **BL-30 está pendente/em andamento**
(`ROADMAP.md:95`; runs parciais em `results/bootstrap_eval_tkg_rev3_abl_full_tgat_*`).
Resultados: R² médio 0.8871 no harness rev3 50-tickers (abaixo do EWMA 0.9905 —
`docs/EXPERIMENTS.md §4.1`); headline 0.824 discutida em `docs/OVERVIEW.md §5`.

### 2.2 `tgn` — baseline temporal com memória (rebaixado)

`dyfo/core/tgn_encoder.py` (509 linhas). Implementação TGN (Rossi et al. 2020) por
composição de blocos: `Time2Vec` (`:27`), `MessageFunction` (`:48` — mensagem
`[s_i‖s_j‖φ(Δt)‖f_e‖edge_type_emb‖event_type_emb]`), `MessageAggregator` (`:104`, média),
`MemoryUpdater` (GRU, `:146`), `TemporalGraphAttention` (`:174`) e `TGNEncoder` (`:297`).
É a única variante encapsulada por `DyFOModule` (`dyfo/core/dyfo_module.py:24,58`), via
`TGNWrapper` na factory (`model_variants.py:134-193`) — inclui o contador de staleness
(BL-12 parcial, `dyfo_module.py:76-145`). Foi o modelo primário até a v0.9 (os p-values
históricos de H4 são desta era — `docs/EXPERIMENTS.md §2`) e foi **rebaixado a baseline**
pela instabilidade recorrente (`ROADMAP.md:66`: "TGN rebaixado para baseline").

### 2.3 `ra_htgn` — TGN heterogêneo relation-aware (BL-17)

`dyfo/core/relation_aware_tgn.py` (922 linhas), classe `RAHTGNEncoder(BaseGraphEncoder)`
(`:804`). Estende o TGN com **4 grupos semânticos** (`SEMANTIC_GROUP_ORDER:27-32`:
node_event, system_event, pair_relation, static_relation; mapeamento evento→grupo em
`EVENT_TYPE_TO_GROUP:34-42`): projeções de mensagem específicas por grupo, agregação
intra-relação por média determinística, fusão inter-relação por atenção semântica
(`dyfo/core/relation_semantic_attention.py:17` — `RelationSemanticAttention`, softmax de
escores escalares por grupo, com `last_attn_weights` para diagnóstico) e GRU compartilhado
sobre a mensagem fundida (docstring `relation_aware_tgn.py:1-9`). Status: implementado e
testado (`tests/test_relation_aware_tgn.py`, 5 testes), mas com **avaliação escassa**:
apenas 2 runs de link prediction em `results/` — R²=0.7575
(`results/link_pred_ra_htgn_s42_20260421_110155/results.json`) e R²=**−0.7564**
(`.../link_pred_ra_htgn_s42_20260421_112649/results.json`) — a segunda run indica
instabilidade de treino não investigada; a variante **não aparece** em
`metrics_by_variant` de nenhum sumário bootstrap rev2/rev3.

### 2.4 `temporal_kg` — braço interpretável (BL-18)

`dyfo/core/temporal_kg.py` (395 linhas): `TemporalKGCore` (`:22`, nn.Module — scorer
recorrente por fato, GRUCell + Time2Vec, com artefatos de interpretabilidade
`last_explanations`/`last_relation_scores`) e `TemporalKGEncoder(BaseGraphEncoder)`
(`:340`). Os eventos/arestas são convertidos deterministicamente em fatos de KG temporal
por `dyfo/core/temporal_kg_adapter.py` (197 linhas): **6 relações canônicas**
(`CANONICAL_RELATIONS:13-20`: correlated_with, in_sector, supply_link,
similar_factor_profile, affected_by_event, exposed_to_macro) com mapeamentos
`EVENT_RELATION_MAP:23-31` e `STATIC_EDGE_RELATION_MAP:34-39`. O pipeline exporta os
artefatos de interpretabilidade quando a variante é temporal_kg
(`scripts/train_link_prediction.py:959`).

**Contradição documental BL-18 (veredito):** `.specs/project/STATE.md:91-97` diz
"Aguardando BL-17" com checklist vazio ("- [ ] Criar dyfo/core/temporal_kg.py"), mas
`ROADMAP.md:54` marca ✅ implementado. **O código confirma o ROADMAP**: os 4 arquivos
existem (core, adapter, runner `scripts/run_bootstrap_eval_temporal_kg.py` de 445 linhas,
`tests/test_temporal_kg.py`), com commits `ef5a9a5` e `0affa5a` anteriores ao topo do log.
`STATE.md` está desatualizado neste ponto. Ressalva de avaliação: há 4 runs de link
prediction (`results/link_pred_temporal_kg_s42_*`, melhor R²=0.7124), porém a variante
**nunca foi avaliada no protocolo bootstrap** — ausente de `metrics_by_variant` em todos
os sumários rev2/rev3 auditados.

### 2.5 `roland` — baseline snapshot+EMA ("ROLAND-like")

`dyfo/core/roland_baseline.py` (268 linhas), classe `ROLANDLikeEncoder(BaseGraphEncoder)`
(`:67`). Declaradamente **"ROLAND-like", não uma reimplementação fiel** de You et al.
2022 — a docstring (`:3-19`) é explícita sobre as diferenças: snapshots mensais de
correlação (em vez de contínuos), atualização de embedding por EMA
(`config.roland_ema_alpha=0.05`, `config.py:36`) em vez de GRU hierárquico, e GAT 2-layer
em vez de GraphSAGE. Qualquer comparação na dissertação deve usar o rótulo "ROLAND-like"
para não superrepresentar o baseline.

### 2.6 `gat_static` — baseline estático (BL-02)

`dyfo/core/gat_static_baseline.py` (258 linhas), classe `GATStaticEncoder(BaseGraphEncoder)`
(`:132`). GAT de 2 camadas (implementação manual `_GATLayer`/`MultiHeadGATLayer`, sem
torch_geometric) sobre um **grafo estático** de correlação média: |ρ| médio calculado
**apenas nas datas de treino** e limiarizado — comentário explícito "preventing look-ahead
leakage" (`gat_static_baseline.py:188`). Sem qualquer componente temporal; é o controle
para a pergunta "a dinâmica temporal adiciona algo?".

## 3. Baselines estatísticos (4)

Nenhum destes instancia encoder; predições vêm de `scripts/train_link_prediction.py:644-681`.

- **`persistence`** — prediz `ρ̂(t+1) = ρ(t)` (lookup direto na correlação do dia,
  `train_link_prediction.py:671-672`). É a régua mais dura do domínio: R²≈0.99 no probe
  (`results/probe_results.txt`) e R² médio 0.8783 no harness rev3.
- **`ewma`** — EMA da série de ρ por par com `ewma_alpha = 0.05`
  (`train_link_prediction.py:515,649-654`). Nota: no probe diagnóstico o EWMA é
  reimplementado à parte com busca de λ ∈ {0.80,0.90,0.94,0.97}
  (`scripts/run_diagnostic_probe.py:137-149`) — parametrizações distintas nos dois lugares.
- **`zero`** — prediz sempre 0.0; só é semanticamente válido com `use_delta_target=True`
  (alvo Δρ = ρ(t+1)−ρ(t), `dyfo/config.py:25`), e o código emite warning caso contrário
  (`train_link_prediction.py:645-647`). Equivale a "ρ não muda".
- **`delta_ewma`** — EMA dos **incrementos** Δρ por par (`train_link_prediction.py:655-664`),
  baseline natural do modo delta-target (coberto por `tests/test_delta_rho_target.py`).

Os quatro foram avaliados no harness rev3 (sumários de 2026-05-01/05-02,
`docs/EXPERIMENTS.md §4`).

## 4. rho_conditioning — veredito consolidado

- **Default `False` no pipeline canônico**, confirmado em três pontos:
  `CorrelationRegressor.__init__` (`dyfo/core/link_prediction.py:237`),
  `train_link_prediction()` (`scripts/train_link_prediction.py:281`) e a chamada do runner
  canônico rev3, que não passa o parâmetro
  (`scripts/run_bootstrap_eval_temporal_kg_rev3.py:253-271`). O probe do null roda
  explicitamente com `False` (`scripts/run_diagnostic_probe.py:19-20`).
- **`True` apenas em 2 scripts de nicho:**
  1. `scripts/run_smoketest_covid_tgat_plus_rho.py` — rotula honestamente a variante
     como `tgat_plus_rho` (docstring `:9-13`); padrão correto.
  2. `scripts/run_spy_vix_covid_compare.py:235` — ativa o conditioning **sob o rótulo
     `"tgat"` sem rebatizar**, e os JSONs de métricas de stress event não registram o
     flag. **Discrepância de rotulagem registrada** — detalhes e implicações em
     `docs/EXPERIMENTS.md §5`.

## 5. Resumo de status por variante

| Variante | Código | Testes unit. | Avaliação bootstrap (rev2/rev3) | Observação |
|---|---|---|---|---|
| tgat | ✅ `tgat_encoder.py` | via smoke/variants | ✅ (várias) | primário; BL-30 re-ablação pendente |
| tgn | ✅ `tgn_encoder.py` | via smoke | ✅ (várias) | baseline; era do p=0.0018 |
| ra_htgn | ✅ `relation_aware_tgn.py:804` | ✅ 5 testes | ❌ ausente | 1 run instável (R²=−0.76) |
| temporal_kg | ✅ `temporal_kg.py:340` | ✅ 2 testes | ❌ ausente | STATE.md desatualizado (diz pendente) |
| roland | ✅ `roland_baseline.py:67` | via variants | ✅ | "ROLAND-like", não fiel |
| gat_static | ✅ `gat_static_baseline.py:132` | via smoke | ✅ | controle estático |
| persistence | ✅ `train_link_prediction.py:671` | — | ✅ | régua principal do null |
| ewma | ✅ `:649-654` | — | ✅ | supera TGAT em R² no rev3 |
| zero | ✅ `:645,669` | ✅ delta tests | ✅ (delta) | requer delta-target |
| delta_ewma | ✅ `:655-664` | ✅ delta tests | ✅ (delta) | baseline do modo Δρ |
