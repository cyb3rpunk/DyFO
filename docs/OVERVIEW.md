# DyFO — Visão Geral (OVERVIEW)

> Documentação nível dissertação, gerada em 2026-07 a partir de auditoria do código e dos
> artefatos do repositório. **Convenção:** afirmações de `.specs/` e `doc/` são tratadas como
> REIVINDICAÇÕES e são validadas contra código/artefatos; onde a validação não foi possível,
> a afirmação está marcada `[NÃO VERIFICADO — alegado em <fonte>]`.
> Nota de proveniência: `.specs/`, `doc/` e `results/` estão **fora do controle de versão**
> (`.gitignore`, ver `docs/TESTING.md §5`), portanto citações a esses caminhos referem-se ao
> estado do disco local na data da auditoria.

---

## 1. O que é o DyFO

**DyFO (Dynamic Financial Ontology)** é o **Módulo 2 (M2)** do sistema MATTS v4.0
(`.specs/project/PROJECT.md:9`). Sua responsabilidade contratual é única: consumir um fluxo
de eventos financeiros com timestamps e produzir, a cada passo de decisão diário `t`, um
embedding do estado do grafo financeiro:

```
e_t ∈ R^100
```

- A dimensão 100 é fixada em `dyfo/config.py:30` (`embedding_dim: int = 100`) e o vetor
  `e_t` é produzido pelos readouts de `dyfo/core/readout.py:18-77` (mean / weighted /
  attention) sobre os embeddings por nó `z_i(t)`.
- O DyFO **não aprende política** — é módulo (não-agente), sem loop de recompensa
  (`.specs/project/PROJECT.md:24-25`).

Posição no MATTS (`.specs/project/PROJECT.md:30-44`):

| Módulo | Nome | Responsabilidade |
|---|---|---|
| M1 | RDM | Detector de regime — produz `π_t ∈ R^K` (K=3) |
| **M2** | **DyFO** | **Grafo financeiro dinâmico — produz `e_t ∈ R^100`** |
| M3 | State Constructor | Concatena `[e_t | π_t | H(π_t) | α_t | x_t]` |
| M4 | Orquestrador | Política MARL |
| M5 | Risk Manager | CVaR, restrições |

O acoplamento M1→M2 (entrada `π_t` como node feature) está **reservado mas não integrado**:
`dyfo/core/node_features.py:173-179` preenche os K=3 slots de `regime_prob` somente se um
`regime_probs` DataFrame for fornecido; no pipeline real (`scripts/train_link_prediction.py:146-148`)
esse argumento nunca é passado, logo os slots ficam **zero-filled** — pendência **BL-09**
(`.specs/project/STATE.md:103`; `.specs/project/ROADMAP.md:48`).

## 2. Papel na tese tri-repo (DyFO → ORION)

O acoplamento DyFO→ORION é **PLANEJADO, não implementado**. Verificado no próprio repo do
ORION:

- `d:\projetos\ORION\.specs\project\ROADMAP.md:28` — "Milestone 6: DyFO Integration (NEW,
  v2.0) — ⏳ ACTIVE", gated em DI-0 (reconciliação de dimensões), com F-6.1 = "RDM `π_t` →
  DyFO node-feature regime slots (DI-1)" (`ROADMAP.md:31` do ORION).
- `d:\projetos\ORION\docs\OVERVIEW.md:56` — "DI-1 ... Not started; no bridge/adapter code
  exists".
- A única coisa que existe hoje no ORION é um **slot desabilitado**: o parâmetro opcional
  `graph_embedding=None` no State Constructor
  (`d:\projetos\ORION\src\orion\modules\state_constructor\constructor.py:62`, guardado por um
  shape-check `(128,)` desatualizado em `constructor.py:74`, desligado por default e nunca
  passado por nenhum caller — conforme `d:\projetos\ORION\docs\MODELS.md:144`).
- O charter do ORION já registra o null do DyFO como "Lesson L-1" e restringe o papel
  futuro do DyFO a "structural representation + regime-conditioned risk, judged on stress
  drawdown/CVaR" (`d:\projetos\ORION\docs\MODELS.md:144`).

A migração de dados do DyFO para o dataset curado do PORTA também é planejada (ORION
D-12/DI-5) e **não foi feita** — ver `docs/DATA.md §6`.

## 3. Hipótese H4 — o que foi confirmado, exatamente

A redação atual de H4 em `.specs/project/PROJECT.md:49-51` é: *"O Sharpe e o MDD do TGAT
(stateless) são superiores aos baselines (TGN, ROLAND) em ≥70% das janelas walk-forward."*
O histórico dos testes, porém, mostra que o "H4 CONFIRMADA (p=0.0018)" de
`.specs/project/STATE.md:3` se refere a um teste **anterior, da era TGN**, e não à redação
TGAT acima. Decomposição com fontes:

1. **p = 0.0018** — Block bootstrap (10.000 iterações, blocos de 5 dias) sobre Sharpe GMVP,
   comparando **TGN vs ROLAND** (30 ativos, TGN v0.9): `P(TGN ≤ ROLAND) = 0.0018 → H4
   SUPPORTED` (`doc/EXPERIMENT_LOG.md:297-299` e `doc/EXPERIMENT_LOG.md:512`). A comparação
   TGN vs GAT-Static no mesmo teste **não** foi significativa (`P = 0.337`,
   `doc/EXPERIMENT_LOG.md:299`). Nenhum JSON em `results/` contém esse teste (o diretório
   `results/` só preserva execuções a partir de 2026-04-16); a fonte primária é o log de
   experimentos, que está fora do git.
2. **p < 0.0001** — citado em `.specs/project/ROADMAP.md:81` como "H4 p-value < 0.0001
   (TGAT > Baselines)". A evidência real com p<0.0001 é da validação robusta v1.0
   (Wilcoxon + Diebold-Mariano com Holm-Bonferroni), e compara **TGN** vs ROLAND/GAT-Static
   em **erro preditivo**, não TGAT vs baselines e não Sharpe
   (`doc/EXPERIMENT_LOG.md:543-546`). O rótulo "(TGAT > Baselines)" no ROADMAP é,
   portanto, **incorreto quanto ao modelo e à métrica** — discrepância registrada.
3. **A mesma validação robusta v1.0 NÃO suportou a H4 financeira**: `P(TGN ≤ ROLAND)
   [Sharpe] = 0.595 → ✗ NÃO SUPORTADA (n.s.)` (`doc/EXPERIMENT_LOG.md:557-559`), com a
   conclusão explícita de que "mais acurácia preditiva não garante maior Sharpe"
   (`doc/EXPERIMENT_LOG.md:562-563`).
4. **Na era TGAT (artefatos atuais), a redação "≥70% das janelas" não se sustenta**:
   - `results/bootstrap_eval_tkg_rev2_20260418_130703/bootstrap_summary_tkg_rev2.json`
     (50 tickers, 9 janelas): tgat vs tgn — 5/9 vitórias (win rate 0.556), Wilcoxon
     p=0.715, Holm-corrigido não significativo (`primary_comparison`).
   - `results/bootstrap_eval_tkg_rev3_20260420_141237/bootstrap_summary_tkg_rev3.json`:
     tgat vs tgn — 4/9 vitórias, Wilcoxon p=0.590.

**Síntese honesta:** existe confirmação estatística forte de superioridade **preditiva** do
encoder temporal sobre baselines de snapshot na era TGN (p<0.0001, DM/Wilcoxon/Holm), e uma
confirmação **financeira** pontual (p=0.0018, TGN vs ROLAND, v0.9) que **não se replicou**
na validação robusta v1.0 nem nos artefatos da era TGAT. Documentos internos (`STATE.md:3`,
`ROADMAP.md:47,81`) apresentam esses resultados de forma agregada e parcialmente trocada;
os dois p-values referem-se a testes distintos, ambos da era TGN.

## 4. O NULL HONESTO CENTRAL — forecast pontual de ρ a h=1

**Resultado:** o DyFO **não bate** persistência/EWMA no forecast pontual de correlação a
horizonte h=1, em probe diagnóstico sem leakage.

Artefato: `results/probe_results.txt` (gerado por `scripts/run_diagnostic_probe.py`;
ambos descritos em `docs/EXPERIMENTS.md §6`):

```
Window: CALM                          Window: BREAK
Persistence     R²= 0.9901            Persistence     R²= 0.9894
EWMA(λ=0.97)    R²= 0.9901            EWMA(λ=0.97)    R²= 0.9894
DyFO (TGAT)     R²=-5.3515            DyFO (TGAT)     R²=-4.3561
Headroom: -6.3416                     Headroom: -5.3456
```

Condições do probe: 15 tickers, DCC-GARCH targets, janelas CALM (teste ≈ 2019-H2) e BREAK
(teste ≈ 2022, ciclo de alta de juros), 500/125/125 dias, `USE_RHO_CONDITIONING = False`
com comentário "Ensure no leakage in DyFO" (`scripts/run_diagnostic_probe.py:19-20`).
O relatório do próprio probe nota que persistência/EWMA usam o alvo defasado `ρ_t`, que o
DyFO deliberadamente exclui do decoder (`results/probe_results.txt:23-27`) — são
referências diagnósticas, não baselines arquiteturais equivalentes.

**Reframe (posição atual da tese):** dado o null acima, o papel do DyFO deixa de ser
"prever ρ pontual melhor que métodos autoregressivos triviais" e passa a ser **canal
estrutural** — representação relacional do estado do mercado e covariância condicionada a
regime — cuja utilidade é julgada **downstream** (drawdown/CVaR sob stress no ORION/M4-M5).
Esse reframe está formalizado no charter do ORION como Lesson L-1
(`d:\projetos\ORION\docs\MODELS.md:144`) e é coerente com a evidência de stress events do
próprio DyFO (`results/stress_event_compare/SPY_BTC_USD_metrics.json` —
`"tgat_event_window_win": true` na janela do crash COVID; ver `docs/EXPERIMENTS.md §7` e
o caveat de rho_conditioning lá registrado).

## 5. O R²=0.824 NÃO contradiz o null

`.specs/project/PROJECT.md:100` e `.specs/project/ROADMAP.md:77` reportam "Test R² = 0.824"
como métrica v1.0 do TGAT. Esse número:

- **É da tarefa de LINK PREDICTION auto-supervisionada** (regressão de ρ_ij(t+1) contínuo a
  partir dos embeddings do par, protocolo walk-forward 60/20/20 —
  `scripts/train_link_prediction.py:55-67,359-369`), no universo de 50 tickers.
- Artefato exato: `results/link_pred_tgat_s42_20260421_032602/results.json` →
  `metrics.test_r_squared = 0.8240663912452635` (50 tickers, 2018-2024, 50 épocas, seed 42).
  O mesmo valor aparece como janela 2 do braço CORR+FACT da ablação em
  `results/bootstrap_eval_tkg_rev3_abl_full_tgat_20260420_214339/bootstrap_summary_tkg_rev3.json`
  (`ablation_results/CORR+FACT/window_metrics[1]/r_squared`).
- **Não é forecast pontual de ρ no protocolo do probe** — universo, janelas e protocolo são
  diferentes, e principalmente: um R² alto de regressão de ρ(t+1) em painel dominado por
  autocorrelação altíssima (persistência sozinha atinge R²≈0.99 no probe e R² médio 0.8783
  no próprio harness rev3 — `results/bootstrap_eval_tkg_rev3_20260501_200449/`) mede
  sobretudo a capacidade de reproduzir o nível corrente de ρ, não ganho preditivo
  incremental sobre o trivial. No mesmo harness rev3, **EWMA (R² médio 0.9905) supera o
  TGAT (0.8871)** em R² — coerente com o null do probe, não contraditório.

**Regra de leitura para a dissertação:** 0.824 valida que os embeddings do TGAT carregam
informação estrutural suficiente para reconstruir o campo de correlações (pré-treino
auto-supervisionado bem-sucedido); o probe mostra que isso **não** se traduz em vantagem
de forecast pontual a h=1 contra persistência. As duas afirmações são simultaneamente
verdadeiras porque medem tarefas diferentes.

## 6. Estado consolidado do backlog (validado contra código)

| BL | Reivindicação (.specs) | Verificação nesta auditoria |
|---|---|---|
| BL-21 TGAT primário | `ROADMAP.md:57,66` | ✅ `dyfo/core/tgat_encoder.py`; factory `model_variants.py:229-232` |
| BL-27 edge_dim fix | `ROADMAP.md:61`; `STATE.md:9-21` | ✅ `tgat_encoder.py:249-256` (`edge_dim=self._et_dim`) e `:377-379` (`edge_attr`); checkpoints antigos **incompatíveis** (`STATE.md:18-19`) |
| BL-28 multi-seed | `ROADMAP.md:62` | ✅ CLI `--seeds` (`run_bootstrap_eval_temporal_kg_rev3.py:951-954`); maior run multi-seed observado usou 3 seeds [42,123,456] (`results/bootstrap_eval_tkg_rev3_abl_full_tgat_20260421_114006/`), não os 5 anunciados |
| BL-29 hyperparams separados | `ROADMAP.md:63` | ✅ `TKG_USE_COSINE=True`, `TKG_PATIENCE=15` (`run_bootstrap_eval_temporal_kg_rev3.py:128-129`) |
| BL-17 RA-HTGN | `ROADMAP.md:53` | ✅ `dyfo/core/relation_aware_tgn.py` (922 linhas, `RAHTGNEncoder:804`) |
| BL-18 Temporal KG | **contradição**: `STATE.md:91-97` diz "aguardando BL-17" (checklist vazio) vs `ROADMAP.md:54` ✅ | ✅ **código confirma o ROADMAP**: `dyfo/core/temporal_kg.py` (395 linhas), `temporal_kg_adapter.py` (197), `scripts/run_bootstrap_eval_temporal_kg.py` (445), `tests/test_temporal_kg.py`; avaliado em `results/link_pred_temporal_kg_s42_*` (4 runs). `STATE.md` está desatualizado |
| BL-09 π_t real | `STATE.md:103` 🔴 | ✅ pendente confirmado (slots zero-filled, ver §1) |
| BL-10 SUPL | `ROADMAP.md:49` 🔴 | ✅ pendente confirmado (código existe, dados não — `docs/DATA.md §5`) |
| BL-12 staleness proxy | `ROADMAP.md:51` 🟡 doc-only | ✅ apenas contador no caminho TGN (`dyfo_module.py:76-145`), não é o proxy do spec |
| BL-30 re-ablação TGAT v2 | `ROADMAP.md:95` 🟡 em andamento | runs parciais em `results/bootstrap_eval_tkg_rev3_abl_full_tgat_*` (4 dirs) |

## 7. Mapa dos demais documentos

- `docs/ARCHITECTURE.md` — pipeline de 6 estágios, variantes, grafo heterogêneo, contratos I/O.
- `docs/DATA.md` — fontes (yfinance/FRED/FF5), universos, DCC-GARCH, limitações.
- `docs/MODELS.md` — catálogo das 10 variantes e status de resultados.
- `docs/EXPERIMENTS.md` — protocolo, catálogo de resultados, H4 em detalhe, probe, rho_conditioning.
- `docs/TESTING.md` — testes, runners, riscos de reprodutibilidade.
