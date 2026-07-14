# DyFO — Protocolo Experimental e Resultados (EXPERIMENTS)

> Convenções de citação e proveniência: ver cabeçalho de `docs/OVERVIEW.md`.
> **Aviso estrutural:** `results/` (1554 arquivos) está integralmente no `.gitignore` e
> nunca foi rastreado pelo git (`git ls-files results` vazio) — todos os artefatos citados
> abaixo existem apenas no disco local. Ver `docs/TESTING.md §5`.

---

## 1. Protocolo de avaliação

### 1.1 Split walk-forward

- **Padrão 60/20/20** por dias de pregão: treino 60%, validação 20%, teste 20%, com estado
  temporal herdado entre splits (sem reset nas fronteiras) —
  `scripts/train_link_prediction.py:16-21` (docstring) e `:359-369` (implementação).
  Splits customizados podem ser injetados via `train_dates/val_dates/test_dates`
  (`train_link_prediction.py:277-279,360-361`), que é como os runners de janelas rolantes
  e o probe operam.
- **Janelas rolantes:** o runner rev3 avalia 9 janelas walk-forward no protocolo padrão
  (observado em todos os sumários com `n_windows: 9`, e.g.
  `results/bootstrap_eval_tkg_rev3_20260501_200449/bootstrap_summary_tkg_rev3.json`,
  `run_config.n_windows = 9`).

### 1.2 Bootstrap em blocos — reivindicação vs artefatos

| Fonte | n_bootstrap | block_size |
|---|---|---|
| `.specs/codebase/TESTING.md:45-46` (reivindicação de protocolo) | 10000 | 10 |
| `doc/EXPERIMENT_LOG.md:297` (teste H4 v0.9) | 10000 | 5 |
| `scripts/run_bootstrap_eval_temporal_kg_rev3.py:114-115` (defaults do código) | 2000 | 5 |
| `scripts/run_bootstrap_eval_temporal_kg_rev2.py:111-112` (defaults do código) | 2000 | 5 |
| Artefatos rev2 30-tickers (`bootstrap_eval_tkg_rev2_20260416_202606`, `_20260417_133120`) | 5000 | 5 |
| Artefatos rev2 50-tickers e todos os rev3 (`run_config` dos JSONs) | 2000 | 5 |

**Discrepância registrada:** o protocolo "10k bootstrap, block_size=10" descrito em
`.specs/codebase/TESTING.md:45-46` **não corresponde** a nenhum artefato encontrado em
`results/` nem aos defaults dos runners; as execuções reais usaram 2000-5000 iterações com
blocos de 5 dias. O único uso documentado de 10k iterações é o teste H4 v0.9
(`doc/EXPERIMENT_LOG.md:297`), com blocos de 5 (não 10).

### 1.3 Multi-seed (BL-28)

- Suporte via `--seeds` no runner rev3 (`run_bootstrap_eval_temporal_kg_rev3.py:951-954`),
  cujo help sugere `--seeds 42 123 456 789 2024` (5 seeds), conforme `STATE.md:23-26`.
- **Artefatos observados:** o maior run multi-seed usa 3 seeds `[42, 123, 456]`
  (`results/bootstrap_eval_tkg_rev3_abl_full_tgat_20260421_114006/bootstrap_summary_tkg_rev3.json`,
  `run_config.seeds`). Runs de link prediction por seed: s42 (200+ diretórios), s123 (~26),
  s456 (~16); **nenhum diretório s789 ou s2024 foi encontrado**. A validação 5-seed
  anunciada em `ROADMAP.md:62,93` está implementada no CLI, mas
  `[NÃO VERIFICADO — nenhum artefato de execução com 5 seeds foi localizado]`.

### 1.4 Testes estatísticos (BL-24)

Integrados ao runner rev3 (importados em `run_bootstrap_eval_temporal_kg_rev3.py:70-75`;
aplicados em `:471-491`):

- **Wilcoxon signed-rank** por janela sobre Sharpe (`run_window_wilcoxon`, one-sided
  "greater").
- **Diebold-Mariano** sobre erros de predição (`diebold_mariano_test`), reportado nos JSONs
  em `pooled_predictive_tests`.
- **Holm-Bonferroni** para múltiplas comparações (`holm_bonferroni`,
  `holm_bonferroni_confirmatory` nos JSONs).
- Block bootstrap por janela para ICs de Sharpe/CVaR (chaves
  `sharpe_bootstrap_ci_*`/`cvar_bootstrap_ci_*` em `window_reports`).

## 2. H4 — decomposição completa dos p-values

Resolução da contradição `STATE.md:3` (p=0.0018) vs `ROADMAP.md:81` (p<0.0001): **são dois
testes distintos, ambos da era TGN**, e nenhum deles corresponde à redação atual de H4
("TGAT ≥ baselines em ≥70% das janelas", `PROJECT.md:49-51`).

| Evidência | Teste | Comparação | Resultado | Fonte |
|---|---|---|---|---|
| v0.9 (BL-08) | Block bootstrap 10k, blocos 5d, Sharpe GMVP | TGN vs ROLAND, 30 ativos | P(TGN≤ROLAND)=**0.0018** ✅ | `doc/EXPERIMENT_LOG.md:297-299,512` |
| v0.9 (BL-08) | idem | TGN vs GAT-Static | P=0.337, n.s. | `doc/EXPERIMENT_LOG.md:299` |
| v1.0 (run `bootstrap_eval_v3_20260413_085728`) | Wilcoxon + DM (HAC), Holm | TGN vs ROLAND/GAT-Static, erro preditivo | **p<0.0001** ✅ | `doc/EXPERIMENT_LOG.md:543-546` |
| v1.0 (mesmo run) | Block bootstrap Sharpe/CVaR | TGN vs ROLAND | Sharpe P=0.595, CVaR P=0.289 — **NÃO suportada** | `doc/EXPERIMENT_LOG.md:557-559` |
| Era TGAT rev2 | Wilcoxon janelas, Sharpe | tgat vs tgn, 50 tickers, 9 janelas | 5/9 vitórias, p=0.715, n.s. | `results/bootstrap_eval_tkg_rev2_20260418_130703/…json` `primary_comparison` |
| Era TGAT rev3 | idem | tgat vs tgn | 4/9 vitórias, p=0.590, n.s. | `results/bootstrap_eval_tkg_rev3_20260420_141237/…json` |

Observações:
- O run `bootstrap_eval_v3_20260413_085728` citado no log **não existe mais em `results/`**
  (o diretório preserva execuções a partir de 2026-04-16); a única fonte é
  `doc/EXPERIMENT_LOG.md` (fora do git).
- O rótulo "(TGAT > Baselines)" em `ROADMAP.md:81` atribui ao TGAT um p-value obtido com
  TGN, em métrica preditiva (não Sharpe) — **erro de atribuição documental**.
- Critério "≥70% das janelas": win rate máximo observado tgat vs tgn = 5/9 ≈ 55,6%.

## 3. Métricas headline v1.0 — proveniência artefato a artefato

Tabela reivindicada em `.specs/project/PROJECT.md:96-103` (repetida em `ROADMAP.md:75-84`):
Test R²=0.824 · Sharpe GMVP=2.615 · MDD=12.4% · Turnover=0.085.

| Métrica | Valor alegado | Artefato localizado | Veredito |
|---|---|---|---|
| Test R² | 0.824 | `results/link_pred_tgat_s42_20260421_032602/results.json` → `test_r_squared=0.8240663912452635` (50 tickers, 2018-2024, 50 épocas, seed 42; mesmo valor = janela 2 do braço CORR+FACT em `results/bootstrap_eval_tkg_rev3_abl_full_tgat_20260420_214339/…json`) | ✅ verificado |
| Sharpe GMVP | 2.615 | Nenhum artefato com 2.615 exato. Mais próximos: `test_sharpe_proxy=2.618114` (`results/link_pred_tgat_s42_20260416_171937/results.json`, **30 tickers**, R²=0.7597) e `2.611735` (`results/link_pred_tgat_s42_20260417_090713/results.json`, **30 tickers**, R²=0.7576); média = 2.6149 | ⚠️ plausivelmente arredondamento/média de runs 30-tickers; como número único, `[NÃO VERIFICADO — alegado em PROJECT.md:101]` |
| MDD | 12.4% | Nenhum artefato em `results/` contém MDD ≈ 0.124 (busca exaustiva em JSONs) | `[NÃO VERIFICADO — alegado em PROJECT.md:102]` |
| Turnover | 0.085 | Nenhum artefato contém turnover ≈ 0.085 | `[NÃO VERIFICADO — alegado em PROJECT.md:103]` |

**Achado importante:** a linha "TGAT v1.0" da tabela é um **composto de execuções
distintas** — o R² vem de um run de 50 tickers cujo próprio Sharpe é 1.536, MDD 5.55% e
turnover 0.123 (`link_pred_tgat_s42_20260421_032602/results.json`), enquanto o Sharpe ~2.61
vem de runs de 30 tickers com R²≈0.76. Não existe um único artefato que produza as quatro
métricas simultaneamente.

**Nota BL-27/BL-30:** todos esses artefatos são anteriores ou contemporâneos ao fix
edge_dim (2026-04-21, `.specs/quick/027-tgat-edge-dim-fix/TASK.md`); checkpoints pré-fix
são **incompatíveis** com o TGAT v2 e "todos os experimentos precisam ser re-rodados"
(`STATE.md:18-19`). A re-ablação BL-30 está em andamento (`ROADMAP.md:95`), com runs
parciais em `results/bootstrap_eval_tkg_rev3_abl_full_tgat_*`.

## 4. Catálogo de resultados em `results/`

| Família | Diretórios | Conteúdo |
|---|---|---|
| `bootstrap_eval_tkg_rev2_*` | 3 (2026-04-16 → 04-18) | tgn/tgat/roland/gat_static; 30tk (5000 boot) e 50tk (2000 boot) |
| `bootstrap_eval_tkg_rev3_*` | ~22 (2026-04-20 → 05-02) | inclui persistence/ewma/zero/delta_ewma; 30/50/100 tickers |
| `bootstrap_eval_tkg_rev3_abl_full_tgat_*` | 4 | re-ablação TGAT v2 (BL-30), inclui multi-seed [42,123,456] |
| `link_pred_<variant>_s<seed>_*` | 300+ | runs individuais de link prediction (tgat/tgn/ra_htgn/temporal_kg/gat_static/roland/persistence/ewma/zero/delta_ewma) |
| `paper_abllation_tgat_*` | 9 | braços de ablação de edge types para o paper |
| `scaling_experiment_20260417_222449` | 1 | BL-22, escala 30/50/100 |
| `stress_event_compare/` | 9 JSONs | pares SPY/VIX, SPY/GLD, SPY/TLT, SPY/BTC, QQQ/BTC, GLD/BTC, XLE/SPY, XLK/SPY |
| `covid_compatible_forecast_experiment`, `smoketest_covid_*` | — | estudos de evento COVID |
| `dyfo_drl_walkforward*` | 6 | ambiente DRL de teste (input DyFO vs EWMA) |
| `probe_*` | 3 arquivos | probe diagnóstico sem leakage (§6) |
| `mdd_turnover_20260418_180825` | 1 | MDD/turnover pós-hoc (apenas roland/gat_static) |

### 4.1 Resultado central do harness rev3 (50 tickers, 9 janelas, seed 42)

`results/bootstrap_eval_tkg_rev3_20260501_200449/bootstrap_summary_tkg_rev3.json`
(médias de R² por variante nas 9 janelas):

| Variante | R² médio | Sharpe proxy médio |
|---|---|---|
| **ewma** | **0.9905** | 1.624 |
| tgat | 0.8871 | 1.608 |
| persistence | 0.8783 | 1.705 |

**Null honesto replicado no próprio harness canônico:** EWMA supera o TGAT em R² de
predição de ρ; a persistência empata na prática. Este resultado é consistente com o probe
(§6) e deve ser reportado junto com qualquer citação do R² de link prediction.

### 4.2 Escala (BL-22) — "50 ótimo"

`results/scaling_experiment_20260417_222449/scaling_summary.json`:

| Config | Test R² | Sharpe proxy |
|---|---|---|
| TGAT_30 | 0.7674 | 2.198 |
| TGAT_50 | **0.8972** | 1.416 |
| TGAT_100 | 0.8943 | 1.702 |

- A conclusão "50 ativos como ideal" (`ROADMAP.md:58`) apoia-se no R² máximo em 50; porém o
  bloco `statistics` do próprio artefato tem Wilcoxon `NaN` e DM p=1.0 (runs únicos, sem
  repetição) — a otimalidade de 50 **não tem suporte estatístico formal** no artefato.
- Em contraste, o run de 100 tickers no harness rev3 (1 janela,
  `results/bootstrap_eval_tkg_rev3_20260501_190349/…json`) registra **tgat R² = −4.47** —
  colapso em escala 100 nessa configuração (TMFG/esparsificação pendente, BL-26
  `ROADMAP.md:91`).

## 5. rho_conditioning — semântica real e default (commit c9e4c4d)

Commit `c9e4c4d` ("fix(decoder): add rho_conditioning to prevent cross-sectional
memorisation", 2026-06-14) — semântica verificada no código atual:

- **O que faz:** quando `use_rho_conditioning=True`, o decoder `CorrelationRegressor`
  recebe a correlação corrente do par `ρ_t` como escalar extra concatenado a
  `[z_i ‖ z_j]` (`dyfo/core/link_prediction.py:224-229` docstring; `:246` input_dim;
  `:276-283` forward). Motivação declarada: impedir que o decoder colapse numa constante
  por par ("cross-sectional memorisation").
- **Default: `False`** — em `CorrelationRegressor.__init__`
  (`dyfo/core/link_prediction.py:237`) e em `train_link_prediction()`
  (`scripts/train_link_prediction.py:281`).
- **Pipeline canônica roda SEM conditioning:** o runner rev3 chama
  `train_link_prediction(...)` sem passar o parâmetro
  (`scripts/run_bootstrap_eval_temporal_kg_rev3.py:253-271`) → False. O probe roda
  explicitamente com `USE_RHO_CONDITIONING = False` ("Ensure no leakage",
  `scripts/run_diagnostic_probe.py:19-20`).
- **Onde é True:** apenas em dois scripts:
  1. `scripts/run_smoketest_covid_tgat_plus_rho.py` — cria e rotula honestamente uma
     variante separada `tgat_plus_rho` ao lado de `tgat` puro e `persistence`
     (docstring `:9-13`). Este é o uso conforme o enquadramento científico
     ("leakage admissível só como baseline rotulado").
  2. `scripts/run_spy_vix_covid_compare.py:235` — ativa `use_rho_conditioning=True` na
     função `train_tgat_for_tickers_and_save_preds`, **mantendo o rótulo `"tgat"`**
     (`model_variant="tgat"`, sem renomear), e os JSONs de saída não registram o flag
     (verificado: nenhuma menção a rho_conditioning em
     `results/stress_event_compare/SPY_BTC_USD_metrics.json`).

**⚠️ DISCREPÂNCIA REGISTRADA (não resolvida aqui):** o caminho (2) produz as séries "tgat"
das figuras de stress event (`figures/stress_event_compare_*.png/pdf` — atualmente
modificadas e não commitadas no working tree) com o decoder condicionado em ρ_t, sem
rotulagem distinta. Isso diverge do enquadramento do briefing científico, que só admite
rho_conditioning como baseline explicitamente rotulado. Qualquer uso dessas figuras na
dissertação deve declarar o conditioning ou re-gerar as séries com o flag em False.

## 6. Probe diagnóstico sem leakage (o null central)

**Scripts (⚠️ UNTRACKED no git — ver `docs/TESTING.md §5`):**

- `scripts/run_diagnostic_probe.py` — roda o probe: 15 tickers (AAPL MSFT JNJ JPM XOM PG
  GOOGL META TSLA PFE V HD CVX ABBV KO, `:13-17`), 2016-01-01→2023-06-01, targets
  DCC-GARCH, duas janelas de teste de 125 dias — CALM (a partir de 2019-07-01, `:70-74`) e
  BREAK (a partir de 2022-01-01, `:76-81`) — com 500 dias de treino e 125 de validação
  cada; TGAT com 15 épocas, seed 42, `use_rho_conditioning=False` (`:154-176`).
  Baselines calculados diretamente da série: persistência (`pred[t]=ρ[t−1]`, `:132-134`) e
  EWMA com λ ∈ {0.80, 0.90, 0.94, 0.97} (melhor λ reportado, `:137-149`).
- `scripts/print_probe_results.py` — recomputa o relatório a partir dos CSVs de predição.
- `scripts/calc_sigma.py` — análise de σ(R²) entre janelas (exclui janela 8); os valores
  "Paper claims: DyFO σ=0.034, TGN σ=0.077" em `calc_sigma.py:83-84` são comentários
  hardcoded `[NÃO VERIFICADO — não recomputados nesta auditoria]`.

**Resultado (`results/probe_results.txt`):**

| Janela | Modelo | R² | MAE | Spearman |
|---|---|---|---|---|
| CALM | Persistence | 0.9901 | 0.0069 | 0.9914 |
| CALM | EWMA(λ=0.97) | 0.9901 | 0.0069 | 0.9914 |
| CALM | **DyFO (TGAT)** | **−5.3515** | 0.3089 | nan |
| BREAK | Persistence | 0.9894 | 0.0090 | 0.9919 |
| BREAK | EWMA(λ=0.97) | 0.9894 | 0.0090 | 0.9918 |
| BREAK | **DyFO (TGAT)** | **−4.3561** | 0.3108 | nan |

Headroom (DyFO − melhor naive): **−6.3416 (CALM), −5.3456 (BREAK)** — "DyFO loses to
persistence & EWMA" em ambas as janelas (`results/probe_results.txt:16-21`).

Notas de leitura:
- O Spearman `nan` do DyFO sugere predições (quase) constantes na janela de teste —
  exatamente o modo de falha "cross-sectional memorisation" que motivou o commit c9e4c4d.
  `[NÃO VERIFICADO em detalhe — os CSVs results/probe_CALM_dyfo_preds.csv /
  probe_BREAK_dyfo_preds.csv não foram analisados]`.
- A nota de interpretação do próprio artefato (`results/probe_results.txt:23-27`) registra
  que persistência/EWMA usam o alvo defasado ρ_t que o DyFO exclui — são réguas
  diagnósticas, e a pergunta relevante passa a ser degradação relativa em regime de quebra
  (o headroom continua fortemente negativo também na BREAK, logo o null se mantém).
- **Este null NUNCA deve ser maquiado com o R²=0.824 de link prediction** — ver
  `docs/OVERVIEW.md §5`.

## 7. Stress events e estudos de evento

`results/stress_event_compare/<PAR>_metrics.json` (9 pares; gerados por
`scripts/run_spy_vix_covid_compare.py`, cujo cabeçalho declara reporting "intentionally
honest: EWMA is expected to be very strong on smooth DCC R²" `:5-8`). Exemplo SPY–BTC-USD
(`SPY_BTC_USD_metrics.json`):

| Modelo | R² full_test | R² covid_crash | MAE crash |
|---|---|---|---|
| persistence | 0.660 | 0.492 | 0.0146 |
| tgat | 0.129 | 0.0415 | 0.0266 |
| ewma | 0.123 | −0.226 | 0.0340 |

- `"comparison": {"tgat_event_window_win": true}` — na janela do crash o TGAT supera o
  EWMA em MAE, embora perca para a persistência em tudo.
- **Caveat obrigatório:** a série `tgat` desses artefatos foi treinada com
  `use_rho_conditioning=True` sob rótulo não qualificado (§5, item 2).
- Estudos de evento COVID adicionais: `results/covid_compatible_forecast_experiment/`,
  `results/smoketest_covid_compare/`, figuras em `figures/`.

## 8. Ablação de edge types (contexto BL-27/BL-30)

- Diagnóstico pré-fix (`.specs/quick/027-tgat-edge-dim-fix/TASK.md`): CORR+FACT
  R²=0.8867 > all_edges R²=0.8825 — adicionar SECT **piorava** o R², sintoma de diluição
  de atenção homogênea; motivou `edge_dim` no GATConv (`tgat_encoder.py:249-256`).
- Braços de ablação e sumários: `results/paper_abllation_tgat_*` e
  `results/bootstrap_eval_tkg_rev3_abl_full_tgat_*` (`ablation_results` por braço:
  e.g. CORR+FACT `window_metrics`).
- Meta do BL-30: validar `all_edges ≥ CORR+FACT` no TGAT v2 (`STATE.md:5`;
  `ROADMAP.md:95`) — **em andamento**, sem veredito final nos artefatos auditados.
