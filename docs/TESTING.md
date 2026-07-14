# DyFO — Testes e Reprodutibilidade (TESTING)

> Convenções de citação e proveniência: ver cabeçalho de `docs/OVERVIEW.md`.

---

## 1. Suíte de testes — 11 arquivos, 32 testes

`pyproject.toml` define `[tool.pytest.ini_options] testpaths = ["tests"]`. Coleta
verificada nesta auditoria: **32 testes** (`python -m pytest tests/ --collect-only -q`).

| Arquivo | Testes | Cobre |
|---|---|---|
| `tests/test_smoke.py` | 2 | forward pass ponta-a-ponta de todos os módulos (`test_forward_pass:10`) e as 3 estratégias de readout (`test_readout_strategies:70`) |
| `tests/test_model_variants.py` | 4 | aceitação de `ra_htgn`/`temporal_kg` no config, rejeição de variante desconhecida, factory `build_encoder` (`:9-24`) |
| `tests/test_relation_aware_tgn.py` | 5 | BL-17: agregação intra-relação por média, mensagens agrupadas por evento, relações estáticas, atualização de memória/atenção, edge features reais no embedding (`:11-199`) |
| `tests/test_relation_semantic_attention.py` | 2 | pesos densos para grupos ativos; fusão identidade com grupo único (`:6-26`) |
| `tests/test_temporal_kg.py` | 2 | BL-18: determinismo do adapter eventos→fatos; export de artefatos de interpretabilidade (`:9-29`) |
| `tests/test_delta_rho_target.py` | 5 | modo delta-target: labels Δρ só para pares presentes em t e t+1, saída linear do regressor em modo delta, métricas de ρ reconstruído, aceitação no config (`:11-58`) |
| `tests/test_bootstrap_eval_temporal_kg.py` | 2 | contrato do runner temporal_kg: variantes/pares declarados e entrypoint existente (`:8-24`) |
| `tests/test_bootstrap_eval_ra_htgn.py` | 2 | idem para o runner ra_htgn (`:8-17`) |
| `tests/test_dyfo_drl_walkforward_protocol.py` | 4 | protocolo DRL: janelas walk-forward disjuntas, sumário pareado janela×seed×episódio, block bootstrap sobre diferenças diárias, PPO value baseline (`:19-83`) |
| `tests/test_smoke_portfolio_drl.py` | 4 | smoke do otimizador DRL com 3 representações de estado (DyFO / raw / EWMA-GMVP) e comparabilidade entre condições (`:389-419`) |
| `tests/test_real_data.py` | — | **integração** com dados reais de mercado (docstring `:1-5`); requer rede/APIs — lento, não é unit test |

### Como rodar

```bash
# suíte rápida (exclui integração com rede)
python -m pytest tests/ -q --ignore=tests/test_real_data.py

# suíte completa (baixa dados reais — yfinance/FRED)
python -m pytest tests/ -q
```

Não há CI: o repositório **não possui `.github/workflows/`** — os testes só rodam
manualmente.

## 2. Runners de avaliação — qual é o canônico (rev2 vs rev3)

**Contradição documental registrada:**

- `.specs/codebase/TESTING.md:12` — `run_bootstrap_eval_temporal_kg_rev2.py` = "Runner
  principal ... ✅ Ativo" (documento datado 2026-04-19).
- `.specs/project/ROADMAP.md:68` — "Runner ativo: `run_bootstrap_eval_temporal_kg_rev3.py`"
  (e BL-28/BL-29 citam o rev3 explicitamente, `ROADMAP.md:62-63`).

**Veredito: o rev3 é o canônico.** Evidências: (a) só o rev3 tem `--seeds` (BL-28,
`run_bootstrap_eval_temporal_kg_rev3.py:951-954`); (b) os artefatos mais recentes são
todos rev3 (até 2026-05-02, vs rev2 até 2026-04-18); (c) `ROADMAP.md:69` congela
explicitamente "rev1/rev2" como não-editáveis. O `.specs/codebase/TESTING.md` ficou
desatualizado (escrito antes do rev3). Nota menor: a docstring do próprio rev3 ainda diz
"Revision 2" no topo (`run_bootstrap_eval_temporal_kg_rev3.py:1` — header copiado do
rev2), mas o código se autoidentifica como Rev 3 internamente (`:778` "Rev 3: Added
financial risk metrics...", `:896`).

Hierarquia dos runners (47 scripts em `scripts/`):

| Runner | Status | Papel |
|---|---|---|
| `run_bootstrap_eval_temporal_kg_rev3.py` | **canônico** | walk-forward + bootstrap + Wilcoxon/DM/Holm + métricas de risco + `--seeds`/`--ablation` |
| `run_bootstrap_eval_temporal_kg_rev2.py` | congelado (`ROADMAP.md:69`) | geração dos artefatos rev2 (2026-04-16→18) |
| `run_bootstrap_eval_v5.py` | legado (H4 era TGN) | protocolo v5; **não editar** (`ROADMAP.md:69`) |
| `train_link_prediction.py` | biblioteca/CLI base | treino de 1 variante; chamado pelos runners |
| `compute_mdd_turnover_full.py` | pós-hoc | MDD/Turnover a partir de sumários bootstrap |
| `run_diagnostic_probe.py` | ⚠️ untracked | probe do null honesto (`docs/EXPERIMENTS.md §6`) |

Parâmetros reais de bootstrap (2000 iterações, blocos de 5) vs os "10k/10" reivindicados:
ver `docs/EXPERIMENTS.md §1.2`.

## 3. Smoke e sanidade fora do pytest

- `scripts/run_smoketest_covid_compare.py` e `run_smoketest_covid_tgat_plus_rho.py` —
  smoketests de estudo de evento (o segundo é o padrão correto de rotulagem
  `tgat_plus_rho`, `docs/EXPERIMENTS.md §5`).
- `colab_ablation_test.ipynb` / `colab_run_bootstrap_eval.ipynb` — execução em Colab
  `[NÃO VERIFICADO — notebooks não auditados]`.

## 4. Gaps de reprodutibilidade — scripts do probe UNTRACKED

Os **3 scripts que produzem/leem a evidência científica mais importante do repo** (o null
do forecast de ρ sem leakage) estão **fora do controle de versão** (`git status` →
"Untracked files"):

1. `scripts/run_diagnostic_probe.py` — gera o probe CALM/BREAK;
2. `scripts/print_probe_results.py` — recomputa o relatório dos CSVs;
3. `scripts/calc_sigma.py` — análise de σ(R²) entre janelas.

Consequência: um clone limpo do repositório **não consegue reproduzir**
`results/probe_results.txt` (que também não está no git — ver §5). Se o disco local for
perdido, o null honesto central da tese perde tanto o gerador quanto o artefato.
**Recomendação: commitar os 3 scripts imediatamente** (são pequenos e não contêm
segredos).

## 5. Gaps de reprodutibilidade — o que está fora do git

Verificado via `.gitignore`, `git ls-files` e `git show`:

| Item | Estado | Evidência | Severidade |
|---|---|---|---|
| `.specs/` (PROJECT/STATE/ROADMAP/ARCHITECTURE/TESTING + features/quick) | **nunca versionado** (`git ls-files .specs` vazio; no `.gitignore`) | toda a "fonte da verdade" documental do projeto existe só no disco local, sem histórico auditável | **crítica** |
| `results/` (1554 arquivos, 541 subdirs) | **nunca versionado** (idem) | TODOS os artefatos que sustentam H4, R²=0.824, probe, ablações | **crítica** |
| `doc/` (13 arquivos, incl. `EXPERIMENT_LOG.md` — fonte única do p=0.0018) | **removido do tracking** no commit `1a0c65f` ("chore: remove doc/ from tracking and add to gitignore", −5032 linhas) | existe no disco, fora do git | **alta** |
| 3 scripts do probe (§4) | untracked | git status | **alta** |
| Figuras de stress event | `figures/stress_event_compare_SPY_BTC_USD.{pdf,png}` e `wf_dcc_baseline_comparison.{pdf,png}` **modificadas e não commitadas** | git status | média |
| `.env` (FRED_API_KEY) | corretamente fora do git, mas há default hardcoded em `config.py:91` | `docs/DATA.md §1.2` | média (segurança) |

Implicação para a dissertação: qualquer número citado destes caminhos deve ser tratado
como **artefato local não auditável por terceiros**. Mitigação mínima recomendada:
(i) commitar `.specs/`, `doc/` e os 3 scripts do probe; (ii) versionar ao menos os JSONs
de sumário (`bootstrap_summary_*.json`, `results.json`, `probe_results.txt`) — são
pequenos; (iii) registrar hash/data dos caches de dados usados em cada run.

## 6. Reprodutibilidade dos experimentos em si

- Seeds: fixadas por run (`params.seed` nos `results.json`); multi-seed disponível no
  rev3, mas o maior run observado usou 3 seeds, não os 5 anunciados
  (`docs/EXPERIMENTS.md §1.3`).
- Checkpoints: `best_model.pt` por run de link prediction; **checkpoints pré-BL-27 são
  incompatíveis** com o TGAT v2 (`STATE.md:18-19`) — qualquer resultado pré-2026-04-21
  não é recarregável no código atual (re-ablação BL-30 pendente).
- Dados: re-download a cada execução sem pin de versão do provedor
  (`docs/DATA.md §7`); caches `.pkl` locais não versionados.
- `mdd_turnover.log` (77 KB, 2026-04-18) registra erros de carregamento de checkpoint
  durante o cômputo de MDD/Turnover — coerente com a quebra de compatibilidade BL-27
  `[NÃO VERIFICADO em detalhe — log não analisado linha a linha]`.
