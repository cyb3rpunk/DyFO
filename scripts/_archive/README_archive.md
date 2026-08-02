# scripts/_archive — Memória de Scripts Obsoletos

> Scripts movidos para cá foram supersedidos pelo runner canônico
> (`run_bootstrap_eval_temporal_kg_rev3.py`) ou representam experimentos
> pontuais concluídos. **Nenhum deve ser executado em produção.**
> São mantidos como memória metodológica caso algum resultado precise ser
> reproduzido ou auditado.

---

## Por que arquivar (e não deletar)?

- Os scripts contêm lógica de experimentos encerrados (BL-22, BL-19, etc.)
  que pode ser necessária para auditar resultados históricos.
- Alguns foram citados no paper draft — deleta-los impediria reprodução.
- Git preserva o histórico, mas ter o arquivo local facilita leitura rápida.

---

## Inventário

| Script | Razão do arquivo | Experimento/BL relacionado |
|---|---|---|
| `run_bootstrap_eval.py` | Versão v1, supersedida por rev3 | BL-08 original |
| `run_bootstrap_eval_v2 copy.py` | Cópia suja do v2 — nunca deveria ter existido | — |
| `run_bootstrap_eval_v2.py` | Supersedido por rev3 | BL-08 v2 |
| `run_bootstrap_eval_v3.py` | Supersedido por rev3 | BL-08 v3 |
| `run_bootstrap_eval_v4.py` | Supersedido por rev3 | BL-08 v4 |
| `run_bootstrap_eval_ra_htgn_rev1.py` | Supersedido por `run_bootstrap_eval_ra_htgn.py` | BL-17 rev1 |
| `run_bootstrap_eval_temporal_kg.py` | Supersedido por rev3 | BL-18 v1 |
| `run_bootstrap_eval_temporal_kg_rev1.py` | Supersedido por rev3 | BL-18 rev1 |
| `run_bootstrap_eval_temporal_kg_rev2.py` | Supersedido por rev3 | BL-19 rev2 |
| `run_scaling_experiment.py` | Experimento BL-22 concluído (50 ativos escolhido) | BL-22 |
| `run_smoketest_covid_compare.py` | Smoke test one-off concluído | TGAT-EWMA T8 |
| `run_smoketest_covid_tgat_plus_rho.py` | Smoke test one-off com rho_conditioning | TGAT-EWMA T8 |
| `run_covid_compatible_forecast_experiment.py` | Experimento isolado de forecast covid | — |
| `run_statistical_validation.py` | Validação estatística ad-hoc, integrada no rev3 | BL-24 |
| `run_walk_forward.py` | Walk-forward v1, supersedido por DRL walkforward | BL-08 wf |
| `run_wf_dcc_baselines.py` | Baselines DCC walk-forward, integrados no rev3 | BL-03 |
| `run_dyfo_walkforward_rev0.py` | Rev0 do DRL walkforward, supersedido | DRL T9 rev0 |
| `abllation_test.py` | Typo no nome; ablation test inicial, supersedido | BL-08 ablation |

---

*Última atualização: 2026-08-02. Movidos durante triage geral do repositório.*
