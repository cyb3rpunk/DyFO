# DyFO — Backlog Unificado

> Fonte única de verdade para melhorias e próximos passos do módulo DyFO.
> Organizado por prioridade para o artigo e para a tese (MATTS).
> Referenciado pelo EXPERIMENT_LOG.md §4.2.

---

## Contribuições planejadas vs. TGN Original (Rossi et al., 2020)

Estas são as contribuições que diferenciam o DyFO e estão mapeadas
diretamente nos BL-items abaixo:

| Aspecto | TGN Original | DyFO (planejado) | BL item |
|---------|-------------|------------------|---------|
| **Correlações** | N/A | DCC-GARCH (Engle 2002) — ✅ implementado | BL-03 |
| **Ablation** | Sem comparação | B16: TGN vs ROLAND vs GAT-Static | BL-02 |
| **Regime conditioning** | Não existe | regime_prob como node feature do RDM (HMM-GAS) | BL-09 |
| **Scalability** | ~10K nós | 30 ativos financeiros com grafos esparsos — ✅ implementado | BL-01 |
| **Statistical validation** | Single run | 500-bootstrap sobre walk-forward | BL-08 |

---

## P0 — Crítico para o Artigo

### BL-01: Escalar para 30–50 ativos
**Status:** ✅ Implementado (30 ativos)
**Justificativa:** Com 10 ativos temos apenas C(10,2)=45 pares — trivial para o modelo.
Com 30 temos 435 pares, com 50 temos 1225 — grafos mais esparsos e link prediction
realmente desafiador. Resultados com 10 ativos não são publicáveis como validação principal.
**Implementação (2026-03-26):**
- 30 tickers S&P 500 por liquidez, cobrindo todos os 11 setores GICS
- v0.6: Test R²=0.628, Spearman=0.877 (degradação mínima vs N=20)
- 435 pares, 351K events, DCC-GARCH 30/30 OK
**Dependência:** BL-05 (integridade de dados) — ✅ satisfeita.

### BL-02: Implementar baselines (ROLAND, GAT-Static)
**Status:** 🔴 Pendente
**Justificativa:** Essencial para ablation B16. Sem baselines, o artigo não tem comparação.
**Ação:**
- ROLAND: EvolveGCN-H sobre snapshots mensais (substituir apenas M2)
- GAT-Static: GAT sobre correlação média do período, sem memória
**Referência:** Manual §6.1, You et al. (2022), Pareja et al. (2020)

### BL-03: Substituir Pearson por DCC-GARCH
**Status:** ✅ Implementado
**Justificativa:** Manual é explícito: "NÃO usar Pearson simples" (§7.1 checklist).
Rolling Pearson é fallback. DCC-GARCH (Engle 2002) captura correlações time-varying
e é o padrão em financial econometrics. Pacote `arch` v8.0.0 já instalado.
**Implementação (2026-03-26):**
- `compute_dcc_garch_correlations()` reescrita com DCC(1,1) completo:
  Step 1: GARCH(1,1) per asset → residuals ε_t (`arch` package)
  Step 2: MLE estimation of (a, b) via grid search + L-BFGS-B (`scipy`)
  Step 3: DCC recursion Q_t = (1-a-b)Q̄ + a(ε_{t-1}ε_{t-1}') + bQ_{t-1}
  Step 4: R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}, sparsification
- `config.py`: campo `correlation_method` ("dcc_garch" | "rolling_pearson")
- `train_link_prediction.py`: DCC computado uma vez, sparsificação como pós-processamento
- Fallback automático para rolling Pearson se GARCH falhar em >50% dos ativos
**Impacto:** Melhora qualidade das arestas CORR e dos labels de link prediction.

### BL-04: Corrigir viés precision/recall
**Status:** ✅ Resolvido (consequência de BL-03)
**Justificativa original:** Modelo prediz "sim" para ~97% dos pares (recall ~97%, precision ~46%).
Causa raiz: com corr_threshold=0.3, 100% dos pares do S&P 500 são positivos.
**Opções investigadas:**
- ❌ v0.3a: Focal loss (α=0.25, γ=2.0) — AUC→0.55 (descartado)
- ❌ v0.3b: pos_weight=0.5 — AUC→0.66 (descartado)
- ❌ v0.3c: neg_ratio=3.0 — AUC→0.62 (descartado)
- ❌ v0.4: Regressão de ρ contínua com Pearson — test R²=-2.08 (overfitting)
- ✅ **v0.5: Regressão + DCC-GARCH labels** — test R²=0.65, Spearman=0.91, cls F1=0.83
**Resolução:** O viés era causado por labels ruidosos (rolling Pearson), não por
arquitetura ou loss function. Com DCC-GARCH (BL-03), a regressão contínua generaliza
e a classificação derivada (threshold @0.5) atinge Precision=0.72, Recall=0.99, F1=0.83.
Nenhuma implementação adicional necessária.

### BL-08: Validação estatística com bootstrap
**Status:** 🔴 Pendente
**Justificativa:** Manual §7.4 exige "500 bootstraps sobre os resultados do ablation".
**Ação:** Após BL-02 (baselines), rodar 500 re-samplings com IC 95% para cada métrica.
**Dependência:** BL-02 (precisa de baselines para comparar).

---

## P1 — Importante para o Artigo

### BL-05: Integridade de dados com retry e validação
**Status:** ✅ Implementado + Auditado
**Ação realizada:** Retry com exponential backoff (3 tentativas, base 2s) em todos os
adapters (yfinance + FRED). Logging detalhado de falhas.
**Auditoria (2026-03-25):** 20 tickers, 100% cobertura de preços, 380 earnings,
316 corporate actions, 8 séries FRED. Sem gaps críticos.

### BL-06: Aumentar epochs para 15–20 com early stopping
**Status:** ✅ Implementado (patience=5)

### BL-07: Threshold tuning no conjunto de validação
**Status:** ✅ Implementado (grid search -2.0 a 2.0 em 41 passos)

### BL-11: Factor edges (FACT / Fama-French)
**Status:** ✅ Implementado
**Justificativa:** OLS loading distance conecta ativos com exposição similar a fatores.
**Implementação (2026-03-27):**
- Novo módulo `dyfo/data/ff_adapter.py`: download/cache dos fatores diários FF5 (2×3) da Ken French Data Library
- `compute_factor_edges()` ativado no pipeline: OLS loadings (janela=252d), threshold L2<0.50
- 22 pares FACT → 44 arestas bidirecionais no grafo estático (total: 108 edges vs 64 sem FACT)
- Edge features = |β_i − β_j| (dim=5, diferença absoluta dos 5 loadings fatoriais)
- Hardening do DCC-GARCH: filtra tickers com resíduos insuficientes antes do Step 2
- v0.7: Test R²=0.806 (+0.178 vs v0.6), Spearman=0.931 (+0.054), MAE=0.049 (−0.020)

---

## P2 — Importante para a Tese (MATTS)

### BL-09: Integração com RDM (regime probabilities)
**Status:** 🔴 Pendente
**Justificativa:** Node features incluem regime_prob (3-dim), atualmente zero-filled.
Com o módulo RDM (HMM-GAS-TVTP) do MATTS, esses 3 dims seriam preenchidos com
π_t do regime detector — contribuição original da tese.
**Dependência:** Módulo 1 (RDM) implementado.

### BL-10: Supply chain edges (SUPL)
**Status:** 🔴 Pendente (stub no código)
**Justificativa:** Manual §2.3 define SUPL como tipo de aresta do grafo heterogêneo.
**Ação:** Carregar relações fornecedor-cliente de fonte externa (FactSet ou OpenCorporates).

### BL-12: Staleness proxy implementation
**Status:** 🟡 Documentado, não implementado
**Justificativa:** Manual §2.5 — injetar PRICE_UPDATE sintético após 5 dias sem evento.
**Impacto:** Baixo para 20 ativos (todos ativos diariamente), alto para universos maiores.

---

## P3 — Futuro / Nice-to-have

### BL-13: Multi-window walk-forward
**Status:** 🔴 Pendente
**Justificativa:** Pseudo-código do ablation B16 (§6.3) prevê walk-forward sobre
múltiplos datasets (SP500, MSCI, commodities, crypto, FF5).

### BL-14: Distributed training / GPU optimization
**Status:** 🔴 Pendente
**Justificativa:** Atualmente single CPU. Com 50 ativos e DCC-GARCH, o tempo
de treinamento pode ser proibitivo.

### BL-15: Downstream task — portfolio optimization
**Status:** 🔴 Pendente
**Justificativa:** O embedding e_t gerado pelo TGN alimenta o State Constructor (M3)
do MATTS. A evaluation downstream final é Sharpe ratio / CVaR do portfólio.
**Dependência:** Módulos 3-5 do MATTS.

### BL-16: Report com gráficos e visualização do grafo/ontologia
**Status:** ✅ Implementado (parcial, foco em curvas de treino)
**Justificativa:** Gerar relatório visual com:
- Curvas de loss (train/val) e métricas (R², Spearman, MAE) por época
- Visualização do grafo heterogêneo (nós=ativos, arestas=CORR/SECT/FACT/SUPL com cores distintas)
- Heatmap da matriz de correlação DCC-GARCH
- Distribuição dos edge features e node features
- Diagrama da ontologia financeira (tipos de nó/aresta)
**Implementação (2026-04-06):**
- `scripts/plot_results.py` corrigido para suportar runs de regressão e classificação
- Plot dinâmico por modo (métrica principal/auxiliar, MAE para regressão, métricas de classificação)
- Painel-resumo de métricas consolidado e geração validada de `training_results.png`
**Escopo restante (opcional):** visualização de ontologia/grafo/heatmap avançado pode evoluir em item dedicado.
**Dependência:** Nenhuma (base funcional entregue).

### BL-17: Diagnóstico de instabilidade de treino (R²)
**Status:** 🟡 Em andamento
**Justificativa:** Em alguns runs houve colapso temporário de validação (R² negativo em épocas intermediárias)
mesmo com recuperação posterior. Isso reduz previsibilidade do treinamento e dificulta comparar variantes.
**Ação (2026-04):**
- Consolidar análise dos runs recentes (v0.7 e pós-v0.7) com foco em curvas por época (loss, R², Spearman, LR)
- Registrar hipóteses de causa (LR alto efetivo, oscilação por mini-batch temporal, pouca paciência)
- Definir critérios objetivos de estabilidade para promoção de configuração (ex.: sem colapso >2 épocas)

### BL-18: Corrigir `plot_results.py` para regressão e classificação
**Status:** ✅ Implementado
**Justificativa:** O script de visualização assume métricas de classificação (`auc`, `f1`, etc.) e quebra
em runs de regressão (`KeyError: 'auc'`), bloqueando parte do BL-16.
**Implementação (2026-04-06):**
- Plot condicional por modo implementado (regressão: R²/MAE/Spearman; classificação: AUC/F1/Precision/Recall)
- Painel de resumo atualizado para métricas disponíveis em cada modo
- Validação executada em run de regressão sem `KeyError`

### BL-19: Estabilização via gradient clipping + LR scheduler
**Status:** 🟡 Implementado e reavaliado no BL-20 (não promovido para default)
**Justificativa:** Configuração adicionada no treino para mitigar oscilações e melhorar convergência.
**Implementação (2026-04-06):**
- `grad_clip_enabled` e `grad_clip_max_norm` expostos em `train_link_prediction.py`
- `ReduceLROnPlateau` integrado (fator, paciência, threshold, cooldown, min_lr configuráveis)
- Histórico expandido com `lr` e `val_score`; persistência de `scheduler_state` no checkpoint
- Scripts de A/B curto e completo criados (`scripts/ab_test_bl19_short.py`, `scripts/ab_test_bl19_full.py`)
**Evidência consolidada (2026-04-06):**
- Triagem curta (10 ativos, 3 épocas): resultados inconsistentes, sem ganho robusto em teste.
- BL-20 (30 ativos, 10 épocas, seeds 42/123/777) salvo em `results/ab_bl19_full_20260406_181547.json`.
- Delta médio (scheduler - baseline): `R²` +0.0265, `Spearman` +0.0211, `MAE` -0.0020, mas com alta variância.
- Win-count por seed: scheduler venceu só 1/3 seeds em `R²`, `Spearman` e `MAE`.
**Conclusão:** manter `scheduler_enabled=False` como default por ora; ativar scheduler apenas em estudos específicos.

### BL-20: A/B completo de robustez (30 ativos, múltiplas seeds)
**Status:** 🟡 Executado (1ª rodada), requer rerun controlada
**Justificativa:** A rodada completa foi concluída, mas houve falhas intermitentes de download
(degradando o universo efetivo em parte dos runs), o que reduz a comparabilidade entre pares A/B.
**Rodada executada (2026-04-06):**
- Configuração: 30 ativos, 10 épocas, 3 seeds (42/123/777), modo regressão.
- Resultado agregado: `results/ab_bl19_full_20260406_181547.json`.
- Síntese: sem evidência consistente para promover scheduler como default.
**Próxima ação (fechamento do BL-20):**
- Repetir A/B com cache local fixo/snapshot de dados para congelar o universo por seed.
- Exigir vitória da variante em maioria de seeds por métrica principal (`R²`, `Spearman`, `MAE`).
**Dependência:** BL-19.

### BL-21: Gate de regressão para preservar baseline v0.7
**Status:** 🔴 Pendente
**Justificativa:** v0.7 atingiu referência forte (Test R²=0.806, Spearman=0.931). Mudanças novas
precisam de proteção explícita contra regressão de desempenho.
**Ação:**
- Definir baseline oficial (run/config/dados) e registrar no log experimental
- Criar checklist de aceitação mínima para novas mudanças (ex.: não degradar além de tolerância definida)
- Exigir comparação lado a lado com v0.7 antes de consolidar alterações de treino
**Dependência:** BL-17 e BL-20.

---

## Sequência de Execução Recomendada

```
BL-03 (DCC-GARCH)    ← Melhora labels e arestas           ✅
    ↓
BL-01 (30+ ativos)   ← Problema discriminante + publicável  ✅
    ↓
BL-11 (FACT edges)   ← Enriquece grafo heterogêneo          ✅
    ↓
BL-02 (baselines)    ← Ablation B16 — core do paper
    ↓
BL-08 (bootstrap)    ← Robustez estatística
```

BL-04 (viés) resolve-se como consequência de BL-03 + BL-01.

---

## Referências Rápidas

| Item | Manual § | Hipótese | Paper |
|------|----------|----------|-------|
| BL-01 | §5.1 | — | — |
| BL-02 | §6.1-6.3 | H4 | You et al. 2022 (ROLAND) |
| BL-03 | §7.1 | — | Engle 2002 |
| BL-04 | — | — | Lin et al. 2017 (focal) |
| BL-08 | §7.4 | — | — |
| BL-09 | §2.2, Compl.3 | H4 | — |
| BL-10 | §2.3 | — | TAGN 2026 |
| BL-11 | §2.3 | — | Korangi 2024 |
| BL-12 | §2.5 | — | GAP-TGN 2026 |
| BL-16 | — | — | — |
| BL-17 | — | — | — |
| BL-18 | — | — | — |
| BL-19 | — | — | — |
| BL-20 | — | — | — |
| BL-21 | — | — | — |
