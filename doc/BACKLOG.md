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
| **Scalability** | ~10K nós | 30-50 ativos financeiros com grafos esparsos | BL-01 |
| **Statistical validation** | Single run | 500-bootstrap sobre walk-forward | BL-08 |

---

## P0 — Crítico para o Artigo

### BL-01: Escalar para 30–50 ativos
**Status:** 🔴 Pendente
**Justificativa:** Com 10 ativos temos apenas C(10,2)=45 pares — trivial para o modelo.
Com 30 temos 435 pares, com 50 temos 1225 — grafos mais esparsos e link prediction
realmente desafiador. Resultados com 10 ativos não são publicáveis como validação principal.
**Ação:** Usar S&P 500 top 30 por liquidez, cobrindo todos os setores GICS.
**Dependência:** BL-05 (integridade de dados).

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
**Status:** 🔴 Pendente (código implementado, dados não carregados)
**Justificativa:** OLS loading distance conecta ativos com exposição similar a fatores.
`compute_factor_edges()` existe em `edge_features.py` mas nunca foi chamado com dados reais.
**Ação:** Baixar FF5 factors de Ken French website, alimentar pipeline.

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

---

## Sequência de Execução Recomendada

```
BL-03 (DCC-GARCH)    ← Melhora labels e arestas
    ↓
BL-01 (30+ ativos)   ← Problema discriminante + publicável
    ↓
BL-02 (baselines)    ← Ablation B16 — core do paper
    ↓
BL-08 (bootstrap)    ← Robustez estatística
    ↓
BL-11 (FACT edges)   ← Enriquece grafo (nice-to-have para paper)
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
