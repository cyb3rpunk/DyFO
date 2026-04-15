# 02 — Especificação do Grafo Financeiro

> Design completo do grafo G=(V,E): nós, arestas tipadas, features e invariantes.

---

## Nós (Ativos)

Cada ativo do portfólio é um nó. O universo é **fixo por dataset** durante o treinamento.

| Dataset | Nós | Tipo |
|---------|-----|------|
| S&P 500 (atual) | 30 tickers | Ações (11 setores GICS) |
| MSCI World ETFs | ~50 ETFs | ETF |
| Commodities futures | ~30 contratos | Futuros |
| Cripto top-20 | 20 tokens | Cripto |
| Fama-French 5 | 5 portfolios | Fator sintético |

**Inductive setting:** novos ativos podem ser adicionados sem re-treino (TGN inicializa
memória com `s_i(0) = 0`).

### 30 Tickers atuais (v0.7)
Cobertura de todos os 11 setores GICS. Selecionados por liquidez (top S&P 500).

---

## Features de Nó `v_i(t)` — dim=20

```
v_i(t) = [
    retorno_log_21d,    # 1  — retorno log acumulado 21 dias
    vol_hist_21d,       # 1  — volatilidade histórica 21 dias (anualizada)
    beta_mercado,       # 1  — beta vs. SPY (janela 63d)
    setor_one_hot,      # 11 — codificação GICS Sector (11 setores)
    market_cap_norm,    # 1  — log(market_cap) normalizado pelo universo
    drawdown_atual,     # 1  — drawdown corrente desde última máxima histórica
    regime_prob,        # K  — π_t do RDM (K=3, zero-filled se M1 ausente)
    volume_norm,        # 1  — volume relativo à média 21d (proxy de liquidez)
]
# Total: 1+1+1+11+1+1+3+1 = 20 dims (com K=3)
```

**Nota de implementação:** `regime_prob` é o acoplamento formal com M1.
Enquanto M1 não estiver disponível, usar `[0, 0, 0]` (zero-filled).
Isso é intencional — não usar valores aleatórios ou uniformes como placeholder.

---

## Tipos de Aresta (Grafo Heterogêneo)

O DyFO implementa **4 tipos de aresta** com features e frequências distintas:

### CORR — Correlação Dinâmica
```
Código:     CORR
Construção: DCC-GARCH ρ_ij(t) com esparsificação
Features:   [ρ_ij, CI_low, CI_high]   dim=3
Frequência: Diária
Threshold:  |ρ| > 0.3 (ou TMFG para universos >100 ativos)
Direcional: Bidirecional (aresta (i,j) e (j,i))
```

**Algoritmo DCC-GARCH:**
```
Step 1: GARCH(1,1) por ativo → resíduos ε_t  (pacote arch)
Step 2: MLE de (a, b) via grid search + L-BFGS-B  (scipy)
Step 3: Q_t = (1-a-b)Q̄ + a(ε_{t-1}ε_{t-1}') + bQ_{t-1}
Step 4: R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
Step 5: Esparsificação (|ρ| > threshold)
```

**Fallback:** Se DCC-GARCH falhar em >50% dos ativos → rolling Pearson (janela 63d).
Logar o fallback explicitamente.

### SECT — Setor Compartilhado
```
Código:     SECT
Construção: GICS Sector_i == GICS Sector_j
Features:   [1.0]   dim=1 (binária)
Frequência: Estática (não muda durante o episódio)
Threshold:  —  (presente ou ausente)
Direcional: Bidirecional
```

### SUPL — Cadeia de Fornecimento
```
Código:     SUPL
Construção: FactSet Supply Chain (ou OpenCorporates)
Features:   [força_do_vínculo]   dim=1
Frequência: Trimestral
Status:     🔴 Stub no código — fonte de dados não integrada
```

### FACT — Co-movimento de Fator (Fama-French 5)
```
Código:     FACT
Construção: |loading_FF5_i - loading_FF5_j|₂ < threshold
Features:   |β_i - β_j|   dim=5 (diferença absoluta dos 5 loadings)
Frequência: Diária (rolling OLS, janela=252d)
Threshold:  L2 < 0.50
Direcional: Bidirecional
```

**Fatores FF5:** Mkt-RF, SMB, HML, RMW, CMA.

**Implementação:** `dyfo/data/ff_adapter.py` — download/cache dos fatores diários da
Ken French Data Library. OLS por ativo sobre janela de 252 dias.

---

## Estatísticas do Grafo (v0.7 — 30 ativos)

| Tipo | Pares | Arestas (bidir.) |
|------|-------|-----------------|
| CORR | ~189 | ~378 |
| SECT | ~varia | ~varia |
| FACT | 22 | 44 |
| **Total** | — | **~108** (após esparsificação) |

Pares totais possíveis: C(30,2) = 435.

---

## Invariantes do Grafo

1. **Sem auto-loops:** `i ≠ j` para todas as arestas
2. **Todas as arestas são bidirecionais** — se (i,j) existe, (j,i) também existe
3. **Grafo esparso:** a esparsificação DCC-GARCH é obrigatória (não usar grafo completo)
4. **Tipos de aresta são disjuntos:** um par (i,j) pode ter arestas de múltiplos tipos
   simultaneamente (heterogêneo puro — não colapsar tipos)
5. **Features normalizadas:** todas as features de aresta devem estar em [-1, 1] ou [0, 1]
   antes de entrar no modelo

---

## Sparsificação para universos grandes

| Tamanho | Estratégia |
|---------|-----------|
| ≤ 50 ativos | Threshold simples: `|ρ| > 0.3` |
| 51-200 ativos | TMFG (Triangulated Maximally Filtered Graph) |
| > 200 ativos | TMFG + threshold adaptativo por percentil |

**Motivação:** Grafo completo com 500 ativos tem 124.750 arestas CORR → over-smoothing.
