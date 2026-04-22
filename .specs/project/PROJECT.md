# PROJECT - Visão Geral do Projeto DyFO

> **Auto-contido:** este documento pode ser passado sozinho para um LLM como contexto de projeto.

---

## O que é o DyFO

**DyFO** (Dynamic Financial Ontology) é o **Módulo 2** do sistema MATTS v4.0.

Sua responsabilidade é **única e bem-delimitada**: receber um fluxo de eventos financeiros
com timestamps e produzir, a cada passo de decisão `t`, um vetor `e_t ∈ R^100` que captura
as relações dinâmicas entre ativos do portfólio.

```text
Eventos financeiros (stream)
        ↓
   [ DyFO / Temporal Encoder (TGAT) ]
        ↓
  e_t ∈ R^100   →  State Constructor (M3)  →  Orquestrador MARL
```

O DyFO **não aprende política** — é um MODULE no FDAM (não-agente), sem loop de recompensa,
sem interação direta com o ambiente financeiro.

---

## Posição no MATTS

O MATTS (Multi-Agent Trading System) é composto por 5 módulos:

| Módulo | Nome              | Responsabilidade                            |
|--------|-------------------|---------------------------------------------|
| M1     | RDM               | Detector de regime (HMM-GAS-TVTP) — produz `pi_t` |
| **M2** | **DyFO**          | **Grafo financeiro dinâmico — produz `e_t`** |
| M3     | State Constructor | Concatena `[e_t | pi_t | H(pi_t) | alpha_t | x_t]` |
| M4     | Orquestrador      | Política MARL sobre o estado aumentado      |
| M5     | Risk Manager      | CVaR, restrições de portfólio             |

O acoplamento com outros módulos é **exclusivamente via contrato de I/O**:

- **Entrada do M1 → M2:** `pi_t ∈ R^K` (probabilidades de regime, K=3) como node feature
- **Saída do M2 → M3:** `e_t ∈ R^100` (embedding do grafo no passo t)

---

## Objetivo de Pesquisa

### Hipótese H4 (ablation B21)

> "O Sharpe e o MDD do TGAT (stateless) são superiores aos baselines (TGN, ROLAND) em ≥70% das janelas walk-forward."

### Tarefa de pré-treinamento (self-supervised)

Link prediction de correlação: dado `z_i(t)` e `z_j(t)`, prever se `|ρ_ij(t+1)| ≥ θ`.

Isso valida que o TGAT aprende representações de grafo úteis antes do treinamento MARL.

O DyFO consolidou a arquitetura **TGAT** como o padrão ouro por sua estabilidade:

1. **BL-21:** `TGAT` (Temporal Graph Attention Network), stateless temporal attention.
2. **BL-22:** Escala otimizada em **50 ações**.
3. **BL-23:** Métricas de risco: MDD e Portfolio Turnover.

---

## Contribuições originais do DyFO vs. TGN original (Rossi et al., 2020)

| Aspecto             | TGN (Rossi et al.) | DyFO (TGAT)                             |
|---------------------|--------------------|-----------------------------------------|
| Memória             | Recurrent (GRU)    | Stateless (Temporal Attn) ✅             |
| Correlações         | N/A                | DCC-GARCH (Engle 2002) ✅               |
| Tipo de grafo       | Homogêneo          | Heterogêneo (CORR/SECT/SUPL/FACT) ✅    |
| Escala              | ~10K nós           | 50 ativos (Config. ótima) ✅            |
| Métricas            | ML (R2/F1)         | Financeiras (Sharpe/MDD/Turnover) ✅    |
| Regime conditioning | Não                | `regime_prob` (M1 → M2) ✅              |
| Validação           | Single run         | 10k-bootstrap walk-forward ✅           |

---

## Stack tecnológico

| Componente                | Tecnologia                          |
|---------------------------|-------------------------------------|
| Framework ML              | PyTorch 2.x + PyTorch Geometric 2.x |
| Modelo base atual         | TGAT (Stateless)                    |
| Braço interpretável       | Temporal KG                         |
| Correlações               | `arch` v8.0.0 (DCC-GARCH)           |
| Dados de mercado          | `yfinance`                          |
| Dados macro               | FRED API                            |
| Fatores Fama-French       | Ken French Data Library             |
| Linguagem                 | Python 3.11+                        |

---

## Métricas de sucesso (v1.0)

| Métrica     | Mínimo publicável | TGAT (v1.0) |
|-------------|-------------------|-------------|
| Test R²     | > 0.60            | **0.824**   |
| Sharpe GMVP | > 1.5             | **2.615**   |
| MDD         | < 15%             | **12.4%**   |
| Turnover    | < 0.15            | **0.085**   |

---

## Referências primárias

| Paper                         | Relevância                             |
|-------------------------------|----------------------------------------|
| Rossi et al. (2020) — TGN     | Baseline de arquitetura                |
| You et al. (2022) — ROLAND    | Baseline de snapshot                   |
| Engle (2002) — DCC-GARCH      | Correlações dinâmicas                  |
| Korangi et al. (2024) — GAT FF5 | Inspiração FACT edges                 |
| TGAT (ICLR 2021)              | Arquitetura stateless de referência    |
| TeSa / CTRL                   | Inspiração para heterogeneidade        |
