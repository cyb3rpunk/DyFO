# 00 — Visão Geral do Projeto DyFO

> **Auto-contido:** este documento pode ser passado sozinho para um LLM como contexto de projeto.

---

## O que é o DyFO

**DyFO** (Dynamic Financial Ontology) é o **Módulo 2** do sistema MATTS v4.0.

Sua responsabilidade é **única e bem-delimitada**: receber um fluxo de eventos financeiros
com timestamps e produzir, a cada passo de decisão `t`, um vetor `e_t ∈ R^100` que captura
as relações dinâmicas entre ativos do portfólio.

```
Eventos financeiros (stream)
        ↓
   [ DyFO / Temporal Encoder ]
        ↓
  e_t ∈ R^100   →  State Constructor (M3)  →  Orquestrador MARL
```

O DyFO **não aprende política** — é um MODULE no FDAM (não-agente), sem loop de recompensa,
sem interação direta com o ambiente financeiro.

---

## Posição no MATTS

O MATTS (Multi-Agent Trading System) é composto por 5 módulos:

| Módulo | Nome | Responsabilidade |
|--------|------|------------------|
| M1 | RDM | Detector de regime (HMM-GAS-TVTP) — produz `pi_t` |
| **M2** | **DyFO** | **Grafo financeiro dinâmico — produz `e_t`** |
| M3 | State Constructor | Concatena `[e_t | pi_t | H(pi_t) | alpha_t | x_t]` |
| M4 | Orquestrador | Política MARL sobre o estado aumentado |
| M5 | Risk Manager | CVaR, restrições de portfólio |

O acoplamento com outros módulos é **exclusivamente via contrato de I/O**:
- **Entrada do M1 → M2:** `pi_t ∈ R^K` (probabilidades de regime, K=3) como node feature
- **Saída do M2 → M3:** `e_t ∈ R^100` (embedding do grafo no passo t)

---

## Objetivo de Pesquisa

### Hipótese H4 (ablation B16)
> "O Sharpe condicional do TGN ≥ ROLAND em ≥70% das janelas walk-forward."

Para validar H4, o DyFO precisa:
1. **BL-02:** Implementar baselines ROLAND e GAT-Static
2. **BL-08:** Validação com 500 bootstraps (IC 95%)

### Tarefa de pré-treinamento (self-supervised)
Link prediction de correlação: dado `z_i(t)` e `z_j(t)`, prever se `|ρ_ij(t+1)| ≥ θ`.

Isso valida que o TGN aprende representações de grafo úteis antes do treinamento MARL.

### Roadmap desta fase

O DyFO entra agora em uma fase com **dois braços explícitos**:

1. **BL-17:** `relation-aware heterogeneous TGN`, inspirado em TeSa/CTRL, como sucessor direto do TGN atual
2. **BL-18:** `Temporal KG`, como ablação interpretável baseada em fatos temporais

Ambos devem:
- manter o universo fixo em **30 ações**
- usar [run_bootstrap_eval_v5.py](../scripts/run_bootstrap_eval_v5.py:1) apenas como referência de protocolo
- evitar qualquer edição no runner `v5`

---

## Contribuições originais do DyFO vs. TGN original (Rossi et al., 2020)

| Aspecto | TGN Original | DyFO |
|---------|-------------|------|
| Correlações | N/A | DCC-GARCH (Engle 2002) ✅ |
| Tipo de grafo | Homogêneo | Heterogêneo (CORR/SECT/SUPL/FACT) ✅ |
| Consciência de relação | Limitada | `ra_htgn` com fusão inter-relação planejada |
| Interpretabilidade | Limitada | braço `Temporal KG` planejado |
| Escala | ~10K nós | 30 ativos financeiros, 435 pares ✅ |
| Regime conditioning | Não existe | `regime_prob` como node feature (M1 → M2) |
| Ablation | Sem comparação | TGN vs. ROLAND vs. GAT-Static (B16) |
| Validação | Single run | 500-bootstrap walk-forward |

---

## Stack tecnológico

| Componente | Tecnologia |
|-----------|-----------|
| Framework ML | PyTorch 2.x + PyTorch Geometric 2.x |
| Modelo base atual | TGN-attn (Rossi et al., 2020) |
| Próxima variante principal | Relation-aware heterogeneous TGN (TeSa/CTRL-inspired) |
| Braço interpretável | Temporal KG |
| Correlações | `arch` v8.0.0 (DCC-GARCH) |
| Dados de mercado | `yfinance` |
| Dados macro | FRED API |
| Fatores Fama-French | Ken French Data Library |
| Linguagem | Python 3.11+ |

---

## Métricas de sucesso (pré-treinamento)

| Métrica | Mínimo publicável | Melhor atual (v0.7) |
|---------|-------------------|---------------------|
| Test R² (regressão ρ) | > 0.60 | **0.806** |
| Spearman rank corr. | > 0.80 | **0.931** |
| MAE | < 0.10 | **0.049** |
| Classificação F1 | > 0.75 | 0.83 |

---

## Referências primárias

| Paper | Relevância |
|-------|-----------|
| Rossi et al. (2020) — TGN | Arquitetura base |
| You et al. (2022) — ROLAND | Baseline BL-02 |
| Pareja et al. (2020) — EvolveGCN | Base do ROLAND |
| Engle (2002) — DCC-GARCH | Correlações dinâmicas |
| Korangi et al. (2024) — GAT FF5 | Inspiração FACT edges |
| GAP-TGN (ICLR 2026) | Staleness mitigation |
| TeSa / CTRL | Inspiração para heterogeneidade relacional |
| RE-Net / TKGs temporais | Inspiração para ablação interpretável |
