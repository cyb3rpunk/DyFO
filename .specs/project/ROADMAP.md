# ROADMAP

> Funcionalidades e milestones do projeto.> Base de conhecimento estruturada para otimizar o uso de LLMs no desenvolvimento do DyFO.
> Cada documento é auto-contido e pode ser passado diretamente como contexto para um LLM.
>
> **Princípio SDD:** Especificação precede implementação. Toda mudança começa aqui.

---

## Como usar este índice

| Situação | Documentos a consultar |
|----------|------------------------|
| Entender o projeto do zero | `00_project_overview.md` → `01_architecture.md` |
| Implementar nova feature | `01_architecture.md` + spec do módulo relevante |
| Adicionar tipo de evento | `03_event_spec.md` |
| Implementar relação-aware TGAT | `04_temporal_encoder_spec.md` + `01_architecture.md` |
| Planejar braço Temporal KG | `04_temporal_encoder_spec.md` + `02_graph_spec.md` |
| Corrigir bug no grafo | `02_graph_spec.md` |
| Ajustar treinamento/avaliação | `05_eval_protocol.md` + `scripts/run_bootstrap_eval_temporal_kg_rev2.py` |
| Calcular métricas financeiras | `scripts/compute_mdd_turnover_full.py` |

---

## Documentos

- [00_project_overview.md](00_project_overview.md) — Visão geral, pivot TGAT, objetivos e hipóteses
- [01_architecture.md](01_architecture.md) — Arquitetura interna, pipeline de 6 estágios, contratos de I/O
- [02_graph_spec.md](02_graph_spec.md) — Nós, arestas tipadas, features e invariantes do grafo financeiro
- [03_event_spec.md](03_event_spec.md) — Catálogo de eventos, vetores de features e regras de disparo
- [04_temporal_encoder_spec.md](04_temporal_encoder_spec.md) — Spec do Encoder: TGAT (Stateless) vs TGN (RNN-memory)
- [05_eval_protocol.md](05_eval_protocol.md) — Protocolo de avaliação Rev 2: Bootstrap, MDD, Turnover e Testes Estatísticos

---

## Status do Projeto (atualizar a cada sprint)

| BL | Descrição | Status |
|----|-----------|--------|
| BL-01 | 30 ativos S&P 500 | ✅ Implementado |
| BL-02 | Baselines ROLAND + GAT-Static | ✅ Implementado |
| BL-03 | DCC-GARCH correlações | ✅ Implementado |
| BL-04 | Viés precision/recall | ✅ Resolvido |
| BL-05 | Integridade de dados + retry | ✅ Implementado |
| BL-06 | Early stopping patience=5 | ✅ Implementado |
| BL-07 | Threshold tuning val set | ✅ Implementado |
| BL-08 | Block Bootstrap 10k iterações | ✅ Implementado (H4 p=0.0018) |
| BL-09 | Integração RDM (regime_prob) | 🔴 Pendente (depende M1) |
| BL-10 | Supply chain edges (SUPL) | 🔴 Pendente |
| BL-11 | Factor edges Fama-French (FACT) | ✅ Implementado |
| BL-12 | Staleness proxy | 🟡 Documentado, não implementado |
| BL-16 | Report visual (gráficos + grafo) | 🔴 Pendente |
| BL-17 | Relation-aware heterogeneous TGN | ✅ Implementado (`ra_htgn`) |
| BL-18 | Temporal KG ablation interpretável | ✅ Implementado (`temporal_kg`) |
| BL-19 | `--variants` filter + `--n_tickers` + `--ablation` | ✅ Rev 2 (`run_bootstrap_eval_temporal_kg_rev2.py`) |
| BL-20 | Ticker registry centralizado (30/50/100) | ✅ `dyfo/core/ticker_registry.py` |
| BL-21 | Pivot para TGAT (Stateless Temporal Attention) | ✅ Implementado (Tier 1) |
| BL-22 | Análise de escala 50/100 ativos | ✅ Concluído (50 ativos como ideal) |
| BL-23 | Métricas MDD + Turnover | ✅ `compute_mdd_turnover_full.py` |
| BL-24 | Testes Wilcoxon + Diebold-Mariano | ✅ Integrado no runner `rev2` |
| BL-27 | TGAT edge_dim fix (relation-aware GATConv) | ✅ `tgat_encoder.py` — GATConv com `edge_dim=et_dim` |
| BL-28 | Multi-seed ablation (5 seeds) | ✅ `run_bootstrap_eval_temporal_kg_rev3.py` — `--seeds` |
| BL-29 | Hyperparams separados para temporal_kg/ra_htgn | ✅ `patience=15`, `cosine=True` |

**Regra de implementação atual:**
- **Modelo Primário:** TGAT (Stateless). TGN rebaixado para baseline (instabilidade recurrent).
- Universo padrão: **50 ações** (30 e 100 disponíveis via `--n_tickers`)
- Runner ativo: `run_bootstrap_eval_temporal_kg_rev3.py` — suporta `--variants`, `--n_tickers`, `--ablation`, `--seeds`
- **Não editar** `run_bootstrap_eval_v5.py` nem runners `rev1`/`rev2`.

---

## Versão do modelo

| Métrica | Melhor resultado (v1.0 - TGAT) |
|---------|------------------------|
| Test R² | 0.824 |
| Sharpe GMVP | 2.615 |
| MDD | 12.4% |
| Turnover | 0.085 |
| H4 p-value | < 0.0001 (TGAT > Baselines) ✅ |
| Arquitetura | TGAT (Stateless) + DCC-GARCH + FACT edges |
| Ativos | 50 tickers S&P 500 (Configuração ótima) |

## Roadmap imediato

1. ✅ **BL-21:** Pivot TGAT concluído.
2. ✅ **BL-22:** Análise de escala 50/100 validada.
3. ✅ **BL-23:** Script de MDD/Turnover unificado.
4. 🔴 **BL-25 (próximo):** Finalizar escrita do paper com os novos dados de TGAT-50.
5. 🔴 **BL-26:** Implementar TMFG real para universo > 100 ativos.
6. ✅ **BL-27:** TGAT edge_dim fix — GATConv relation-aware. **Requer re-ablação.**
7. ✅ **BL-28:** Multi-seed ablation support (`--seeds 42 123 456 789 2024`).
8. ✅ **BL-29:** Hyperparams separados para temporal_kg/ra_htgn (patience=15, cosine=True).
9. 🟡 **BL-30:** Re-ablação TGAT v2 (edge_dim) — validar que `all_edges ≥ CORR+FACT`.
