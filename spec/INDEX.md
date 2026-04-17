# DyFO — Spec-Driven Development Index

> Base de conhecimento estruturada para otimizar o uso de LLMs no desenvolvimento do DyFO.
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
| Implementar relação-aware heterogeneous TGN | `04_tgn_spec.md` + `01_architecture.md` + `03_event_spec.md` |
| Planejar braço Temporal KG | `04_tgn_spec.md` + `02_graph_spec.md` |
| Corrigir bug no grafo | `02_graph_spec.md` |
| Ajustar treinamento/avaliação | `04_tgn_spec.md` + `scripts/run_bootstrap_eval_v5.py` |

---

## Documentos

- [00_project_overview.md](00_project_overview.md) — Contexto, posição no MATTS, objetivos e hipóteses
- [01_architecture.md](01_architecture.md) — Arquitetura interna, pipeline de 6 estágios, contratos de I/O
- [02_graph_spec.md](02_graph_spec.md) — Nós, arestas tipadas, features e invariantes do grafo financeiro
- [03_event_spec.md](03_event_spec.md) — Catálogo de eventos, vetores de features e regras de disparo
- [04_tgn_spec.md](04_tgn_spec.md) — Especificação do TGN: message function, GRU, GAT, readout
- [05_eval_protocol.md](05_eval_protocol.md) — Protocolo de avaliação experimental Rev 2: `--variants`, `--n_tickers`, `--ablation`

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

**Regra de implementação atual:**
- Universo padrão: **30 ações** (50 e 100 disponíveis via `--n_tickers`)
- Usar [run_bootstrap_eval_v5.py](../scripts/run_bootstrap_eval_v5.py) como baseline experimental congelado
- Runner ativo: `run_bootstrap_eval_temporal_kg_rev2.py` — suporta `--variants`, `--n_tickers`, `--ablation`
- **Não editar** `run_bootstrap_eval_v5.py`, `run_bootstrap_eval_ra_htgn.py` nem `run_bootstrap_eval_temporal_kg_rev1.py`

---

## Versão do modelo

| Métrica | Melhor resultado (v0.9) |
|---------|------------------------|
| Test R² | 0.789 |
| Spearman | 0.939 |
| MAE | 0.053 |
| F1 | 0.766 |
| Sharpe GMVP | 2.437 |
| H4 p-value | 0.0018 (TGN > ROLAND) ✅ |
| Arquitetura | TGN-attn 1L + DCC-GARCH + FACT edges |
| Ativos | 30 tickers S&P 500 (11 setores GICS) |

## Roadmap imediato

1. ✅ **BL-17:** `ra_htgn` implementado e validado.
2. ✅ **BL-18:** `temporal_kg` implementado e validado.
3. ✅ **BL-19 / Rev 2:** `--variants`, `--n_tickers`, `--ablation` disponíveis em `rev2`.
4. 🔴 **BL-20 (próximo):** Rodar ablação completa (`--ablation full`) para CORR/SECT/FACT e documentar contribuição relativa de cada tipo de aresta.
5. 🔴 **BL-21 (futuro):** Validar `--n_tickers 50` e `--n_tickers 100` — requer implementação de TMFG para universo > 50.
