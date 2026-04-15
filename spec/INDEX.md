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

**Nota:** nesta branch, o núcleo do SDD ativo está concentrado nos documentos `00` a `04`.
Os demais itens do índice antigo ainda não existem como arquivos separados e não devem ser
assumidos como fonte de verdade.

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
| BL-17 | Relation-aware heterogeneous TGN | 🔴 Planejado neste SDD |
| BL-18 | Temporal KG ablation interpretável | 🔴 Planejado após BL-17 |

**Regra de implementação atual:**
- Manter o universo fixo de **30 ações**
- Usar [run_bootstrap_eval_v5.py](../scripts/run_bootstrap_eval_v5.py) como baseline experimental de referência
- **Não editar** o runner atual para BL-17 nem para BL-18; cada variante terá script/entrypoint próprio

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

1. **BL-17:** implementar uma versão `relation-aware heterogeneous TGN` inspirada em TeSa/CTRL.
   Ela preserva o fluxo de eventos contínuos, mas separa agregação intra-relação e inter-relação.
2. **BL-18:** implementar um braço `Temporal KG` para ablação interpretável, sem substituir o pipeline BL-17.
3. Validar ambos usando a mesma filosofia walk-forward de [run_bootstrap_eval_v5.py](../scripts/run_bootstrap_eval_v5.py:1),
   porém em scripts novos para evitar regressão no protocolo já validado.
