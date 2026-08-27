# DyFO Financial Ontology Schema Specification

> **Documento:** Schema de Entidades e Relações Ontológicas  
> **Status:** ✅ Especificado (REQ-G4, REQ-G5, REQ-G6)  
> **Versão:** 1.0 (Compatível com DyFO_LITE, PORTA RQ6-SUF e ORION_LITE HFKG I9/I11)

---

## 1. Visão Geral

Este documento formaliza o esquema ontológico das entidades e relações exportadas pelo módulo **DyFO** através de `DyFOAdapter.export_structural_graph()`.

O objetivo é fornecer identificadores ontologicamente estáveis e mapeáveis para ontologias financeiras compartilhadas (OWL/RDF) utilizadas no ecossistema de pesquisa do Doutorado (`PORTA`, `ORION_LITE` HFKG e `DOC_MASTER`).

---

## 2. Padrão de Identificação de Entidades (`entity_id`)

Todas as entidades de ativos financeiros no DyFO utilizam a convenção de ticker no padrão **`.US`**, garantindo resolução determinística sem a necessidade de tabelas de tradução auxiliares.

- **Formato:** `<TICKER>.US` (em maiúsculas)
- **Exemplos:**
  - `AAPL.US` (Apple Inc.)
  - `MSFT.US` (Microsoft Corporation)
  - `JPM.US` (JPMorgan Chase & Co.)
  - `TXN.US` (Texas Instruments Inc.)
  - `SPY.US` (SPDR S&P 500 ETF Trust)

---

## 3. Esquema de Relações e Predicados OWL Candidatos

O DyFO modela o mercado financeiro através de quatro classes complementares de arestas relacionais:

| Tipo DyFO | Descrição Semântica | Predicado OWL Candidato | Atributos Mapeados |
|---|---|---|---|
| `CORR` | Co-movimento estatístico dinâmico (DCC-GARCH time-varying) | `fibo:correlatesWith` | `{"weight": rho, "p_val": float, "lag": 1}` |
| `SECT` | Pertencimento a setor/indústria idêntica | `fibo:belongsToSector` | `{"gics_sector": "45", "gics_sub_industry": "452020"}` |
| `SUPL` | Proximidade na cadeia de suprimentos (Supplier-Customer) | `fibo:suppliesTo` | `{"strength": float, "tier": int}` |
| `FACT` | Co-exposição a fatores de risco sistemáticos (Fama-French 5) | `fibo:sharesFactorExposureWith` | `{"model": "FF5", "distance_l2": float}` |

---

## 4. Estrutura do Snapshot Exportado (`StructuralGraphSnapshot`)

```python
@dataclass(frozen=True)
class RelationEdge:
    source_entity_id: str      # ex.: "AAPL.US"
    target_entity_id: str      # ex.: "MSFT.US"
    weight: float              # Peso bruto da relação (ex.: rho para CORR)
    attributes: dict           # Metadados e atributos semânticos

@dataclass(frozen=True)
class StructuralGraphSnapshot:
    as_of_date: date
    causal_cutoff_date: date   # == as_of_date (invariante causal de auditoria)
    entity_ids: list[str]      # (N,) Ordenação canônica estável
    node_embeddings: ndarray   # (N, 100) Embedding temporal relation-aware
    edges_by_relation: dict    # {"CORR": [...], "SECT": [...], "SUPL": [...], "FACT": [...]}
    relation_attention_weights: ndarray | None # (N, 4) Atenção semântica opcional
```

---

## 5. Invariantes Causais e de Integridade

1. **Ausência Explícita (REQ-G3):** Se uma relação não possui dados materializados naquela data (por exemplo, `SUPL`), `edges_by_relation["SUPL"]` retorna como uma lista vazia `[]`, nunca omitindo a chave ou retornando pesos `0.0` silenciosos.
2. **Causalidade Estrita (REQ-G2 / REQ-D1..D3):** Nenhum embedding ou aresta presente no snapshot para a data $t$ pode ser gerado a partir de observações datadas com timestamp $> t$.
3. **Determinismo (REQ-G1):** Chamadas repetidas a `export_structural_graph(as_of_date)` para a mesma data devem retornar dados byte-idênticos.
