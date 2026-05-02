# 04 — Temporal Encoder Spec (TGAT)

> Fonte de verdade para a arquitetura de produção do DyFO:
> 1. **TGAT v2** (Relation-Aware) - Stateless [Atual]
>    - Baseado em `tgat_v2_relation_aware_spec.md`
> 2. **TGN** (Recurrent Memory) - Baseline legado (estabilidade < TGAT)

---

## Objetivo da Fase Atual

Consolidar o **TGAT** como o encoder primário para o sistema de trading. A transição para uma arquitetura stateless foi motivada pela instabilidade da memória recorrente (GRU) em janelas longas de predição financeira.

### Vantagens do TGAT (Stateless)
- **Determinismo:** Sem estados ocultos persistentes que degradam com o tempo.
- **Eficiência:** Paralelização total sobre o histórico de janelas.
- **Robustez:** Melhor desempenho em janelas walk-forward de 500+ dias.

---

## Escopo Consolidado

### Universo de Ativos
- **50 ações** (S&P 500) definido como o tradeoff ideal entre complexidade e poder preditivo.
- Suporte experimental para 30 e 100 ativos.

O encoder baseia-se em Atenção Temporal em vez de memória recorrente.  
> **Versão Atual:** v2.0 (Relation-Aware — veja [tgat_v2_relation_aware_spec.md](tgat_v2_relation_aware_spec.md))

```text
Eventos financeiros
    ↓
Message Function (φ)
    ↓
Temporal Multi-Head Attention (Stateless)
    ↓
Relation-Aware GAT (Structural Readout)
    ↓
Global Readout
```

---

## Detalhes Técnicos (TGAT)

### 1. Message Function
- Codifica atributos de eventos (DCC-GARCH, FF5, Sectors).
- Incorpora `Time2Vec` para codificação de tempo contínuo.

### 2. Temporal Attention Layer
- Busca vizinhos históricos relevantes no stream de eventos.
- Calcula scores de atenção baseados em proximidade temporal e relevância do evento.
- **Dimensão:** 100-dim embeddings.

### 3. Graph GAT Layer (Relation-Aware)
- Propaga informações entre ativos correlacionados no grafo heterogêneo.
- **Diferenciação:** Utiliza `edge_attr` para ponderar `CORR`, `SECT`, `FACT` semanticamente.
- **Mecanismo:** GATConv v2 com `edge_dim=16`.

---

## Status das Variantes

| Variante | Status | Papel no Projeto |
|----------|--------|------------------|
| `tgat` | ✅ Ativo | **Tier 1** - Modelo v2.0 (Relation-Aware) |
| `tgn` | ⚠️ Estável | Baseline (com memória GRU) |
| `temporal_kg` | ✅ Estável | Braço interpretável para ablação |
| `ra_htgn` | 🟡 Pendente | Pesquisa futura em fusão semântica |

---

## Métricas e Avaliação

O TGAT deve ser avaliado utilizando o script `run_bootstrap_eval_temporal_kg_rev2.py` com as seguintes métricas financeiras:

1. **Sharpe Ratio (GMVP):** Eficiência ajustada ao risco.
2. **Max Drawdown (MDD):** Risco de cauda e estabilidade.
3. **Turnover:** Custo transacional estimado.
4. **Win Rate:** Percentual de janelas walk-forward onde supera o baseline.

---

## Regras de Implementação (SDD)

1. **Não reintroduzir GRU/RNN** no fluxo primário de `tgat_encoder.py`.
2. **Persistência:** O modelo é stateless entre batches de inferência.
3. **Compatibilidade:** Manter o contrato de I/O de 100 dimensões para o M3 (State Constructor).
4. **Documentação:** Toda mudança no encoder deve ser refletida primeiro nesta especificação.
