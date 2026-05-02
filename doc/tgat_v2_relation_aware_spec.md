# TGAT v2: Relation-Aware Temporal Graph Attention

> **Documento de Especificação Técnica**  
> **Status:** ✅ Implementado (QF-027)  
> **Data:** 2026-04-22  
> **Versão:** 2.0 (Evolution of TGAT v1)

---

## 1. Visão Geral

Esta especificação detalha a evolução do modelo **TGAT (Temporal Graph Attention Network)** para uma arquitetura **Relation-Aware**. A mudança foi motivada pela identificação de uma limitação estrutural no readout de grafo ("GAT structural readout"), onde a atenção era calculada de forma homogênea, ignorando a semântica dos diferentes tipos de relações (CORR, SECT, FACT).

## 2. Diagnóstico: GATConv Homogêneo

Na versão 1.0 do TGAT, a camada de structural readout utilizava uma implementação padrão de `GATConv` que considerava apenas a topologia do grafo e as features dos nós:

### Comportamento v1.0 (Limitação)
O modelo tratava uma aresta de **Correlação (DCC-GARCH)**, uma de **Setor (Estática)** e uma de **Fator (Fama-French)** como vizinhos idênticos no cálculo de atenção.
- **Consequência:** Diluição de atenção. Vizinhos de alta relevância temporal (CORR) tinham seu sinal diluído por dezenas de vizinhos estáticos (SECT) redundantes.
- **Evidência:** No teste de ablação, a variante `all_edges` (todas as arestas) apresentava performance **inferior** à variante `CORR+FACT`, demonstrando que a adição de arestas redundantes prejudicava o modelo.

## 3. Arquitetura Relation-Aware (v2.0)

A versão 2.0 introduz a **Atenção Condicional à Relação**. A camada GAT agora recebe o embedding do tipo de aresta (`edge_type_emb`) para modular os pesos de atenção.

### 3.1. Formulação Matemática

A função de atenção original no GAT era:
$$ \alpha_{ij} = \text{softmax}_j \left( \text{LeakyReLU}\left( \mathbf{a}^T [ \mathbf{W}\mathbf{h}_i \, \Vert \, \mathbf{W}\mathbf{h}_j ] \right) \right) $$

Na **TGAT v2.0**, a função evolui para integrar o vetor de atributos da aresta ($e_{ij}$):
$$ \alpha_{ij} = \text{softmax}_j \left( \text{LeakyReLU}\left( \mathbf{a}^T [ \mathbf{W}\mathbf{h}_i \, \Vert \, \mathbf{W}\mathbf{h}_j \, \Vert \, \mathbf{W}_e \mathbf{e}_{ij} ] \right) \right) $$

Onde:
- $\mathbf{h}_i, \mathbf{h}_j$ são os embeddings dos nós.
- $\mathbf{e}_{ij}$ é o embedding de dimensão 16 correspondente ao `edge_type` (CORR, SECT, FACT, etc.).
- $\mathbf{W}_e$ é a matriz de projeção aprendida para as relações.

### 3.2. Estrutura de Camadas

```mermaid
graph TD
    A[Histórico de Eventos] --> B[Temporal Attention Module]
    B --> C{Readout Estrutural}
    C -- "Nós (h_temporal)" --> D[GATConv V2]
    C -- "Arestas (Typing)" --> D
    D -- "Relation-Aware Attention" --> E[Final Node Embedding]
    
    subgraph "GATConv V2 Detail"
    D1[Concatenate: Src || Tgt || Edge_Attr]
    D1 --> D2[Apply Attention Weights α]
    D2 --> D3[Aggregate Neighbors]
    end
```

## 4. Detalhes de Implementação

As modificações foram realizadas no arquivo `dyfo/core/tgat_encoder.py`.

### 4.1. Inicialização do GAT
Foi habilitado o parâmetro `edge_dim` para suportar features de arestas.

```python
# Inicialização (v2.0)
self.gat = GATConv(
    in_channels=self._d_model + self._node_feat_dim,
    out_channels=self._d_model // self._n_heads,
    heads=self._n_heads,
    dropout=self._dropout_p,
    concat=True,
    edge_dim=self._et_dim,  # <-- Diferencial v2.0
)
```

### 4.2. Forward Pass
Os tipos de arestas são convertidos em embeddings antes de serem passados para a camada GAT.

```python
# Step 2: GAT structural readout (v2.0)
edge_type_emb_gat = self.edge_type_emb(edge_type_ids.to(device))
gat_out = self.gat(
    gat_in, 
    edge_index.to(device), 
    edge_attr=edge_type_emb_gat  # <-- Fluxo de informação relacional
)
```

## 5. Impacto e Compatibilidade

| Característica | Detalhe |
|:---|:---|
| **Poder de Predição** | Espera-se que `all_edges` ≥ `CORR+FACT` devido à capacidade de ignorar redundância. |
| **Interpretabilidade** | Pesos de atenção $\alpha$ agora podem ser analisados por tipo de relação. |
| **Performance** | Overhead computacional desprezível (~1-2%). |
| **Compatibilidade** | **Breaking Change.** Checkpoints treinados na v1.0 não carregam na v2.0. |

## 6. Próximos Passos

1. **Re-Ablação:** Executar o script de ablação com o modelo v2.0 para confirmar a solução do problema de diluição.
2. **Interpretabilidade:** Implementar logging de `self.gat.explain_node` ou extração de pesos por tipo de aresta para validação visual.
