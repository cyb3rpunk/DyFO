# DyFO — Walkthrough & Gap Analysis

> **Data:** Abril 2026  
> **Escopo:** Análise do main local vs TGN (Rossi et al., 2020) + Proposta MATTS v4.0  
> **Objetivo:** Mapear o que foi implementado, o que diverge do paper de referência, o que a proposta exige, e o que falta para DyFO se tornar um entregável (paper avulso + módulo MATTS).

---

## Parte 1 — DyFO (main local) vs TGN Original

### 1.1 O que o TGN define e o DyFO implementa corretamente

| Componente TGN | Definição no Paper | Implementação DyFO | Arquivo |
|---|---|---|---|
| **Memory** | Vetor `s_i(t)` por nó, GRU, inicializado em zero | GRU dim=172, inicializado em zero, herdado entre splits | `tgn_encoder.py` |
| **Message Function** | Identity: `msg = [s_i ‖ s_j ‖ φ(Δt) ‖ e_ij]` | `[s_i ‖ s_j ‖ Time2Vec(Δt) ‖ f_e ‖ edge_type_emb ‖ event_type_emb]` | `tgn_encoder.py` |
| **Message Aggregator** | Mean ou Last | Ambos implementados (configurável) | `tgn_encoder.py` |
| **Memory Updater** | GRU ou LSTM | GRU | `tgn_encoder.py` |
| **Embedding Module** | Temporal Graph Attention (1L, 10 vizinhos) | GAT 1 camada, 2 heads, 10 vizinhos + Time2Vec | `tgn_encoder.py` |
| **Time2Vec** | `φ(t) = [linear, sin(periodic...)]` | Implementado com 1 linear + (dim-1) periódicos | `tgn_encoder.py` |
| **Link Predictor** | MLP sobre `[z_i ‖ z_j]` | MLP 200→64→32→1 (classificação) e regressão contínua | `link_prediction.py` |
| **Protocolo Walk-forward** | Split cronológico 70/15/15 | Split 60/20/20 com memória herdada (sem reset) | `train_link_prediction.py` |

A arquitetura central do TGN está fielmente implementada. O pipeline de 4 etapas (message → aggregate → update memory → GAT embedding) é respeitado.

---

### 1.2 Diferenças e extensões do DyFO em relação ao TGN

#### 1.2.1 Extensões legítimas (contribuições DyFO)

**a) Taxonomia de 7 eventos financeiros tipados**

O TGN define apenas 2 tipos genéricos de evento (node-wise e interaction). O DyFO estende para 7 tipos com semântica financeira:

| Evento DyFO | Tipo TGN equiv. | Contribuição |
|---|---|---|
| `PRICE_UPDATE` | node-wise | Retorno, volatilidade, volume por ativo |
| `EARNINGS_REPORT` | node-wise | Surprise EPS, revenue beat, guidance delta |
| `FED_DECISION` | interaction (broadcast) | ΔRate, dot-plot, sentiment — afeta todos os nós |
| `CREDIT_DOWNGRADE` | node-wise | Notch delta, outlook, contagion |
| `CORP_ACTION` | node-wise | M&A, split, dividendo |
| `CORRELATION_UPDATE` | interaction (bilateral) | ρ novo, Δρ, significância |
| `MACRO_RELEASE` | node-wise (regime-dep.) | Surprise, revisão, impacto vol — via z-score FRED |

**b) Correlações DCC-GARCH como arestas**

O TGN não tem correlações financeiras. O DyFO usa DCC-GARCH(1,1) (Engle 2002) para estimar correlações time-varying: `Q_t = (1-a-b)Q̄ + a(ε_{t-1}ε_{t-1}') + bQ_{t-1}`. Isso substitui edges estáticos por edges com features econometricamente fundamentadas — contribuição direta para a literatura de TGN aplicado a finanças.

**c) 4 tipos de aresta heterogênea**

O TGN usa grafos homogêneos. O DyFO define:
- `CORR` — correlação dinâmica (DCC-GARCH)
- `SECT` — co-setor GICS (estático)
- `SUPL` — cadeia de fornecimento (stub)
- `FACT` — proximidade de loading Fama-French (implementado, não ativado)

**d) Global readout (e_t)**

O paper TGN menciona "global memory como trabalho futuro". O DyFO implementa 3 estratégias de readout para produzir o embedding do grafo completo `e_t ∈ ℝ^100`: mean, weighted (market-cap), e attention. Esse `e_t` é a saída que alimenta o State Constructor do MATTS — não existe no TGN original.

**e) Regressão contínua de ρ**

O TGN é avaliado apenas em classificação binária de links. O DyFO estende para regressão de ρ contínuo com `CorrelationRegressor` (MLP + tanh), avaliado por R² e Spearman. Melhor adequação ao problema financeiro onde a magnitude da correlação importa.

**f) Node features financeiras de 20 dimensões**

O TGN usa features genéricas. O DyFO define 20 dims: log return 21d, volatilidade histórica, beta, setor one-hot (11), log market cap, drawdown, regime probability (3, zero-filled), volume normalizado.

---

#### 1.2.2 Divergências problemáticas (gaps de implementação)

**Gap 1 — Raw Message Store não verificado**

O paper TGN (Seção 3.2, Figura 2) descreve um mecanismo crucial: o **Raw Message Store** que armazena mensagens de batches anteriores para atualizar a memória antes de prever o batch atual — evitando information leakage e garantindo que os módulos de memória recebam gradiente.

**Status no DyFO:** O código implementa o pipeline sequencial (message → memory → embedding) mas não há evidência explícita do Raw Message Store com delay de 1 batch. Se a memória for atualizada com as interações do próprio batch antes de prever, o treinamento tem leakage e os módulos de memória não recebem gradiente adequado.

**Impacto:** Alto. Sem o Raw Message Store correto, o modelo pode convergir por razões erradas e as métricas no treino são infladas. **Necessita verificação e correção antes da submissão.**

**Gap 2 — Edges heterogêneas tratadas homogeneamente**

O DyFO define 4 tipos de aresta, mas o `tgn_encoder.py` usa um único `edge_type_emb` embedding — todos os tipos passam pelo mesmo GAT. A literatura 2024-2025 (e.g., HGT, R-GCN) demonstra que agregação por tipo melhora performance em grafos heterogêneos.

**Impacto:** Médio para o paper. A contribuição de ter edges heterogêneas perde força se não houver aggregation por tipo. Pode ser explorado como ablation (B16 estendido).

**Gap 3 — Sem baselines implementadas (BL-02)**

O TGN valida contra 10 baselines (CTDNE, Jodie, DyRep, TGAT, GAE, VGAE, DeepWalk, Node2Vec, GAT, GraphSAGE). O DyFO tem apenas o próprio modelo — sem ROLAND, sem GAT-static, sem DyRep.

**Impacto:** Crítico. Sem ablation B16, o paper não tem comparação e não é publicável.

**Gap 4 — Validação com única run (sem intervalo de confiança)**

O TGN reporta mean ± std sobre 10 runs. O DyFO tem resultados de run única (v0.6: R²=0.628, Spearman=0.877). Sem bootstrap ou múltiplas runs, não é possível afirmar significância estatística.

**Impacto:** Crítico para publicação. BL-08 (500-bootstrap) está pendente.

**Gap 5 — Regime probabilities zero-filled (BL-09)**

Os 3 dims de `regime_prob` nos node features estão zerados. Sem o módulo RDM (HMM-GAS-TVTP) integrado, essa feature não existe — o que é aceitável para um paper avulso de DyFO, mas obrigatório para o paper MATTS.

**Gap 6 — Factor edges não ativadas (BL-11)**

O código `compute_factor_edges()` existe em `edge_features.py` mas nunca é chamado com dados reais (FF5 não carregado no pipeline de treino). As arestas FACT estão silenciosamente ausentes dos experimentos.

**Gap 7 — Inductive setting não avaliado**

O TGN reporta resultados em settings transductive E inductive (nós não vistos no treino). O DyFO avalia apenas transductive (universo fixo de ativos). Para um paper avulso, o setting inductive não é mandatório (finanças têm universo fixo), mas deve ser justificado na seção de limitations.

---

### 1.3 Resumo do alinhamento DyFO vs TGN

| Dimensão | Alinhamento | Nota |
|---|---|---|
| Arquitetura central (Memory+Message+GAT) | ✅ Fiel | Pipeline de 4 etapas correto |
| Raw Message Store (gradient flow) | ⚠️ Não verificado | Risco de leakage — verificar urgente |
| Tipos de evento (2 vs 7) | ✅ Extensão legítima | Contribuição original |
| Edges heterogêneas | ⚠️ Parcial | Definidas, não agregadas por tipo |
| Correlações DCC-GARCH | ✅ Contribuição | Além do escopo do TGN |
| Baselines | ❌ Ausentes | BL-02 pendente — bloqueador |
| Validação estatística | ❌ Single run | BL-08 pendente — bloqueador |
| Walk-forward protocol | ✅ Correto | 60/20/20 com memória herdada |
| Global readout (e_t) | ✅ Extensão | Não existe no TGN — contribuição |
| Setting inductive | ❌ Não avaliado | Justificar em limitations |

---

## Parte 2 — DyFO na Proposta MATTS v4.0

### 2.1 O que o MATTS v4.0 exige do DyFO

O DyFO é o **Módulo 2 (M2)** do MATTS. Segundo a proposta, suas responsabilidades são:

**Entrada:** Grafo financeiro `G = (V, E)` com eventos em tempo contínuo  
**Saída:** Embedding `e_t ∈ ℝ^d` do grafo completo (atualizado a cada batch de eventos)

Requisitos funcionais para integração com MATTS:

| Requisito | Status | Descrição |
|---|---|---|
| Produzir `e_t` como vetor contínuo | ✅ Implementado | Readout strategies → `e_t ∈ ℝ^100` |
| Processar eventos assíncronos (não snapshots) | ✅ Implementado | EventStreamBuilder + TGN CTDG |
| Interface com State Constructor (M3) | ❌ Não integrado | `e_t` existe mas não há pipeline M2→M3 |
| Receber `π_t` do RDM como node feature | ❌ Zero-filled | BL-09 pendente, depende de M1 |
| Ablation B16 (TGN vs ROLAND vs GAT-static) | ❌ Pendente | BL-02, questão de pesquisa Q5 da tese |
| Ser classificado como MODULE no FDAM | ✅ Satisfeito | Entrada/saída determinística, sem política RL |
| Validação walk-forward com 500-bootstrap | ❌ Pendente | BL-08, exigido por Q8 |

### 2.2 Questão de Pesquisa Q5 (específica do DyFO)

A proposta MATTS v4.0 define:

> **Q5:** Grafos temporais de eventos contínuos (TGN) superam representações em snapshots discretos (ROLAND, GAT estático) para capturar correlações dinâmicas entre ativos em tempo real?

Esta é a pergunta científica central do **paper avulso de DyFO**. O Ablation B16 é a evidência direta. Sem BL-02 (baselines), Q5 não pode ser respondida.

### 2.3 Proposição teórica relacionada (P5 no Módulo 4)

A proposta menciona que o DyFO-TGN deve satisfazer:  
- Entrada bem definida: grafo `G_t`  
- Saída determinística: embedding `e_t`  
- Sem política aprendida em loop de recompensa  
- Sem interação direta com o ambiente  
- Stateless entre episódios (memória resetada entre runs, não entre splits do mesmo run)

O DyFO atual satisfaz todos os 5 critérios do FDAM para MODULE.

### 2.4 O que o MATTS requer além do paper avulso

| Requisito MATTS | Necessário para paper avulso? | Necessário para MATTS? |
|---|---|---|
| `π_t` do RDM como node feature (BL-09) | Não (pode ser feature zero) | Sim — obrigatório |
| Interface M2→M3 (State Constructor) | Não | Sim — obrigatório |
| Supply chain edges (BL-10) | Não | Opcional (P2) |
| Downstream Sharpe/CVaR evaluation (BL-15) | Não | Sim — avaliação final |
| Portfolio execution (M5) | Não | Sim — fora do escopo DyFO |

---

## Parte 3 — Gap Analysis: O que falta para DyFO ser um entregável

### 3.1 Entregável A: Paper Avulso (venues: ICML/NeurIPS/ICLR ou Q1 finanças)

**Claim principal do paper:** TGN com eventos financeiros contínuos e correlações DCC-GARCH supera ROLAND (snapshots discretos) e GAT-estático para previsão de estrutura de correlação em portfólios.

#### Bloqueadores (impedem submissão):

| Item | Backlog | Esforço estimado | Impacto |
|---|---|---|---|
| **Implementar ROLAND baseline** (EvolveGCN-H sobre snapshots mensais) | BL-02 | Alto | Ablation B16 sem comparação → paper rejeitado |
| **Implementar GAT-static baseline** (GAT sobre correlação média, sem memória) | BL-02 | Médio | Idem |
| **500-bootstrap validation** com IC 95% | BL-08 | Médio (depende de BL-02) | Robustez estatística obrigatória |
| **Verificar/corrigir Raw Message Store** | — | Alto | Risco de resultados inflados por leakage |
| **Múltiplas runs** (mínimo 5) com mean ± std | — | Médio | Sem intervalos de confiança não é publicável |

#### Importantes (fortalecem o paper mas não bloqueiam):

| Item | Backlog | Esforço | Impacto |
|---|---|---|---|
| **Factor edges (FACT) ativadas** com FF5 real | BL-11 | Baixo (código existe) | Enriquece o grafo heterogêneo |
| **Heterogeneous message aggregation** por tipo de aresta | — | Alto | Diferencial arquitetural robusto |
| **Staleness proxy** (PRICE_UPDATE sintético após 5 dias) | BL-12 | Baixo | Mitiga degradação em ativos inativos |
| Escalar para 50 ativos (1225 pares) | BL-01 ext | Médio | Resultado mais discriminante |
| Justificar ausência de setting inductive | — | Baixo | Limitations section |

#### Seções que faltam ser escritas:

- Abstract, Introduction, Related Work
- Methods (formalização DyFO-TGN em linguagem de paper, incluindo event types, DCC-GARCH, readout)
- Experiments (Ablation B16 completo, tabelas de resultado com bootstrap)
- Conclusion

---

### 3.2 Entregável B: Módulo DyFO integrado no MATTS

Além de tudo do Entregável A:

| Item | Backlog | Dependência externa |
|---|---|---|
| **RDM integration** — preencher `regime_prob` com `π_t` do M1 | BL-09 | Módulo RDM (M1) implementado |
| **Interface M2→M3** — pipeline `e_t` → State Constructor | BL-15 | Módulo State Constructor (M3) |
| **Supply chain edges** (opcional) | BL-10 | Fonte de dados externa (FactSet) |
| **Downstream evaluation** — Sharpe / CVaR do portfólio final | BL-15 | Módulos M3-M5 completos |
| **Multi-window walk-forward** sobre múltiplos datasets | BL-13 | Dados MSCI, commodities, crypto |

---

### 3.3 Sequência de execução recomendada

```
AGORA (bloqueadores):
1. Verificar Raw Message Store em tgn_encoder.py / train_link_prediction.py
   → Se ausente: implementar delayed memory update (Figura 2 do paper)
   
2. BL-02: Implementar ROLAND baseline
   → EvolveGCN-H: snapshots mensais de correlação → GCN sobre sequência de grafos
   → Manter mesmo universo de ativos (30), mesmo período (2020-2024)
   
3. BL-02: Implementar GAT-static baseline
   → Correlação média do período de treino → GAT estático → link predictor
   → Mesma arquitetura de decoder, sem memória
   
4. BL-11: Ativar FACT edges no pipeline de treino (FF5 já tem cache)

CURTO PRAZO (solidificam):
5. 5+ runs com seeds diferentes → mean ± std para todas as métricas
6. BL-08: 500-bootstrap sobre Ablation B16 completo
7. Seção de limitações: inductive setting, universo fixo, dados US-centric

MÉDIO PRAZO (paper MATTS):
8. BL-09: Integração com RDM quando M1 estiver pronto
9. Interface e_t → State Constructor (M3)
10. BL-15: Avaliação downstream (Sharpe, CVaR) após M3-M5
```

---

## Parte 4 — Estado Atual dos Resultados (v0.6)

| Métrica | Valor | Contexto |
|---|---|---|
| Test R² | 0.628 | Regressão contínua de ρ, 30 ativos |
| Test Spearman | 0.877 | Rank correlation das correlações previstas |
| Precision (@0.5 threshold) | 0.72 | Classificação derivada |
| Recall | 0.99 | — |
| F1 | 0.83 | — |
| #Eventos | 351K | PRICE + EARNINGS + CORP + MACRO + CORR |
| #Pares | 435 | C(30,2) |
| Epochs | best/15 | Com early stopping patience=5 |

Esses números são promissores mas são **single-run, sem baseline comparativa**. Não são publicáveis no estado atual.

---

## Parte 5 — Conclusão: DyFO hoje vs DyFO publicável

| Dimensão | Estado Atual | Estado para Publicação |
|---|---|---|
| Arquitetura TGN | ✅ Completa | Verificar Raw Message Store |
| Eventos financeiros | ✅ 7 tipos implementados | — |
| Correlações DCC-GARCH | ✅ Implementado | — |
| Factor edges | ⚠️ Código existe, não ativado | Ativar BL-11 |
| Heterogeneous aggregation | ❌ Homogênea | Opcional (paper mais forte) |
| Baselines | ❌ Ausentes | BL-02 — bloqueador |
| Validação estatística | ❌ Single run | BL-08 + múltiplas runs |
| Raw Message Store | ⚠️ Não verificado | Verificar — potencial bloqueador |
| Paper escrito | ❌ Ausente | Todas as seções |
| Integração MATTS | ❌ Interface não existe | Pós BL-09 + M1 pronto |

**O DyFO está funcionalmente sólido e os resultados preliminares são positivos. O único caminho crítico para publicação é: (1) verificar Raw Message Store, (2) implementar baselines ROLAND e GAT-static, (3) rodar 5+ seeds + 500-bootstrap, (4) escrever o paper.**

---

*Documento gerado em 2026-04-07 com base no estado do repositório main local, artigo TGN (Rossi et al., 2020) e Proposta MATTS v4.0.*
