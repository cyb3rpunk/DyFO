# Guia e Roteiro de Apresentação — DyFO (BRACIS 2026)

**Título Oficial:**  
*Heterogeneous Temporal Graph Attention with Typed Edge Conditioning for Financial Co-movement Forecasting*

**Formato:** Apresentação Oral (11 Slides)  
**Deck Interativo:** [`doc/bracis_presentation_deck.html`](file:///d:/projetos/DyFO/doc/bracis_presentation_deck.html)

---

## 🎯 Roteiro Slide a Slide com Falas do Apresentador

### Slide 1: Título e Visão Geral da Pesquisa
- **Mensagem Chave:** O DyFO introduz uma modificação cirúrgica no TGAT para previsão causal de co-movimentos e matrizes de covariância em tempo contínuo.
- **Fala Sugerida:**
  > *"Bom dia a todos. Hoje apresento o DyFO, uma arquitetura de Grafos Temporais Heterogêneos desenvolvida no âmbito da nossa pesquisa de Doutorado para resolver um problema fundamental em finanças quantitativas: prever matrizes dinâmicas de correlação e covariância em tempo contínuo a partir de fluxos assíncronos de eventos de mercado."*

---

### Slide 2: Contexto e Não-Estacionariedade em Redes Financeiras
- **Mensagem Chave:** A necessidade de matrizes dinâmicas $\mathbf{\Sigma}_{t+1}$ precisas e o fracasso dos métodos econométricos clássicos (EWMA, DCC-GARCH) em capturar topologias ricas sob choques rápidos.
- **Fala Sugerida:**
  > *"O gerenciamento de risco e portfólio depende criticamente de prever a estrutura de dependência futura dos ativos. Métodos clássicos como EWMA ou DCC-GARCH assumem dinâmica linear ou sofrem de atraso temporal durante choques de cauda. Grafos temporais nos permitem tratar a covariância como um problema de Link Prediction contínuo, fundindo múltiplos canais de transmissão de risco."*

---

### Slide 3: Diagnóstico da Falha: O Problema da Diluição de Atenção
- **Mensagem Chave:** Modelos SOTA homogêneos (TGAT original) falham em redes financeiras porque arestas estáticas densas (SECT) afogam a atenção sobre arestas dinâmicas críticas (CORR).
- **Fala Sugerida:**
  > *"Por que os modelos de grafos temporais clássicos falham no mercado financeiro? Redes financeiras são inerentemente heterogêneas: temos correlações estatísticas dinâmicas, setores industriais GICS e fatores sistemáticos de Fama-French. No TGAT original, homogêneo, as arestas estáticas de setor dominam numericamente o grafo e diluem o peso de atenção, reduzindo o R² em vez de ajudar."*

---

### Slide 4: Contribuição Metodológica: Relation-Aware TGAT v2
- **Mensagem Chave:** A formulação da atenção condicionada por tipo de aresta $\mathbf{W}_e \mathbf{e}_{ij}$ em `GATConv(edge_dim=16)`.
- **Fórmula:**
  $$\alpha_{ij} = \text{softmax}_j \left( \text{LeakyReLU}\left( \mathbf{a}^T [ \mathbf{W}\mathbf{h}_i \, \Vert \, \mathbf{W}\mathbf{h}_j \, \Vert \, \mathbf{W}_e \mathbf{e}_{ij} ] \right) \right)$$
- **Fala Sugerida:**
  > *"Nossa contribuição cirúrgica foi injetar embeddings aprendíveis de tipo de aresta diretamente no mecanismo de atenção do GATConv, modulando o fluxo de mensagens por relação sem memória recorrente. Isso elimina a deriva de estado de longo prazo típica do TGN e evita a perda de sinal temporal causada pela discretização em snapshots do ROLAND."*

---

### Slide 5: Escalaridade de Universo: N=18 (DRL), N=50 (Paper) e N=104 (PORTA)
- **Mensagem Chave:** **Diferenciação transparente e justificada do número de tickers em cada experimento.**
- **Dados:**
  - $N=18$ (Multi-Asset Class): 153 links/dia (Ações, TLT, GLD, BTC-USD) &rarr; Foco em balanceamento de alocação DRL.
  - $N=50$ (Paper Benchmark): 1.225 links/dia em 11 setores GICS &rarr; Foco em significância estatística do link-prediction.
  - $N=104$ (PORTA Ecosystem): 5.356 links/dia &rarr; Escala industrial de longo prazo (2005–2026).
- **Fala Sugerida:**
  > *"É crucial destacar a distinção de escala entre os nossos experimentos: no benchmark preditivo do artigo usamos N=50 ativos cobrindo todos os 11 setores do S&P 500 para maximizar o rigor estatístico sobre 1.225 links diários (R² = 0.893). Já na validação de DRL usamos N=18 ativos multi-classe (ações, títulos do tesouro, ouro e bitcoin) para testar a diversificação inter-mercado. E no ecossistema completo do PORTA, o DyFO opera sobre 104 ativos com 5.356 links diários sem perda de convergência."*

---

### Slide 6: Resultados Preditivos Walk-Forward (50 Ativos S&P 500)
- **Mensagem Chave:** DyFO supera amplamente GAT-Static e ROLAND em 9 janelas não-sobrepostas.
- **Métricas:** DyFO ($R^2 = 0.893$, Spearman $\rho = 0.958$, Pearson $r = 0.952$, MAE $= 0.035$).
- **Fala Sugerida:**
  > *"Em 9 janelas walk-forward de teste entre 2018 e 2025, o DyFO alcançou R² de 0.893 e correlação de rank de Spearman de 0.958, superando tanto o GAT-Estático (0.565) quanto o ROLAND (0.390). O teste de Diebold-Mariano com correção Newey-West confirma significância estatística com p < 0.0001."*

---

### Slide 7: Estudo de Ablação: Eliminando a Diluição de Atenção
- **Mensagem Chave:** No TGAT homogêneo, adicionar SECT derruba o R² (-0.0042); no TGAT v2 Relation-Aware, gera ganho de sinergia (+0.0410).
- **Fala Sugerida:**
  > *"O estudo de ablação comprova a nossa hipótese central: quando adicionamos relações estáticas de setor no TGAT homogêneo, o desempenho cai devido à diluição. No Relation-Aware TGAT v2, o condicionamento de aresta destrava a sinergia positiva, elevando o R² para 0.893 e o Sharpe proxy para 2.68."*

---

### Slide 8: Utilidade Econômica em Gerenciamento de Portfólio DRL
- **Mensagem Chave:** DyFO-DRL+ bate EWMA-GMVP com 100% de win-rate (+1.72% alpha), menor turnover (0.025 vs 0.083), enquanto o Raw-DRL colapsa para a carteira $1/N$.
- **Fala Sugerida:**
  > *"Para validar a utilidade econômica, integramos os embeddings do DyFO a um agente DRL de alocação de portfólio. O DyFO-DRL+ superou o baseline clássico EWMA-GMVP em 100% dos episódios de teste com menor turnover. Mais importante: o agente sem grafo (Raw-DRL) colapsou monotonicamente para a alocação ingênua 1/N (entropia 2.890), provando que os embeddings relacionais são o habilitador que permite ao DRL aprender políticas eficientes."*

---

### Slide 9: Robustez em Regimes de Estresse (Crash do COVID-19)
- **Mensagem Chave:** Rastreamento contínuo da rápida decorrelação do par SPY - ^VIX em março de 2020 sem lag temporal.
- **Fala Sugerida:**
  > *"Durante o choque de março de 2020, o DyFO rastreou com precisão a rápida decorrelação não-linear entre o SPY e o índice de volatilidade VIX, sem atraso de lag e sem sofrer contaminação por outliers graças ao Huber Loss e à codificação contínua Time2Vec."*

---

### Slide 10: Integração na Tríade do Doutorado (PORTA, DyFO e ORION)
- **Mensagem Chave:** Arquitetura de software modular, contrato estritamente *read-only* com o PORTA e exportação padronizada em ontologia OWL/RDF.
- **Fala Sugerida:**
  > *"O DyFO não é um modelo isolado, mas o motor relacional da nossa Tríade de Doutorado. Ele consome dados curados em modo estritamente read-only do PORTA, exporta matrizes de covariância estruturais causais para os alocadores do PORTA e fornece embeddings relacionais de estado para o agente multimodal ORION."*

---

### Slide 11: Conclusões e Próximos Passos
- **Mensagem Chave:** Resumo das contribuições e próximos passos da pesquisa.
- **Fala Sugerida:**
  > *"Em conclusão: propusemos uma modificação cirúrgica que resolve a diluição de atenção em grafos financeiros, demonstramos superioridade estatística e utilidade prática em DRL, e integramos o DyFO a um ecossistema reproduzível e robusto. Muito obrigado."*
