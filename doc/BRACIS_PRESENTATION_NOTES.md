# Guia e Roteiro de Apresentação — DyFO (BRACIS 2026)

**Conferência:** *Brazilian Conference on Intelligent Systems (BRACIS 2026)*  
**Trilha:** *Machine Learning / Geometric Deep Learning / Multi-Agent & Reinforcement Learning*  
**Título Oficial do Trabalho:**  
*Heterogeneous Temporal Graph Attention with Typed Edge Conditioning for Financial Co-movement Forecasting*

**Formato:** Apresentação Oral (12 Slides • 15 Minutos)  
**Deck Interativo:** [`doc/bracis_presentation_deck.html`](file:///d:/projetos/DyFO/doc/bracis_presentation_deck.html)

---

## 🏛️ Posicionamento Estratégico & Enquadramento Teórico

> **Premissa de Excelência para o BRACIS:** O público é composto por pesquisadores seniores de Inteligência Artificial, revisores internacionais de Geometric Deep Learning e membros de bancas de Doutorado.  
> **Estratégia Pedagógica:** A apresentação parte dos **fundamentos teóricos de IA** (o que é um grafo temporal, como o TGAT clássico de Xu et al. 2020 formula a atenção em tempo contínuo e qual sua premissa homogênea implícita), introduz a **proposta do DyFO** em redes heterogêneas complexas, diagnostica a **falha analítica da diluição de atenção**, e demonstra empiricamente a resolução desse gargalo e seus impactos em Link Prediction e Deep Reinforcement Learning.

---

## 🎯 Roteiro Slide a Slide com Falas Acadêmicas Fluídas (Para Treino e Memorização)

---

### Slide 1: Título & Visão Geral da Pesquisa
- **Categoria:** `BRACIS 2026 • Inteligência Artificial & Finanças Quantitativas`
- **Mensagem Chave:** Introdução do DyFO como uma arquitetura de *Dynamic Heterogeneous Graph Neural Network* para streaming contínuo não-estacionário, resolvendo a diluição de atenção e o colapso de políticas em DRL.
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"Bom dia aos membros do comitê avaliador e colegas pesquisadores. Apresento o DyFO, uma arquitetura de Grafos Temporais Heterogêneos desenvolvida para resolver dois gargalos fundamentais em Inteligência Artificial: primeiro, a falha de atenção em grafos dinâmicos quando confrontados com arestas heterogêneas de naturezas e densidades distintas; segundo, o colapso entrópico de políticas em Deep Reinforcement Learning de alta dimensionalidade. Validamos nossa formulação na modelagem preditiva contínua de matrizes estocásticas de co-movimento sob regimes de choque extremo."*

---

### Slide 2: Fundamentos de IA: De GNNs Estáticas a Grafos Temporais Contínuos (TGAT)
- **Categoria:** `Fundamentos de IA & Modelagem Relacional`
- **Mensagem Chave:** Nivelamento conceitual: a transição de GNNs estáticas para grafos temporais em tempo contínuo (CTDG), a formulação do mecanismo de atenção do TGAT clássico (Xu et al., 2020) com Time2Vec, e sua premissa implícita de homogeneidade.
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"Para situar nossa contribuição, comecemos pelos fundamentos. GNNs estáticas clássicas, como GCN e GAT, assumem que a topologia da rede é invariante no tempo. Em sistemas dinâmicos reais, no entanto, as interações ocorrem como um fluxo assíncrono de eventos contínuos.  
  > O marco recente nessa área é o TGAT (Temporal Graph Attention Network) de Xu et al. (2020), que estendeu a atenção de grafos para o tempo contínuo ao incorporar uma representação harmônica de Fourier — o Time2Vec — calculada sobre a defasagem temporal \(t - t_j\).  
  > Contudo, o TGAT clássico possui uma premissa implícita fundamental: ele foi desenhado para grafos **homogêneos**, onde todas as arestas compartilham a mesma natureza semântica e a mesma densidade. A pergunta que motiva nosso trabalho é: o que acontece quando aplicamos essa formulação a sistemas complexos onde coexistem múltiplos tipos de arestas com dinâmicas conflitantes?"*

---

### Slide 3: Contexto de Aplicação & Proposta Central do DyFO
- **Categoria:** `Contexto de Aplicação & Proposta Central do Trabalho`
- **Mensagem Chave:** O mercado financeiro como um Grafo Temporal Heterogêneo (HTG) com 4 canais de informação (\(\text{CORR}, \text{SECT}, \text{FACT}, \text{SUPL}\)) e os dois objetivos do DyFO: previsão causal de covariância e regularização indutiva em DRL.
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"No mercado financeiro, a estrutura de dependência mútua entre ativos é altamente não-estacionária. Tradicionalmente, modelos estatísticos como EWMA e DCC-GARCH ignoram a topologia relacional e sofrem com atraso de fase durante transições de regime.  
  > Nós formulamos o mercado como um Grafo Temporal Heterogêneo com quatro canais de arestas: correlações estatísticas dinâmicas e voláteis (\(\text{CORR}\)); relações estruturais estáticas densas de setor (\(\text{SECT}\)); co-exposição a fatores de risco sistemáticos (\(\text{FACT}\)); e cadeias de suprimentos fundamentais (\(\text{SUPL}\)).  
  > A proposta central do DyFO é estender o TGAT com **Condicionamento Relacional Tipado**, permitindo ao modelo prever causalmente a matriz de covariância \(\mathbf{\Sigma}_{t+1}\) em tempo contínuo e atuar como viés indutivo indispensável para impedir o colapso de políticas em DRL."*

---

### Slide 4: Diagnóstico Teórico da Falha: Por que Modelos Homogêneos (TGAT) Falham?
- **Categoria:** `Diagnóstico Teórico da Falha`
- **Mensagem Chave:** A prova da **Diluição de Atenção**: arestas estáticas densas de setor (\(\mathcal{O}(N_{\text{sector}}^2)\)) sufocam links dinâmicos no cálculo do softmax, degradando o \(R^2\) em \(-0.0042\).
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"Ao aplicarmos o TGAT homogêneo a esse grafo heterogêneo, descobrimos uma falha analítica severa que denominamos Diluição de Atenção.  
  > Olhem a figura à direita: as arestas estáticas de setor possuem densidade combinatória muito maior do que as correlações dinâmicas voláteis. Como o TGAT padrão não possui semântica de aresta, a função softmax distribui a massa de probabilidade de atenção uniformemente sobre a vizinhança topológica. As arestas de setor afogam o sinal dinâmico, causando uma queda de \(R^2\) de \(-0.0042\) em relação ao modelo que usa apenas correlação. Mais dados resultaram em pior desempenho."*

---

### Slide 5: Contribuição Metodológica: Relation-Aware TGAT v2
- **Categoria:** `Contribuição Teórica & Arquitetural em IA`
- **Mensagem Chave:** Injeção direta de embeddings relacionais tipados \(\mathbf{W}_e \mathbf{e}_{ij} \in \mathbb{R}^{16}\) no softmax do `GATConv`, Time2Vec contínuo, Huber Loss (\(\delta=1.0\)), e vantagens conceituais sobre TGN (sem drift recorrente) e ROLAND (sem discretização em snapshots).
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"Nossa resposta metodológica é o Relation-Aware TGAT v2. Modificamos cirurgicamente o núcleo do mecanismo de atenção multi-head, injetando um embedding aprendível de relação \(\mathbf{e}_{ij} \in \mathbb{R}^{16}\) projetado por uma matriz linear \(\mathbf{W}_e\). A atenção passa a ser explicitamente modulada pelo tipo do link:
  > \[\alpha_{ij} = \text{softmax}_j \left( \text{LeakyReLU}\left( \mathbf{a}^T [ \mathbf{W}\mathbf{h}_i \, \Vert \, \mathbf{W}\mathbf{h}_j \, \Vert \, \mathbf{W}_e \mathbf{e}_{ij} ] \right) \right)\]
  > Combinamos essa projeção com a codificação temporal contínua Time2Vec e otimizamos via Huber Loss com \(\delta=1.0\). Diferente do TGN, não utilizamos células recorrentes GRU, eliminando a deriva de memória em sequências longas (\(T > 1000\)); e diferente do ROLAND, processamos eventos em tempo contínuo sem perda por discretização em snapshots."*

---

### Slide 6: Protocolo Experimental & Escalaridade Combinatória (N=18, 50, 100)
- **Categoria:** `Protocolo Experimental & Escalaridade Combinatória`
- **Mensagem Chave:** As 3 escalas combinatórias do benchmark, crescimento quadrático de arestas \(\mathcal{O}(N^2)\), e avaliação causal walk-forward sem vazamento temporal.
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"Para garantir rigor científico absoluto, estruturamos nossos experimentos em três escalas combinatórias bem definidas: com N=50 cobrindo todos os 11 setores do S&P 500, gerando 1.225 arestas diárias para validação estatística formal dos modelos de IA; com N=18 em ambiente multi-ativo heterogêneo (ações, títulos do tesouro, commodities e criptoativos) para avaliar regularização indutiva em DRL; e com N=100 no S&P 100 (e N=104 no ecossistema do Doutorado), totalizando mais de 5.300 arestas por dia, onde aplicamos sparsificação por limiar para manter alta fidelidade preditiva sem explosão computacional. Todos os testes seguem um protocolo walk-forward estritamente causal, com treino, validação e teste sem vazamento temporal."*

---

### Slide 7: Validação Empírica & Benchmarks SOTA (N=50 S&P 500)
- **Categoria:** `Validação Empírica & Benchmarks SOTA`
- **Mensagem Chave:** DyFO supera amplamente GAT-Static e ROLAND em 9 janelas não-sobrepostas (2018–2025). Superioridade comprovada por teste de Diebold-Mariano (\(p < 0.0001\)).
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"Os resultados empíricos em 9 janelas walk-forward independentes entre 2018 e 2025 demonstram a superioridade inequívoca do DyFO. Nossa arquitetura alcançou R² de 0.893, correlação de rank de Spearman de 0.958 e erro médio absoluto de apenas 0.035. Em comparação, o GAT-Estático atinge R² de 0.684 e o ROLAND atinge 0.518, colapsando ainda mais durante o regime de estresse de 2020 para 0.390. O teste estatístico de Diebold-Mariano com correção de autocorrelação de Newey-West rejeita a hipótese nula de igualdade preditiva com p < 0.0001."*

---

### Slide 8: Estudo de Ablação: De Diluição a Sinergia Relacional
- **Categoria:** `Estudo de Ablação`
- **Mensagem Chave:** Resolução da diluição de atenção provada em 4 configurações: o condicionamento de aresta converte a interferência de \(-0.0042\) em ganho de sinergia de \(+0.0410\) no \(R^2\) e eleva o Sharpe proxy de 2.45 para 2.68.
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"Este estudo de ablação é a prova de fogo da nossa contribuição teórica. Avaliamos quatro configurações sob as mesmas condições: apenas correlação, TGAT homogêneo com setor, TGAT com fatores e o nosso Relation-Aware TGAT v2. Enquanto a adição ingênua de arestas setoriais no TGAT homogêneo reduz o R² de 0.852 para 0.848 devido à diluição de atenção, o Relation-Aware TGAT v2 não apenas estanca a perda, mas destrava uma sinergia positiva, elevando o R² para 0.893 (+0.0410 de ganho) e o Sharpe proxy de 2.45 para 2.68. A semântica explícita de aresta é o elemento que transforma ruído estrutural em sinal preditivo de alta fidelidade."*

---

### Slide 9: Regularização Indutiva em DRL: Quebra de Simetria e Superação do Colapso Entrópico
- **Categoria:** `Utilidade Econômica & Quebra de Simetria em DRL`
- **Mensagem Chave:** Agente DRL padrão (Raw-DRL) sofre de colapso entrópico para política uniforme \(w_i = 1/N\) (\(H = 2.890 \approx \ln 18\)). DyFO-DRL+ quebra a simetria com embeddings de grafo (\(H = 2.615\)), gerando +1.72% de alpha (\(p=0.0312\)) com 63% menor turnover.
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"No domínio de Aprendizado por Reforço Profundo multi-agente e alocação de recursos, deparamo-nos com uma patologia teórica comum: agentes DRL alimentados apenas com séries temporais de preço e volatilidade sofrem de colapso de política, convergindo monotonicamente para a alocação uniforme 1/N. Isso é evidenciado pela entropia de Shannon máxima de H = 2.890, exatamente igual a ln(18). Ao injetarmos os embeddings de nós do DyFO no espaço de estados do PPO, os vetores topológicos atuam como uma regularização indutiva que quebra a simetria do espaço de ações (reduzindo H para 2.615). Como resultado prático, o DyFO-DRL+ gerou +1.72% de alpha acumulado com significância estatística (p=0.0312) e turnover 63% menor em relação ao benchmark EWMA-GMVP."*

---

### Slide 10: Robustez Sob Regimes de Estresse (Crash da COVID-19 em 2020)
- **Categoria:** `Regimes de Estresse & Robustez Não-Linear`
- **Mensagem Chave:** Rastreamento contínuo em tempo real da rápida decorrelação do par SPY - ^VIX em março de 2020 sem lag temporal. Huber Loss e Time2Vec garantem estabilidade sob saltos extremos.
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"Em sistemas críticos de IA, a robustez sob quebras estruturais de regime é mandatória. Analisamos o comportamento do modelo durante o crash de março de 2020, quando o índice VIX saltou de 15 para 82 pontos. O DyFO rastreou com precisão a decorrelação abrupta do par SPY versus VIX em tempo real, sem o atraso temporal de 10 a 20 dias característico de estimadores de janela móvel e sem instabilidade numérica de gradientes, graças à combinação da parametrização Time2Vec com a regularização da Huber Loss."*

---

### Slide 11: Integração na Tríade do Doutorado (PORTA, DyFO e ORION)
- **Categoria:** `Arquitetura de Sistemas & Ontologia Semântica`
- **Mensagem Chave:** Arquitetura modular de software, contrato estritamente *read-only* com o PORTA e exportação padronizada em ontologia OWL/RDF (`<TICKER>.US`).
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"O DyFO é o pilar de inteligência relacional e topológica de um ecossistema integrado de Doutorado. Ele consome dados curados em modo estritamente read-only do repositório PORTA, eliminando qualquer risco de vazamento; exporta snapshots estruturais causais para os modelos de alocação de risco do PORTA; e provê continuamente vetores de embedding relacional \(\mathbf{z}_t \in \mathbb{R}^{100}\) para o construtor de estados de percepção multimodal do agente ORION. Toda a representação de entidades segue uma ontologia semântica formal em OWL/RDF com predicados relacionais estritamente tipados."*

---

### Slide 12: Conclusões & Síntese das Contribuições em IA no BRACIS
- **Categoria:** `Conclusões & Contribuições para a Comunidade de IA`
- **Mensagem Chave:** Resumo das 3 contribuições principais para a comunidade de IA (resolução da diluição de atenção, regularização indutiva em DRL, rigor causal estrito) e reprodutibilidade aberta.
- **🗣️ Fala Sugerida (Fluida para Treino):**
  > *"Em síntese, apresentamos à comunidade do BRACIS três contribuições fundamentais: primeira, a resolução teórica e empírica da diluição de atenção em Dynamic Graph Neural Networks heterogêneas via condicionamento explícito de arestas; segunda, a demonstração de que embeddings topológicos dinâmicos funcionam como regularizadores indutivos que quebram a simetria em DRL de alta dimensionalidade; e terceira, um protocolo experimental rigoroso, causal e 100% reproduzível para aprendizado de máquina em streaming não-estacionário. Agradeço a atenção de todos e coloco-me à disposição para perguntas."*

---

---

## 🖼️ Guia Prático de Falas para Cada Imagem (Treino & Memorização Oral)

---

### 📷 Imagem 1 (Slides 1 e 4): Diagrama da Arquitetura TGAT Homogêneo vs Relation-Aware TGAT v2
*Arquivo visual:* `figures/bracis_slides/slide_01_tgat_v2_architecture.png`  
*Legenda no Slide:* `Arquitetura Relation-Aware TGAT v2 vs TGAT Homogêneo`

#### 🗣️ Texto para Fala Fluida:
> *"Olhem aqui para a figura à direita. No lado esquerdo dela, em vermelho, temos o comportamento do TGAT clássico. Notem que o nó central recebe conexões de correlação dinâmica, fatores de risco e setor estático com o mesmo peso estrutural. Como as arestas de setor são muito mais densas, elas 'sufocam' os links dinâmicos no cálculo do softmax — isso é a **Diluição de Atenção**.  
> Agora, vejam o contraste no lado direito, em verde: esta é a nossa inovação cirúrgica. Nós injetamos um embedding aprendível de 16 dimensões para cada tipo de aresta direto no mecanismo de atenção. O modelo agora sabe exatamente o que é correlação volátil e o que é setor estático, preservando o sinal dinâmico com alta fidelidade."*

#### 🧠 Âncoras de Memorização:
1. **Lado esquerdo (vermelho):** TGAT padrão mistura tudo $\to$ setor denso afoga correlação dinâmica (Diluição de Atenção).
2. **Lado direito (verde):** Injeção do embedding de aresta de 16D direto no cálculo de atenção.
3. **Takeaway:** O modelo passa a diferenciar a natureza de cada link sem perder sinal.

---

### 📷 Imagem 2 (Slide 6): Escalaridade de Universo e Densidade Combinatória
*Arquivo visual:* `figures/bracis_slides/slide_04_universe_scaling_density.png`  
*Legenda no Slide:* `Complexidade de Grafo e Acurácia por Escala de Ativos`

#### 🗣️ Texto para Fala Fluida:
> *"Neste gráfico, mostramos a transparência e o rigor da nossa escala experimental.  
> No painel da esquerda, vejam como o número de conexões diárias cresce quadraticamente com $\mathcal{O}(N^2)$: vai de 153 links no painel multi-ativo de 18 nós, passa por 1.225 links no benchmark principal do artigo com 50 ativos do S&P 500, e chega a mais de 5.300 links no ecossistema completo de 104 ativos.  
> No painel da direita, o ponto crucial: observem que mesmo sob densidade combinatorial extrema, as barras azuis de $R^2$ e verdes de correlação de Spearman permanecem acima de 0.86 e 0.94. Isso prova que nossa arquitetura escala sem perder precisão nem explodir computacionalmente."*

#### 🧠 Âncoras de Memorização:
1. **Painel Esquerdo (Complexidade):** $N=18$ (153 links), $N=50$ (1.225 links do paper) e $N=104$ (>5.300 links).
2. **Painel Direito (Generalização):** $R^2$ acima de 0.86 e Spearman acima de 0.94 em todas as escalas.
3. **Takeaway:** O grafo cresce quadraticamente, mas a capacidade preditiva se mantém estável.

---

### 📷 Imagem 3 (Slide 7): Comparativo de Acurácia Preditiva e Spearman Rank
*Arquivo visual:* `figures/bracis_slides/slide_02_predictive_and_portfolio_results.png` *(Painel Esquerdo)*  
*Legenda no Slide:* `Comparativo de Acurácia Preditiva e Spearman Rank`

#### 🗣️ Texto para Fala Fluida:
> *"Observem a comparação direta entre o DyFO e os modelos de ponta da literatura.  
> As duas primeiras barras mostram o DyFO em azul e verde com $R^2$ de 0.893 e Spearman de 0.958. Vejam como o TGN recorrente fica atrás, em 0.803, por causa do acúmulo de drift na memória GRU.  
> E olhem a queda acentuada nos baselines estáticos e no ROLAND, que colapsa para 0.390 em regimes de estresse porque discretiza o tempo em snapshots mensais. Nosso teste de Diebold-Mariano com correção Newey-West confirma essa superioridade com $p < 0.0001$."*

#### 🧠 Âncoras de Memorização:
1. **Primeiras barras (DyFO):** Liderança com $R^2 = 0.893$ e Spearman $\rho = 0.958$.
2. **TGN vs ROLAND:** TGN perde por drift recorrente (0.803); ROLAND perde por snapshots discretos (0.390).
3. **Estatística:** Teste Diebold-Mariano atesta significância com $p < 0.0001$.

---

### 📷 Imagem 4 (Slide 8): Estudo de Ablação — R² e Sharpe Ratio Proxy
*Arquivo visual:* `figures/bracis_slides/slide_05_ablation_edge_types.png`  
*Legenda no Slide:* `Ablação Estrutural: R² e Sharpe Ratio por Configuração`

#### 🗣️ Texto para Fala Fluida:
> *"Esta é a comprovação empírica definitiva do nosso mecanismo de atenção.  
> Olhem para a terceira coluna do gráfico da esquerda: quando adicionamos arestas de setor no TGAT homogêneo, a barra cinza cai 0.0042 pontos. Ou seja: mais informação gerou um resultado pior por causa da diluição.  
> Agora vejam a barra vermelha do DyFO na mesma coluna: o $R^2$ dá um salto de 0.852 para 0.893 — um ganho líquido de +0.0410. E no gráfico da direita, esse ganho se traduz diretamente em utilidade econômica, elevando o Sharpe proxy de 2.45 para 2.68. O condicionamento relacional transforma interferência destrutiva em ganho de sinergia."*

#### 🧠 Âncoras de Memorização:
1. **Coluna 3 (Setor):** Barra cinza cai ($-0.0042$, diluição no homogêneo); barra vermelha salta ($+0.0410$, ganho no DyFO).
2. **Gráfico da Direita (Sharpe):** Sobe de 2.45 para 2.68.
3. **Takeaway:** O vetor de tipo de aresta converte ruído estrutural em sinergia positiva.

---

### 📷 Imagem 5 (Slide 9): Retorno Acumulado e Entropia de Alocação em DRL
*Arquivo visual:* `figures/bracis_slides/slide_02_predictive_and_portfolio_results.png` *(Painel Direito)*  
*Legenda no Slide:* `Retorno Acumulado e Entropia de Alocação`

#### 🗣️ Texto para Fala Fluida:
> *"Aqui apresentamos a utilidade em Aprendizado por Reforço Profundo.  
> Reparem na quarta barra, do Raw-DRL sem grafo: a entropia dele é 2.89, exatamente igual a $\ln(18)$. Isso significa que o agente padrão sofre colapso entrópico e vira uma carteira uniforme ingênua $1/N$.  
> Agora olhem para a primeira barra, do DyFO-DRL+: a entropia cai para 2.615 e o retorno acumulado sobe para +3.37%. O grafo atua como um viés indutivo que quebra a simetria do espaço de ações, permitindo ao agente aprender alocações estruturadas com +1.72% de alpha e 63% menor turnover sobre o benchmark."*

#### 🧠 Âncoras de Memorização:
1. **Quarta barra (Raw-DRL):** Entropia $H = 2.89 \approx \ln(18) \to$ colapso na política ingênua $1/N$.
2. **Primeira barra (DyFO-DRL+):** Entropia cai para $2.615 \to$ quebra de simetria com retorno de +3.37%.
3. **Impacto:** Regularização indutiva topológica gera +1.72% de alpha e 63% menor turnover.

---

### 📷 Imagem 6 (Slide 10): Rastreamento Contínuo de Correlação SPY-VIX na COVID-19
*Arquivo visual:* `figures/bracis_slides/slide_03_stress_regime_spy_vix.png`  
*Legenda no Slide:* `Rastreamento Contínuo de Correlação SPY-VIX (COVID-19)`

#### 🗣️ Texto para Fala Fluida:
> *"Nesta série temporal, avaliamos o comportamento do modelo durante o crash de março de 2020.  
> Na faixa sombreada em rosa, o par SPY versus VIX sofre uma decorrelação abrupta, saltando de -0.85 para -0.25. A linha preta é o alvo real e a curva vermelha é a previsão do DyFO. Vejam como o DyFO acompanha perfeitamente a inflexão da curva em tempo real.  
> Em contraste, a linha tracejada cinza de persistência sofre um atraso de fase sistemático, e a linha pontilhada azul média erra completamente. A combinação de Time2Vec contínuo com Huber Loss impediu a explosão de gradientes nesse regime extremo."*

#### 🧠 Âncoras de Memorização:
1. **Zona rosa (Crash 2020):** Correlação salta de -0.85 para -0.25.
2. **Linha vermelha (DyFO):** Rastreia colado na linha preta (alvo) sem atraso temporal.
3. **Linhas cinza e azul (Baselines):** Sofrem atraso de fase ou erram o regime completamente.

---

### 📷 Imagem 7 (Slide 11): Fluxo de Dados e Contratos da Tríade de Pesquisa
*Arquivo visual:* `figures/bracis_slides/slide_06_triad_architecture_integration.png`  
*Legenda no Slide:* `Fluxo de Dados e Contratos da Tríade de Pesquisa`

#### 🗣️ Texto para Fala Fluida:
> *"Para concluir a visão de engenharia, este diagrama ilustra o papel do DyFO na tríade do Doutorado.  
> O DyFO, na caixa vermelha central, recebe da caixa azul do PORTA os tensores multidimensionais de preços e regimes em modo estritamente read-only, eliminando qualquer risco de vazamento causal.  
> Em seguida, ele exporta snapshots ontológicos em OWL/RDF de volta para os modelos de risco do PORTA e injeta continuamente vetores de embedding relacional de 100 dimensões na caixa verde do agente multimodal ORION. Trata-se de uma arquitetura modular, formalmente tipada e com ciência 100% aberta."*

#### 🧠 Âncoras de Memorização:
1. **Caixa Azul (PORTA $\to$ DyFO):** Alimentação de dados curados em modo *read-only* estrito.
2. **Caixa Vermelha (DyFO - Centro):** Processamento relacional contínuo e exportação ontológica (OWL/RDF).
3. **Caixa Verde (DyFO $\to$ ORION):** Injeção de embeddings relacionais de 100D no estado do agente DRL.

---

---

## ⏱️ Guia de Gestão de Tempo (15 Minutos de Apresentação Oral • 12 Slides)

| Bloco | Slides | Tópicos Centrais | Tempo Sugerido | Acumulado |
| :--- | :--- | :--- | :--- | :--- |
| **I. Motivação & Fundamentos de IA** | 1, 2, 3 | Título, GNNs para TGAT, Redes Heterogêneas e Proposta do DyFO | 4 min 00 s | 4 min 00 s |
| **II. Diagnóstico da Falha & Solução TGAT v2** | 4, 5 | Diluição de Atenção & Formulação Relation-Aware | 3 min 00 s | 7 min 00 s |
| **III. Metodologia, Escala & Validação SOTA** | 6, 7 | Escalas $N=18, 50, 100$, Benchmarks & Diebold-Mariano | 2 min 30 s | 9 min 30 s |
| **IV. Ablação & Aplicação em DRL** | 8, 9 | Ganho Sinérgico ($+0.0410$) & Quebra de Simetria ($H=2.615$) | 2 min 30 s | 12 min 00 s |
| **V. Estresse, Tríade Doutoral & Conclusões** | 10, 11, 12 | COVID-19, Ontologia OWL/RDF & 3 Contribuições em IA | 3 min 00 s | 15 min 00 s |

---

## 🔬 Banco de Perguntas e Respostas Teóricas da Banca (Q&A)

### Pergunta 1: O que exatamente é o TGAT e por que vocês o escolheram como base?
- **Resposta Técnica:**
  > *"O TGAT (Temporal Graph Attention Network) proposto por Xu et al. no ICLR 2020 é a extensão do mecanismo de atenção de grafos (GAT) para grafos dinâmicos em tempo contínuo. Ele utiliza uma representação harmônica de Fourier (baseada no teorema de Bochner e no Time2Vec) para codificar a defasagem temporal de eventos contínuos diretamente no cálculo de atenção. Escolhemos o TGAT como nossa espinha dorsal porque ele é stateless em relação a memória de nós (não usa células recorrentes GRU/LSTM), o que evita a deriva de estado e permite escalabilidade analítica em horizontes longos."*

### Pergunta 2: Por que vocês não utilizaram o TGN (Temporal Graph Network) com memória recorrente GRU/LSTM?
- **Resposta Técnica:**
  > *"O TGN (Rossi et al., 2020) depende de um módulo de memória de nós baseado em GRU/RNN. Em séries temporais financeiras longas e não-estacionárias com milhares de passos temporais, memórias recorrentes sofrem de dois problemas graves: deriva de estado acumulado (state drift) e a necessidade de Backpropagation Through Time truncado (TBPTT), que introduz viés no aprendizado de longo prazo. O DyFO utiliza atenção puramente baseada em vizinhança temporal com Time2Vec, garantindo inferência sem estado latente recorrente (stateless recurrence), o que elimina o drift e assegura maior estabilidade sob choques de mercado."*

### Pergunta 3: Como vocês garantem que não existe *Look-Ahead Bias* na previsão do grafo?
- **Resposta Técnica:**
  > *"A causalidade estrita foi a diretriz central da nossa engenharia. O cálculo dos alvos supervisionados de correlação utiliza estritamente janelas causais passadas \([t-W, t]\) via Rolling Pearson causal. Na etapa de inferência no instante \(t\), o DyFO recebe apenas eventos ocorridos em \(t' \le t\) para prever a topologia de co-movimento em \(t+1\). Nosso pipeline foi auditado e validado com testes unitários automatizados específicos para invariância causal e isolamento temporal."*

### Pergunta 4: O que comprova matematicamente que os embeddings de grafo são responsáveis pelo ganho no DRL?
- **Resposta Técnica:**
  > *"A comprovação reside no diagnóstico da Entropia de Shannon da distribuição de pesos de alocação da política do PPO. O agente Raw-DRL (sem grafo) colapsa monotonicamente para \(H \approx 2.890\), que coincide exatamente com o limite teórico de máxima incerteza \(\ln(N) = \ln(18) = 2.89037\), caracterizando uma política uniforme ingênua \(1/N\). Ao adicionar os embeddings do DyFO, a entropia cai para \(H = 2.615\), demonstrando que o sinal topológico quebrou a simetria do espaço de ações, permitindo ao agente convergir para fronteiras eficientes de Markowitz dinâmicas com +1.72% de alpha e 63% menor turnover."*

### Pergunta 5: Como o DyFO escala para grafos maiores com centenas de nós (\(N \ge 100\))?
- **Resposta Técnica:**
  > *"Em grafos completos, o número de arestas cresce quadraticamente com \(O(N^2)\). Para o universo de \(N=100\) (S&P 100) e \(N=104\) (PORTA), aplicamos sparsificação adaptativa por limiar (\(\tau = 0.30\)), que filtra arestas de correlação estatisticamente espúrias e reduz o grafo a uma estrutura esparsa. Em nossos experimentos, o modelo manteve \(R^2 = 0.865\) com tempo de inferência inferior a 12 milissegundos por snapshot temporal. Para o roadmap futuro, estamos implementando sparsificação topológica baseada em TMFG (Triangulated Maximally Filtered Graph) com complexidade linear em arestas \(O(3N-6)\)."*

### Pergunta 6: Por que adotar a Huber Loss (\(\delta=1.0\)) em vez do tradicional Mean Squared Error (MSE)?
- **Resposta Técnica:**
  > *"Séries financeiras e distribuições de correlação empíricas possuem caudas pesadas (leptocurtose) e saltos de regime (jumps). O MSE eleva ao quadrado os erros em regimes extremos (como em março de 2020), gerando explosão de normas de gradientes e instabilidade no treinamento do otimizador AdamW. A Huber Loss comporta-se quadraticamente para erros pequenos (\(|e| \le \delta\)) e linearmente para grandes desvios (\(|e| > \delta\)), limitando a magnitude máxima dos gradientes a \(\pm \delta\). Isso proporcionou convergência monotônica sem necessidade de gradient clipping excessivo."*

### Pergunta 7: Como foi configurado o teste estatístico de Diebold-Mariano para validar a significância preditiva?
- **Resposta Técnica:**
  > *"O teste de Diebold-Mariano (1995) foi aplicado comparando as séries temporais diárias da função de perda quadrática das previsões do DyFO contra o GAT-Static e o ROLAND ao longo de todas as janelas de teste out-of-sample. Para lidar com a autocorrelação e heteroscedasticidade inerentes a séries temporais em streaming contínuo, utilizamos o estimador de variância assintótica com correção espectral de Newey-West com defasagem ótima \(h = \lfloor 4(T/100)^{2/9} \rfloor\). A estatística \(t = -14.82\) rejeita a hipótese nula com \(p < 0.0001\)."*
