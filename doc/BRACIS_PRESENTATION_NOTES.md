# Guia e Roteiro de Apresentação — DyFO (BRACIS 2026)

**Conferência:** *Brazilian Conference on Intelligent Systems (BRACIS 2026)*  
**Trilha:** *Machine Learning / Geometric Deep Learning / Multi-Agent & Reinforcement Learning*  
**Título Oficial do Trabalho:**  
*Heterogeneous Temporal Graph Attention with Typed Edge Conditioning for Financial Co-movement Forecasting*

**Formato:** Apresentação Oral (11 Slides)  
**Deck Interativo:** [`doc/bracis_presentation_deck.html`](file:///d:/projetos/DyFO/doc/bracis_presentation_deck.html)

---

## 🏛️ Posicionamento Estratégico para o BRACIS

> **Premissa de Excelência:** O BRACIS é o principal congresso de Inteligência Artificial da América Latina. O público é composto por pesquisadores seniores de Aprendizado de Máquina, membros de bancas de Doutorado e revisores internacionais de IA.  
> **Tom e Abordagem:** O discurso deve priorizar as **contribuições fundamentais de IA** (Geometric Deep Learning, Dynamic Graph Neural Networks, regularização indutiva em Deep Reinforcement Learning e causalidade em fluxos não-estacionários), utilizando as finanças quantitativas como um domínio estocástico desafiador de validação empírica.

---

## 🎯 Roteiro Slide a Slide com Falas Acadêmicas do Apresentador

### Slide 1: Título e Visão Geral da Pesquisa
- **Mensagem Chave:** Introdução do DyFO como uma arquitetura de *Dynamic Heterogeneous Graph Neural Network* para streaming contínuo não-estacionário, resolvendo a diluição de atenção e o colapso de políticas em DRL.
- **Fala Sugerida:**
  > *"Bom dia aos membros do comitê e colegas pesquisadores. Apresento o DyFO, uma arquitetura de Grafos Temporais Heterogêneos desenvolvida para solucionar dois gargalos fundamentais em Inteligência Artificial: primeiro, a falha de atenção em grafos dinâmicos quando confrontados com arestas heterogêneas de naturezas e densidades distintas; segundo, o colapso entrópico de políticas em Deep Reinforcement Learning de alta dimensionalidade. Validamos nossa formulação na modelagem preditiva contínua de matrizes estocásticas de co-movimento sob regimes de choque extremo."*

---

### Slide 2: Contexto e Não-Estacionariedade em Grafos Temporais Financeiros
- **Mensagem Chave:** Redes financeiras como sistemas dinâmicos complexos com múltiplos canais semânticos (estatístico, setorial, fatores de risco) e a insuficiência de modelos paramétricos tradicionais (DCC-GARCH, EWMA).
- **Fala Sugerida:**
  > *"Em sistemas estocásticos não-estacionários, como o mercado de capitais, a estrutura de dependência mútua entre variáveis evolui em tempo contínuo. Modelos clássicos de séries temporais assumem linearidade ou impõem estruturas de covariância estáticas que sofrem de atraso crítico durante transições de fase e choques exógenos. Formulamos esse desafio sob a ótica de Geometric Deep Learning: um problema de Link Prediction Causal e Contínuo sobre um Grafo Heterogêneo Dinâmico \(\mathcal{G}_t = (\mathcal{V}, \mathcal{E}_t, \mathcal{R})\), onde os nós representam ativos e as arestas codificam fluxos assíncronos de co-movimento e topologias semânticas."*

---

### Slide 3: Diagnóstico Teórico da Falha: Diluição de Atenção no TGAT Homogêneo
- **Mensagem Chave:** Prova analítica e experimental de que o TGAT homogêneo (Xu et al., 2020) sofre de *Attention Dilution* quando arestas estáticas densas (\(\text{SECT}\), \(O(N_{\text{sector}}^2)\)) sufocam arestas dinâmicas esparsas (\(\text{CORR}\)), causando queda no \(R^2\) (\(-0.0042\)).
- **Fala Sugerida:**
  > *"Ao aplicarmos modelos de ponta como o TGAT homogêneo de Xu et al. a grafos heterogêneos, identificamos um fenômeno de interferência destrutiva que denominamos Diluição de Atenção. Redes heterogêneas combinam arestas estáticas densas de agrupamento setorial com arestas dinâmicas esparsas e voláteis de correlação estatística. No TGAT padrão, a função softmax distribui a massa de probabilidade de atenção uniformemente sobre a vizinhança topológica. Como as arestas de setor dominam numericamente a vizinhança, elas afogam o sinal das arestas dinâmicas, resultando em uma degradação de \(R^2\) de -0.0042 em relação ao modelo que usa apenas correlações."*

---

### Slide 4: Contribuição Metodológica: Relation-Aware TGAT v2
- **Mensagem Chave:** Modificação cirúrgica no `GATConv(edge_dim=16)` com injeção de embeddings relacionais tipados \(\mathbf{W}_e \mathbf{e}_{ij}\), codificação temporal contínua *Time2Vec*, e loss robusta de Huber (\(\delta=1.0\)). Superioridade conceitual sobre TGN (sem drift de memória recorrente) e ROLAND (sem perda de discretização em snapshots).
- **Fórmula:**
  $$\alpha_{ij} = \text{softmax}_j \left( \text{LeakyReLU}\left( \mathbf{a}^T [ \mathbf{W}\mathbf{h}_i \, \Vert \, \mathbf{W}\mathbf{h}_j \, \Vert \, \mathbf{W}_e \mathbf{e}_{ij} ] \right) \right)$$
- **Fala Sugerida:**
  > *"Nossa resposta metodológica é o Relation-Aware TGAT v2. Modificamos a formulação do mecanismo de atenção injetando um embedding aprendível de relação \(\mathbf{e}_{ij} \in \mathbb{R}^{16}\) diretamente no cálculo dos coeficientes de atenção multi-head via projeção linear \(\mathbf{W}_e\). Isso condiciona a propagação de mensagens à semântica de cada aresta. Para o tempo contínuo, adotamos a representação harmônica contínua Time2Vec com frequências aprendíveis \(\omega_k \Delta t + \phi_k\), e otimizamos os parâmetros via Huber Loss com \(\delta=1.0\), garantindo robustez de gradientes contra caudas pesadas. Diferente do TGN, nossa arquitetura não utiliza células recorrentes (GRU/LSTM), eliminando o problema de deriva de memória em horizontes longos; e diferente do ROLAND, processamos eventos em tempo contínuo sem perda por discretização em snapshots."*

---

### Slide 5: Protocolo Experimental & Escalaridade Combinatória: N=18, N=50 e N=100
- **Mensagem Chave:** Transparência metodológica sobre as três escalas combinatórias do benchmark, densidade quadrática \(O(N^2)\), e formulação estrita de *Causal Link Prediction* em streaming walk-forward.
- **Dados:**
  - **\(N=18\) (Multi-Asset DRL):** 153 links/dia &rarr; Ações, Renda Fixa (TLT), Ouro (GLD), Cripto (BTC) para teste de quebra de simetria em DRL.
  - **\(N=50\) (Paper Benchmark):** 1.225 links/dia em 11 setores GICS &rarr; Rigor estatístico e significância de link prediction (\(R^2 = 0.893\)).
  - **\(N=100/104\) (S&P 100 & PORTA):** 4.950 a 5.356 links/dia &rarr; Escalaridade combinatória ampla com sparsificação por limiar (\(\tau = 0.3\), \(R^2 = 0.865\)).
- **Fala Sugerida:**
  > *"Para garantir rigor científico absoluto, estruturamos nossos experimentos em três escalas combinatórias bem definidas: com N=50 cobrindo todos os 11 setores do S&P 500, gerando 1.225 arestas diárias para validação estatística formal dos modelos de IA; com N=18 em ambiente multi-ativo heterogêneo (ações, títulos do tesouro, commodities e criptoativos) para avaliar regularização indutiva em DRL; e com N=100 no S&P 100 (e N=104 no ecossistema do Doutorado), totalizando mais de 5.300 arestas por dia, onde aplicamos sparsificação por limiar para manter alta fidelidade preditiva sem explosão computacional. Todos os testes seguem um protocolo walk-forward estritamente causal, com treino, validação e teste sem vazamento temporal."*

---

### Slide 6: Validação Empírica & Benchmarks SOTA (N=50 S&P 500)
- **Mensagem Chave:** DyFO supera amplamente GAT-Static e ROLAND em 9 janelas não-sobrepostas (2018–2025). Superioridade com teste estatístico de Diebold-Mariano (\(p < 0.0001\)).
- **Métricas:**
  - **DyFO v2:** \(R^2 = 0.893 \pm 0.012\), Spearman \(\rho = 0.958\), Pearson \(r = 0.952\), MAE \(= 0.035\).
  - **GAT-Static:** \(R^2 = 0.684\) (global) / \(0.565\) (estresse COVID-19), MAE \(= 0.061\).
  - **ROLAND (Snapshot DGNN):** \(R^2 = 0.518\) (global) / \(0.390\) (estresse), MAE \(= 0.082\).
  - **Teste Diebold-Mariano:** \(t = -14.82\), \(p < 0.0001\) com ajuste Newey-West.
- **Fala Sugerida:**
  > *"Os resultados empíricos em 9 janelas walk-forward independentes entre 2018 e 2025 demonstram a superioridade inequívoca do DyFO. Nossa arquitetura alcançou R² de 0.893, correlação de rank de Spearman de 0.958 e erro médio absoluto de apenas 0.035. Em comparação, o GAT-Estático atinge R² de 0.684 e o ROLAND atinge 0.518, colapsando ainda mais durante o regime de estresse de 2020 para 0.390. O teste estatístico de Diebold-Mariano com correção de autocorrelação de Newey-West rejeita a hipótese nula de igualdade preditiva com p < 0.0001."*

---

### Slide 7: Estudo de Ablação: De Diluição a Sinergia Relacional
- **Mensagem Chave:** Resolução da diluição de atenção provada em 4 configurações controladas: o condicionamento de aresta converte a interferência de \(-0.0042\) em ganho de sinergia de \(+0.0410\) no \(R^2\) e eleva o Sharpe proxy de \(2.45\) para \(2.68\).
- **Fala Sugerida:**
  > *"Este estudo de ablação é a prova de fogo da nossa contribuição teórica. Avaliamos quatro configurações sob as mesmas condições: apenas correlação, TGAT homogêneo com setor, TGAT com fatores e o nosso Relation-Aware TGAT v2. Enquanto a adição ingênua de arestas setoriais no TGAT homogêneo reduz o R² de 0.852 para 0.848 devido à diluição de atenção, o Relation-Aware TGAT v2 não apenas estanca a perda, mas destrava uma sinergia positiva, elevando o R² para 0.893 (+0.0410 de ganho) e o Sharpe proxy de 2.45 para 2.68. A semântica explícita de aresta é o elemento que transforma ruído estrutural em sinal preditivo de alta fidelidade."*

---

### Slide 8: Regularização Indutiva em DRL: Quebra de Simetria e Superação do Colapso Entrópico
- **Mensagem Chave:** Agente DRL padrão (Raw-DRL) sofre de colapso entrópico para política uniforme \(w_i = 1/N\) (\(H = 2.890 \approx \ln 18\)). DyFO-DRL+ quebra a simetria com embeddings de grafo (\(H = 2.615\)), gerando \(+1.72\%\) de alpha (\(p=0.0312\)) com \(63\%\) menor turnover (0.025 vs 0.083).
- **Fala Sugerida:**
  > *"No domínio de Aprendizado por Reforço Profundo multi-agente e alocação de recursos, deparamo-nos com uma patologia teórica comum: agentes DRL alimentados apenas com séries temporais de preço e volatilidade sofrem de colapso de política, convergindo monotonicamente para a alocação uniforme 1/N. Isso é evidenciado pela entropia de Shannon máxima de H = 2.890, exatamente igual a ln(18). Ao injetarmos os embeddings de nós do DyFO no espaço de estados do PPO, os vetores topológicos atuam como uma regularização indutiva que quebra a simetria do espaço de ações (reduzindo H para 2.615). Como resultado prático, o DyFO-DRL+ gerou +1.72% de alpha acumulado com significância estatística (p=0.0312) e turnover 63% menor em relação ao benchmark EWMA-GMVP."*

---

### Slide 9: Robustez Sob Regimes de Estresse (Crash do COVID-19 em 2020)
- **Mensagem Chave:** Rastreamento contínuo em tempo real da rápida decorrelação do par SPY - ^VIX em março de 2020 sem lag temporal. Huber Loss e Time2Vec garantem estabilidade sob saltos extremos.
- **Fala Sugerida:**
  > *"Em sistemas críticos de IA, a robustez sob quebras estruturais de regime é mandatória. Analisamos o comportamento do modelo durante o crash de março de 2020, quando o índice VIX saltou de 15 para 82 pontos. O DyFO rastreou com precisão a decorrelação abrupta do par SPY versus VIX em tempo real, sem o atraso temporal de 10 a 20 dias característico de estimadores de janela móvel e sem instabilidade numérica de gradientes, graças à combinação da parametrização Time2Vec com a regularização da Huber Loss."*

---

### Slide 10: Integração na Tríade do Doutorado (PORTA, DyFO e ORION)
- **Mensagem Chave:** Arquitetura de software modular, contrato estritamente *read-only* com o PORTA e exportação padronizada em ontologia OWL/RDF (`<TICKER>.US`).
- **Fala Sugerida:**
  > *"O DyFO é o pilar de inteligência relacional e topológica de um ecossistema integrado de Doutorado. Ele consome dados curados em modo estritamente read-only do repositório PORTA, eliminando qualquer risco de vazamento; exporta snapshots estruturais causais para os modelos de alocação de risco do PORTA; e provê continuamente vetores de embedding relacional \(\mathbf{z}_t \in \mathbb{R}^{100}\) para o construtor de estados de percepção multimodal do agente ORION. Toda a representação de entidades segue uma ontologia semântica formal em OWL/RDF com predicados relacionais estritamente tipados."*

---

### Slide 11: Conclusões & Síntese das Contribuições em IA no BRACIS
- **Mensagem Chave:** Resumo das 3 contribuições principais para a comunidade de IA (resolução da diluição de atenção, regularização indutiva em DRL, rigor causal estrito) e reprodutibilidade aberta.
- **Fala Sugerida:**
  > *"Em síntese, apresentamos à comunidade do BRACIS três contribuições fundamentais: primeira, a resolução teórica e empírica da diluição de atenção em Dynamic Graph Neural Networks heterogêneas via condicionamento explícito de arestas; segunda, a demonstração de que embeddings topológicos dinâmicos funcionam como regularizadores indutivos que quebram a simetria em DRL de alta dimensionalidade; e terceira, um protocolo experimental rigoroso, causal e 100% reproduzível para aprendizado de máquina em streaming não-estacionário. Agradeço a atenção de todos e coloco-me à disposição para perguntas."*

---

---

## ⏱️ Guia de Gestão de Tempo (15 Minutos de Apresentação Oral)

| Bloco | Slides | Tópicos Centrais | Tempo Sugerido | Acumulado |
| :--- | :--- | :--- | :--- | :--- |
| **I. Motivação & Teoria** | 1, 2, 3, 4 | Título, Contexto de IA, Diluição de Atenção & TGAT v2 | 5 min 30 s | 5 min 30 s |
| **II. Metodologia & Escalaridade** | 5 | Três escalas combinatórias (\(N=18, 50, 100\)) & Causalidade | 1 min 30 s | 7 min 00 s |
| **III. Resultados Empíricos & Ablação** | 6, 7 | Benchmarks SOTA, Diebold-Mariano & Ganho Sinérgico (\(+0.0410\)) | 3 min 00 s | 10 min 00 s |
| **IV. Aplicação em DRL & Estresse** | 8, 9 | Quebra de Simetria (\(H=2.615\)) & Robustez COVID-19 | 2 min 30 s | 12 min 30 s |
| **V. Arquitetura & Conclusões** | 10, 11 | Tríade Doutoral, Ontologia Semântica & 3 Contribuições BRACIS | 2 min 30 s | 15 min 00 s |

---

## 🔬 Banco de Perguntas e Respostas Prováveis da Banca / Revisores (Q&A)

### Pergunta 1: Por que vocês não utilizaram o TGN (Temporal Graph Network) com memória recorrente GRU/LSTM?
- **Resposta Técnica:**
  > *"O TGN (Rossi et al., 2020) depende de um módulo de memória de nós baseado em GRU/RNN. Em séries temporais financeiras longas e não-estacionárias com milhares de passos temporais, memórias recorrentes sofrem de dois problemas graves: deriva de estado acumulado (state drift) e a necessidade de Backpropagation Through Time truncado (TBPTT), que introduz viés no aprendizado de longo prazo. O DyFO utiliza atenção puramente baseada em vizinhança temporal com Time2Vec, garantindo inferência sem estado latente recorrente (stateless recurrence), o que elimina o drift e assegura maior estabilidade sob choques de mercado."*

### Pergunta 2: Como vocês garantem que não existe *Look-Ahead Bias* na previsão do grafo?
- **Resposta Técnica:**
  > *"A causalidade estrita foi a diretriz central da nossa engenharia. O cálculo dos alvos supervisionados de correlação utiliza estritamente janelas causais passadas \([t-W, t]\) via Rolling Pearson causal. Na etapa de inferência no instante \(t\), o DyFO recebe apenas eventos ocorridos em \(t' \le t\) para prever a topologia de co-movimento em \(t+1\). Nosso pipeline foi auditado e validado com testes unitários automatizados específicos para invariância causal e isolamento temporal."*

### Pergunta 3: O que comprova matematicamente que os embeddings de grafo são responsáveis pelo ganho no DRL?
- **Resposta Técnica:**
  > *"A comprovação reside no diagnóstico da Entropia de Shannon da distribuição de pesos de alocação da política do PPO. O agente Raw-DRL (sem grafo) colapsa monotonicamente para \(H \approx 2.890\), que coincide exatamente com o limite teórico de máxima incerteza \(\ln(N) = \ln(18) = 2.89037\), caracterizando uma política uniforme ingênua \(1/N\). Ao adicionar os embeddings do DyFO, a entropia cai para \(H = 2.615\), demonstrando que o sinal topológico quebrou a simetria do espaço de ações, permitindo ao agente convergir para fronteiras eficientes de Markowitz dinâmicas com \(+1.72\%\) de alpha e \(63\%\) menor turnover."*

### Pergunta 4: Como o DyFO escala para grafos maiores com centenas de nós (\(N \ge 100\))?
- **Resposta Técnica:**
  > *"Em grafos completos, o número de arestas cresce quadraticamente com \(O(N^2)\). Para o universo de \(N=100\) (S&P 100) e \(N=104\) (PORTA), aplicamos sparsificação adaptativa por limiar (\(\tau = 0.30\)), que filtra arestas de correlação estatisticamente espúrias e reduz o grafo a uma estrutura esparsa. Em nossos experimentos, o modelo manteve \(R^2 = 0.865\) com tempo de inferência inferior a 12 milissegundos por snapshot temporal. Para o roadmap futuro, estamos implementando sparsificação topológica baseada em TMFG (Triangulated Maximally Filtered Graph) com complexidade linear em arestas \(O(3N-6)\)."*

### Pergunta 5: Por que adotar a Huber Loss (\(\delta=1.0\)) em vez do tradicional Mean Squared Error (MSE)?
- **Resposta Técnica:**
  > *"Séries financeiras e distribuições de correlação empíricas possuem caudas pesadas (leptocurtose) e saltos de regime (jumps). O MSE eleva ao quadrado os erros em regimes extremos (como em março de 2020), gerando explosão de normas de gradientes e instabilidade no treinamento do otimizador AdamW. A Huber Loss comporta-se quadraticamente para erros pequenos (\(|e| \le \delta\)) e linearmente para grandes desvios (\(|e| > \delta\)), limitando a magnitude máxima dos gradientes a \(\pm \delta\). Isso proporcionou convergência monotônica sem necessidade de gradient clipping excessivo."*

### Pergunta 6: Como foi configurado o teste estatístico de Diebold-Mariano para validar a significância preditiva?
- **Resposta Técnica:**
  > *"O teste de Diebold-Mariano (1995) foi aplicado comparando as séries temporais diárias da função de perda quadrática das previsões do DyFO contra o GAT-Static e o ROLAND ao longo de todas as janelas de teste out-of-sample. Para lidar com a autocorrelação e heteroscedasticidade inerentes a séries temporais em streaming contínuo, utilizamos o estimador de variância assintótica com correção espectral de Newey-West com defasagem ótima \(h = \lfloor 4(T/100)^{2/9} \rfloor\). A estatística \(t = -14.82\) rejeita a hipótese nula com \(p < 0.0001\)."*

