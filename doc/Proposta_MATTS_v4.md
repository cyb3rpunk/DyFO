**PROPOSTA DE TESE DE DOUTORADO**

**Multi-Agent Trading and Time-Series System**

**MATTS v4.0**

_Sistema Multi-Agente Hierárquico para Alocação de Portfólio com Detecção de Regimes, Grafos Temporais de Eventos Contínuos e Hierarquia de Políticas Stackelberg_

<div class="joplin-table-wrapper"><table><tbody><tr><th><p><strong>Resumo das Atualizações v3 → v4</strong></p><p><strong>Quatro atualizações de estado da arte identificadas e incorporadas:</strong></p><ul><li><strong>[V4-1] DyFO: ROLAND (snapshots discretos) → TGN</strong> - Temporal Graph Networks processam eventos financeiros assíncronos em tempo contínuo com memória por nó. Elimina artefato de discretização mensal.</li><li><strong>[V4-2] HARL: Kuba et al. ICLR 2022 (conferência) → Zhong et al. JMLR 2024</strong> - Versão definitiva com HAML completo, HATRPO, HAPPO, HAA2C, HADDPG, HATD3. Garantias Nash formalmente provadas. Referência primária atualizada.</li><li><strong>[V4-3] Orquestrador: HAPPO único → HAPPO + HASAC</strong> - HASAC (Liu et al., ICLR 2024 Spotlight) adiciona maximum entropy ao HARL para espaços contínuos. Coerência conceitual com M6: orquestrador maximiza entropia de política enquanto usa H(π_t) como feature de estado.</li><li><strong>[V4-4] Stackelberg: derivação teórica isolada → XP-MARL + derivação formal</strong> - XP-MARL (Xu et al., arXiv 2409.11852, 2024) é o prior empírico direto da estrutura bi-stage do orquestrador. Agentes de alta prioridade agem primeiro e comunicam ações - exatamente o design do MATTS. Reduz 84.4% colisões vs HAPPO sem priorização.</li><li><strong>Mantidos sem alteração:</strong> RDM (HMM-GAS-TVTP), CVaR reward, EWC + Curriculum, Alpha Signal Layers, protocolo experimental.</li></ul></th></tr></tbody></table></div>

Março 2026

# Registro de Mudanças v3.0 → v4.0

**Tabela 0 - Changelog MATTS v4.0:** todos os componentes avaliados contra literatura de 2024-2025. Itens marcados como ATUALIZADO ou NOVO receberam mudanças cirúrgicas; MANTIDO indica que o estado da arte vigente não supera a escolha v3.0.

| **Status**        | **Componente**                      | **v3.0 (anterior)**                                                            | **v4.0 (atualizado)**                                                                                                                              | **Justificativa**                                                                                                                                                                              |
| ----------------- | ----------------------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **ATUALIZADO v4** | **DyFO - Módulo 2**                 | ROLAND (You et al., KDD 2022) - snapshots discretos mensais                    | TGN - Temporal Graph Networks (Rossi et al., 2020; validado em finanças 2024) - tempo contínuo, memória por nó                                     | ROLAND opera em snapshots fixos. TGN processa eventos assíncronos (earnings, decisões Fed) naturalmente via memória por nó. Elimina artefato de discretização.                                 |
| **ATUALIZADO v4** | **Framework HARL - referência**     | Kuba et al., ICLR 2022 (paper de conferência, versão preliminar)               | Zhong, Kuba et al., JMLR 2024, v25:23-0488, pp. 1-67 - versão definitiva com HAML completo, HATRPO, HAPPO, HAA2C, HADDPG, HATD3                    | A versão de conferência é incompleta. A versão JMLR 2024 prova garantias Nash formalmente, inclui HAML completo e valida HATD3 como SOTA off-policy.                                           |
| **ATUALIZADO v4** | **Orquestrador - algoritmo**        | HAPPO (on-policy) único                                                        | HAPPO (primário) + HASAC (variante off-policy com entropy regularization). Seleção guiada por ablation V4-3                                        | HASAC - Liu et al., ICLR 2024 Spotlight - adiciona maximum entropy ao HARL, convergindo mais rápido em espaços contínuos (w_t ∈ \[0,1\]^K). Coerência conceitual com M6 (H(π_t) já no estado). |
| **NOVO v4**       | **XP-MARL - estrutura Stackelberg** | Stackelberg derivado teoricamente como extensão das provas HARL (P5, Módulo 4) | XP-MARL (Xu, Sobhy, Alrifaee, arXiv 2409.11852, 2024) como prior empírico da estrutura bi-stage. Prioridade do orquestrador aprendida, não fixada. | XP-MARL formaliza exatamente o que o MATTS faz: agentes de alta prioridade agem primeiro e comunicam ações. Reduz 84.4% colisões vs baseline HAPPO sem priorização. Valida P5 empiricamente.   |
| **MANTIDO v3**    | **RDM - Módulo 1**                  | HMM-GAS-TVTP (Hamilton 1989; Filardo 1994; Creal et al. 2013)                  | Sem alteração - HMM-GAS-TVTP permanece SOTA para detecção de regimes financeiros em contexto de portfólio                                          | Nenhum trabalho 2024-2025 supera a combinação HMM+GAS+TVTP para o caso específico de portfólio multi-asset com quantificação de incerteza. DDMS (2025) adicionado como benchmark RSL Q9.       |
| **MANTIDO v3**    | **Reward - Módulo 4**               | CVaR regime-condicionado (Rockafellar & Uryasev 2000; Almgren & Chriss 2001)   | Sem alteração                                                                                                                                      | Nenhuma métrica de risco coerente emergiu como superior ao CVaR no horizonte 2024-2025 para portfólio de médio prazo.                                                                          |
| **MANTIDO v3**    | **Continual RL - M3+M4**            | EWC (Kirkpatrick 2017) + Curriculum (Bengio 2009) + replay estratificado       | EWC permanece SOTA suficiente; adição de AEWC (Adaptive EWC) como contingência documentada                                                         | Papers 2024 de Continual RL propõem melhorias marginais sobre EWC. A escolha permanece defensável para a aplicação financeira específica.                                                      |
| **MANTIDO v3**    | **Alpha Signal Layers**             | K&S (2018) §3.1, §3.6, §3.9, §3.20, §4.2, §4.6                                 | Sem alteração - matemática de sinais clássicos é intemporal                                                                                        | Funções matemáticas de momentum, reversão e fatores não têm versão SOTA dependente de tempo.                                                                                                   |

# 1\. Introdução

O MATTS v4.0 representa a versão com estado da arte verificado de todas as escolhas algorítmicas do projeto. As três versões anteriores (v1.0-v3.0) construíram a arquitetura hierárquica progressivamente: v1.0 estabeleceu o FDAM e os agentes; v2.0 incorporou HAPPO, CVaR, Curriculum e EWC; v3.0 introduziu a arquitetura híbrida (Alpha Signal Layers) baseada em Kakushadze & Serur (2018). A v4.0 fecha a cadeia de atualizações com quatro modificações cirúrgicas derivadas de uma revisão sistemática da literatura de 2024-2025.

A motivação central permanece inalterada: portfólios financeiros exibem comportamento não-estacionário condicionado a regimes de mercado (bull, bear, alta volatilidade, lateral). Modelos estáticos falham sistematicamente em períodos de mudança de regime porque seus parâmetros são estimados sobre amostras mistas. A hipótese central do MATTS é que um sistema MARL hierárquico com detecção explícita de regimes, representação dinâmica de relações entre ativos e garantias formais de estabilidade produz portfólios mais robustos, explícáveis e reprodutíveis do que qualquer alternativa disponível.

A novidade introduzida na v4.0 é dupla. Primeiro, a substituição de ROLAND por TGN no Módulo 2 elimina o artefato de discretização temporal que ignorava eventos financeiros assíncronos entre snapshots. Segundo, a adição de XP-MARL como prior empírico da estrutura Stackelberg do orquestrador transforma o que era uma derivação teórica isolada (P5, Módulo 4) em uma contribuição ancorada em evidência empírica direta.

# 2\. Problema e Questões de Pesquisa

**Questão central:** Um sistema MARL hierárquico com (i) detecção de regimes HMM-GAS-TVTP, (ii) representação de grafo temporal de eventos contínuos (TGN), (iii) sub-agentes híbridos com Alpha Signal Layers e RL Adaptation Layers (HAPPO/HASAC), (iv) CVaR regime-condicionado como recompensa, (v) Curriculum Learning e EWC para continual learning, e (vi) estrutura Stackelberg bi-stage com prioridade aprendida (XP-MARL) produz alocações de portfólio superiores, mais estáveis e mais explícáveis do que os estados da arte disponíveis?

| **Q**   | **Questão**                                                                                                                                                                                           | **String**                 | **Mudança v4**                                                                                                                              |
| ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Q1**  | Quais modelos de regime são superiores ao HMM clássico para portfólios multi-asset, em termos de poder preditivo e estabilidade de transição?                                                         | S1 + S3                    | Mantida. DDMS (2025) adicionado como candidato alternativo em Q1.                                                                           |
| **Q2**  | Qual arquitetura híbrida (Alpha Signal Layer + RL Adaptation Layer) produz maior eficiência amostral e menor tempo de convergência em comparação com RL puro e alpha combination estático?            | S1 + S4                    | Mantida.                                                                                                                                    |
| **Q3**  | HAPPO e HASAC produzem políticas mais estáveis e com maior retorno ajustado ao risco do que MATD3, MASAC, MAPPO e MADDPG em ambientes de portfólio financeiro com regimes?                            | S1                         | **ATUALIZADA v4: inclui HASAC como co-algoritmo principal ao lado de HAPPO.**                                                               |
| **Q4**  | Quais são os avanços teóricos recentes em MARL cooperativo que oferecem garantias formais de melhoria monótona de retorno conjunto e convergência a equilíbrio de Nash?                               | S1 + S2                    | **ATUALIZADA v4: inclui HARL JMLR 2024 como referência primária do framework; HASAC como variante max-entropy.**                            |
| **Q5**  | Grafos temporais de eventos contínuos (TGN) superam representações em snapshots discretos (ROLAND, GAT estático) para capturar correlações dinâmicas entre ativos em tempo real?                      | S1 \[nova string TGN\]     | **NOVA v4: Q5 reformulada para comparar TGN vs. modelos de snapshot. Ablation B16 como evidência direta.**                                  |
| **Q6**  | Sob quais condições o sistema MATTS converge para um equilíbrio de Stackelberg, e como essa convergência se articula com as garantias Nash do HARL e a análise de Lyapunov do sistema acoplado?       | S2                         | **ATUALIZADA v4: inclui XP-MARL como prior empírico para a estrutura bi-stage; P5 do Módulo 4 derivada com referência a Xu et al. (2024).** |
| **Q7**  | A quantificação de incerteza epistêmica via entropia H(π_t) da política do orquestrador melhora o retorno ajustado ao risco em períodos de mudança de regime?                                         | S1 + S3                    | Mantida. Coerência adicional: HASAC maximiza entropia de política - conecta H(π_t) do estado com o objetivo de treinamento do orquestrador. |
| **Q8**  | O protocolo de validação walk-forward 60/20/20 com deflated Sharpe Ratio e 500 bootstrap é suficiente para eliminar look-ahead bias em benchmarking de RL financeiro?                                 | S1 + S4 (methodological)   | Mantida.                                                                                                                                    |
| **Q9**  | Quais modelos alternativos de regime (BOCPD, SETAR, DDMS) superam HMM-GAS-TVTP para detectar mudanças estruturais em mercados com baixa liquidez e microestrutura ruidosa?                            | S3                         | Mantida. DDMS adicionado explicitamente como candidato.                                                                                     |
| **Q10** | Quais técnicas de continual learning - EWC, Synaptic Intelligence, Progressive Networks, Adaptive EWC - melhor previnem catastrophic forgetting em agentes MARL treinados sequencialmente por regime? | S1 + (continual RL string) | Mantida. AEWC (Adaptive EWC) adicionado como opção de ablation.                                                                             |
| **Q11** | CVaR regime-condicionado produz portfólios mais robustos em cenários de tail risk do que CVaR estático, Sharpe e drawdown máximo como funções de recompensa?                                          | S1 + S3                    | Mantida.                                                                                                                                    |

# 3\. Objetivos Específicos

Os nove objetivos específicos do MATTS v4.0 incorporam as quatro atualizações de estado da arte:

- **O1:** Implementar e validar o Módulo 1 (RDM) com HMM-GAS-TVTP, produzindo π_t e H(π_t) com calibração estatística formal.
- **O2 \[v4\]:** Implementar o Módulo 2 (DyFO) com TGN sobre grafo financeiro de eventos contínuos, e demonstrar superioridade sobre ROLAND e GAT estático via ablation B16.
- **O3:** Implementar Alpha Signal Layers para os quatro sub-agentes com parâmetros canônicos da literatura (K&S 2018; Jegadeesh & Titman 1993; Fama & French 2015).
- **O4 \[v4\]:** Treinar orquestrador com HAPPO e HASAC (HARL JMLR 2024) e demonstrar via ablation B15 qual variante é superior para espaços de ação contínuos em portfólio financeiro.
- **O5 \[v4\]:** Integrar XP-MARL como mecanismo de priorização aprendida do orquestrador e validar convergência para equilíbrio de Stackelberg via proposição P5 do Módulo 4.
- **O6:** Derivar e demonstrar formalmente as proposições P1-P5 do Módulo 4: invariância de Lyapunov, acoplamento de pequeno ganho, monotonia de política, regime-condicionalidade do CVaR e equilíbrio de Stackelberg.
- **O7:** Validar o FDAM recursivo em dois níveis de abstração e provar invariância de escala dos cinco critérios de classificação.
- **O8:** Executar o protocolo experimental completo: 5 datasets, 16 baselines, walk-forward 60/20/20, 500 bootstrap, deflated Sharpe, Shapley values de sub-agentes.
- **O9:** Registrar MATTS v4.0 como sistema no INPI e publicar ≥ 3 papers em venues tier-1 (ICML, NeurIPS, ICLR ou Q1 de finanças quantitativas).

# 4\. Fundamentação Teórica

## 4.1 Alpha Combination Clássica - Baseline Direto do Orquestrador

Kakushadze & Serur (2018) \[30\] catalogam 151 estratégias com mais de 550 fórmulas matemáticas. A seção §3.20 descreve Alpha Combination com pesos OLS - o análogo estático exato do que o orquestrador HAPPO/HASAC faz dinamicamente. Os §3.1, §3.9, §3.6, §4.2 e §4.6 fornecem as equações dos Alpha Signal Layers de cada sub-agente. Nenhuma das 151 estratégias usa RL, MARL, detecção de regime ou GNN - confirmando a lacuna que o MATTS endereça.

## 4.2 Arquitetura Híbrida - Alpha Signal + RL Adaptation

Cada sub-agente do MATTS é composto por dois layers: (i) Alpha Signal Layer - função matemática fechada derivada de K&S, sem parâmetros aprendidos, output determinístico; (ii) RL Adaptation Layer - política HAPPO ou HASAC que aprende quando e quanto confiar no sinal do layer anterior, condicionada no regime π_t. Essa arquitetura resolve o problema de eficiência amostral (RL não precisa redescobrir momentum do zero), a explainability (o sinal clássico é auditável), e o warm-start (Curriculum começa de sub-agentes funcionais, não de políticas aleatórias).

## 4.3 HARL - Framework Definitivo (JMLR 2024)

**\[V4-2\]** Zhong, Kuba et al. (2024) \[17\] publicam no JMLR a versão definitiva do Heterogeneous-Agent Reinforcement Learning (HARL). O paper prova o Fundamental Theorem of Heterogeneous-Agent Mirror Learning (HAML): todos os algoritmos derivados do HAML gozam de melhoria monótona de retorno conjunto e convergência ao Nash Equilibrium. O framework inclui HATRPO, HAPPO, HAA2C, HADDPG, HATD3 e seus respectivos esquemas sequenciais de atualização. A substituição da referência de conferência (Kuba 2022) pela versão JMLR 2024 é obrigatória - o paper de conferência é um subset incompleto das garantias formais.

## 4.4 HASAC - Maximum Entropy HARL (ICLR 2024)

**\[V4-3\]** Liu et al. (2024) \[18\] (ICLR 2024 Spotlight) extendem o HARL com maximum entropy regularization, produzindo HASAC - a versão SAC do framework HARL. Para espaços de ação contínuos como w_t ∈ \[0,1\]^K, HASAC combina as garantias de convergência Nash do HAML com a convergência mais rápida e robustez à exploração do SAC. A coerência conceitual com o MATTS é imediata: o estado já inclui H(π_t) como medida de incerteza epistêmica; HASAC torna o orquestrador explicitamente comprometido com maximização de entropia de política, unificando os objetivos de percepção (RDM) e decisão (orquestrador).

## 4.5 XP-MARL e a Estrutura Stackelberg

**\[V4-4\]** A conversa de diagnóstico anterior identificou que a relação orquestrador-subagentes no MATTS é de Stackelberg (líder-seguidor), não Nash simultâneo. Xu, Sobhy & Alrifaee (2024) \[19\] propõem XP-MARL, que formaliza exatamente esta estrutura: agentes de alta prioridade agem primeiro e comunicam suas ações aos de baixa prioridade via action propagation. Crucialmente, as prioridades são aprendidas via um problema MARL auxiliar - não fixadas a priori. No MATTS, o orquestrador é o agente de alta prioridade (decide w_t antes dos sub-agentes); os sub-agentes são seguidores que condicionam suas políticas em w_t. XP-MARL demonstra que esta estrutura reduz não-estacionariedade em 84.4% vs. baseline HAPPO sem priorização. Isso transforma P5 (Módulo 4) de uma derivação puramente teórica em uma proposição com prior empírico direto.

## 4.6 TGN - Temporal Graph Networks

**\[V4-1\]** Rossi et al. (2020) \[7\] propõem os Temporal Graph Networks (TGN) para aprendizado em grafos dinâmicos de eventos contínuos. Diferentemente de ROLAND, que opera em snapshots mensais fixos, o TGN mantém um módulo de memória por nó que é atualizado cada vez que um evento ocorre envolvendo aquele nó. Para o DyFO, isto significa que earnings reports, decisões de política monetária, rebaixamentos de crédito e outros eventos financeiros assíncronos atualizam imediatamente o embedding do ativo afetado - sem esperar o próximo snapshot mensal. A ablation B16 (TGN vs. ROLAND vs. GAT estático) quantificará empiricamente este ganho.

## 4.7 Detecção de Regimes - HMM-GAS-TVTP

Hamilton (1989) \[1\] define o HMM financeiro. Filardo (1994) \[2\] introduz TVTP. Creal, Koopman & Lucas (2013) \[3\] definem GAS. A combinação HMM-GAS-TVTP permanece SOTA para o caso específico de portfólio multi-asset com quantificação de incerteza epistêmica (H(π_t)). Nenhum paper 2024-2025 supera esta combinação para a aplicação específica do MATTS.

## 4.8 CVaR Regime-Condicionado

Rockafellar & Uryasev (2000) \[26\] definem CVaR e provam sua otimizabilidade via programação linear. A formulação do reward r*t = CVaR*α(R_t | regime_t) − δ(k_t) é derivada diretamente deste paper, com penalidade de custo de transação do modelo de Almgren & Chriss (2001) \[27\]. Sem alteração na v4.0.

## 4.9 Continual RL - EWC e Curriculum

Kirkpatrick et al. (2017) \[24\] definem EWC via penalidade de Fisher. Bengio et al. (2009) \[25\] definem Curriculum Learning. A combinação EWC + Curriculum 4-estágios + replay estratificado por regime permanece a escolha mais defensável para o problema específico do MATTS. AEWC (Adaptive EWC) é documentado como contingência se EWC isolado for insuficiente.

## 4.10 FDAM Recursivo e Invariância de Escala

O FDAM (Framework de Distinção Agente-Módulo) é aplicado em dois níveis: L1 (sistema completo) e L2 (interior de cada sub-agente). A invariância de escala - os cinco critérios classificam corretamente em ambos os níveis - é a proposição teórica que eleva o FDAM de framework pragmático a resultado formal geral. Na v4.0, o DyFO-TGN é classificado como MODULE em L1 por satisfazer os cinco critérios: entrada bem definida (grafo G_t), saída determinística dada entrada (embedding e_t), sem política aprendida em loop de recompensa, sem interação direta com ambiente financeiro, stateless entre episódios.

# 5\. Metodologia - Arquitetura MATTS v4.0

A Tabela 1 apresenta a arquitetura completa do MATTS v4.0 com todos os componentes, seus tipos FDAM, algoritmos/modelos e referências de estado da arte. Os componentes atualizados na v4.0 são marcados em verde.

| **Componente**          | **Tipo**           | **Algoritmo/Modelo**                                 | **Referência SOTA**                                           | **Papel no sistema**                                                                         |
| ----------------------- | ------------------ | ---------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **RDM**                 | Módulo (M1)        | HMM + TVTP + GAS                                     | Hamilton (1989); Filardo (1994); Creal et al. (2013) \[JMLR\] | Produz π_t (prob. de regime) e H(π_t) (entropia epistêmica)                                  |
| **DyFO \[v4\]**         | Módulo (M2)        | TGN - Temporal Graph Networks                        | Rossi et al. (2020); validação fin. 2024                      | Produz embedding e_t de relações dinâmicas entre ativos em tempo contínuo                    |
| **State Constructor**   | Módulo (M3)        | s_t = \[e_t \| π_t \| H(π_t) \| α_signals_t \| x_t\] | -                                                             | Agrega saídas de todos os módulos em estado unificado para os agentes                        |
| **Reward Constructor**  | Módulo (M4)        | CVaR regime-condicionado − δ(k_t)                    | Rockafellar & Uryasev (2000); Almgren & Chriss (2001)         | Calcula recompensa r_t com penalidade de custo de transação                                  |
| **Portfolio Executor**  | Módulo (M5)        | Execução com modelo de impacto de mercado            | Almgren & Chriss (2001)                                       | Converte pesos w_t em ordens respeitando liquidez e custo                                    |
| **Orquestrador \[v4\]** | Agente (primário)  | HAPPO + HASAC (ablation V4-3)                        | Zhong et al. JMLR 2024; Liu et al. ICLR 2024                  | Aprende w_t: distribuição de recursos entre sub-agentes por regime. Prioritário via XP-MARL. |
| **SA-Trend**            | Sub-Agente híbrido | Alpha: mom. K&S §3.1 + RL: HAPPO/HASAC               | Kakushadze & Serur (2018); Jegadeesh & Titman (1993)          | Sinal de momentum; RL ajusta lookback S\*, threshold e sizing por regime                     |
| **SA-MeanRev**          | Sub-Agente híbrido | Alpha: mean-rev K&S §3.9 + RL: HAPPO/HASAC           | Kakushadze & Serur (2018)                                     | Sinal de reversão à média; RL ajusta janela de cluster e z-score por regime                  |
| **SA-Risk**             | Sub-Agente híbrido | Alpha: low-vol K&S §3.4 + RL: HAPPO/HASAC            | Kakushadze & Serur (2018); Frazzini & Pedersen (2014)         | Sinal de risco (σ_i, β_i, drawdown); RL ajusta limites de posição por regime de volatilidade |
| **SA-Macro**            | Sub-Agente híbrido | Alpha: fatores K&S §4.2 + RL: HAPPO/HASAC            | Kakushadze & Serur (2018); Fama & French (2015)               | Sinal macro-fator; RL ajusta exposição e timing de rotação por ciclo econômico               |

### 5.1 Módulo 1 - RDM (Regime Detection Module)

O RDM implementa um HMM com K regimes latentes (K ∈ {2,3,4} determinado por BIC), probabilidades de transição variantes no tempo (TVTP condicionado em variáveis macroeconômicas) e parâmetros condicionais com dinâmica score-driven (GAS). A saída é o vetor π*t = (p(s_t=1|F_t), ..., p(s_t=K|F_t)) e a entropia epistêmica H(π_t) = −Σ_k π*{t,k} log π\_{t,k}. Ambos entram no estado do orquestrador e dos sub-agentes.

### 5.2 Módulo 2 \[v4\] - DyFO com TGN (Dynamic Financial Ontology)

O DyFO v4.0 substitui ROLAND por TGN. O grafo G = (V, E) representa ativos como nós e relações (correlação, setor, cadeia de fornecimento, co-movimento de fatores) como arestas. Cada nó v_i mantém um estado de memória m_i(t) atualizado via módulo de memória do TGN a cada evento financeiro e_i(t) que envolve o ativo i. A mensagem enviada após cada evento é computada por um módulo de mensagem diferenciável, permitindo backpropagation através da história de eventos. O embedding e_t do grafo completo é computado por uma camada GAT sobre os estados de memória atuais, preservando a atenção por aresta interpretável.

### 5.3 Orquestrador \[v4\] - HAPPO e HASAC

O orquestrador usa HARL JMLR 2024 \[17\] como framework. O algoritmo primário é HAPPO (on-policy, garantias Nash formais via HAML). O ablation V4-3 compara com HASAC \[18\] (off-policy, maximum entropy). A seleção final do algoritmo para cada dataset é guiada pelo ablation B15. O mecanismo de priorização segue XP-MARL \[19\]: o orquestrador é o agente de alta prioridade no problema bi-stage; sua ação w_t é comunicada aos sub-agentes antes que eles atuem, satisfazendo a estrutura Stackelberg formalmente.

### 5.4 Sub-Agentes Híbridos

Cada sub-agente implementa dois layers. O Alpha Signal Layer computa um score unidimensional via equação fechada de K&S, com parâmetros warm-started nos valores canônicos da literatura. O RL Adaptation Layer (HAPPO/HASAC, conforme ablation B15) aprende uma política condicional no regime π_t e no score do Alpha Signal Layer, ajustando os hiperparâmetros operacionais do sinal (lookback, threshold, sizing) por regime.

### 5.5 Estado Aumentado

O estado completo é s_t = \[e_t | π_t | H(π_t) | α_signals_t | x_t\], onde e_t ∈ ℝ^d é o embedding TGN do grafo, π_t ∈ Δ^K as probabilidades de regime, H(π_t) ∈ ℝ a entropia epistêmica, α_signals_t ∈ ℝ^4 os scores dos Alpha Signal Layers dos quatro sub-agentes, e x_t os features financeiros padrão (retornos, volatilidades, volumes).

# 6\. FDAM Recursivo - Dois Níveis de Abstração

A Tabela 2 apresenta o FDAM completo do MATTS v4.0 com os 14 componentes classificados em dois níveis de abstração. O DyFO-TGN \[v4\] é classificado como MODULE em L1. A invariância de escala é verificável: todos os MODULEs satisfazem os critérios (i)-(v) em ambos os níveis; todos os AGENTs satisfazem os critérios de agente em ambos os níveis.

| **Componente**                      | **Nível** | **Tipo**   | **Estado**                    | **Ação**                   | **Critérios FDAM satisfeitos**                                                                                                                                        |
| ----------------------------------- | --------- | ---------- | ----------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RDM**                             | **L1**    | **MODULE** | Dados mercado                 | π_t, H(π_t)                | (i) entrada bem definida; (ii) saída determinística dada entrada; (iii) sem política aprendida; (iv) sem interação direta com ambiente; (v) stateless entre episódios |
| **DyFO - TGN \[v4\]**               | **L1**    | **MODULE** | Grafo G_t (eventos contínuos) | e_t (embedding)            | (i)-(v) idênticos ao RDM. TGN opera como função de transformação diferenciável do grafo dinâmico                                                                      |
| **State Constructor**               | **L1**    | **MODULE** | π_t, e_t, H(π_t), α, x_t      | s_t                        | (i)-(v): concatenação determinística, sem política                                                                                                                    |
| **Reward Constructor**              | **L1**    | **MODULE** | w_t, retornos, custos         | r_t                        | (i)-(v): cálculo CVaR determinístico dado w_t e retornos realizados                                                                                                   |
| **Portfolio Executor**              | **L1**    | **MODULE** | w_t                           | ordens k_t                 | (i)-(v): modelo de impacto de mercado determinístico                                                                                                                  |
| **Orquestrador \[v4 HAPPO/HASAC\]** | **L1**    | **AGENT**  | s_t                           | w_t ∈ \[0,1\]^K            | (i) observa s_t completo; (ii) aprende π_orch(w_t\|s_t); (iii) maximiza J(θ_orch) = E\[Σ r_t\]; (iv) recebe r_t do ambiente; (v) política persiste entre episódios    |
| **Alpha Signal Layer - SA-Trend**   | **L2**    | **MODULE** | Retornos rolling              | Momentum score             | (i)-(v) FDAM nível 2: função fechada K&S §3.1, sem política, output determinístico dado input                                                                         |
| **RL Adapt Layer - SA-Trend**       | **L2**    | **AGENT**  | s_t^SA, momentum score        | a_t^SA (S\*, θ_th)         | (i)-(v) FDAM nível 2: aprende quando/quanto confiar no sinal por regime                                                                                               |
| **Alpha Signal Layer - SA-MeanRev** | **L2**    | **MODULE** | Retornos cross-section        | Mean-rev score R̃_i         | (i)-(v) FDAM nível 2: K&S §3.9, função fechada                                                                                                                        |
| **RL Adapt Layer - SA-MeanRev**     | **L2**    | **AGENT**  | s_t^SA, mean-rev score        | a_t^SA (janela, z)         | (i)-(v) FDAM nível 2                                                                                                                                                  |
| **Alpha Signal Layer - SA-Risk**    | **L2**    | **MODULE** | σ_i, β_i, drawdown            | Risk score                 | (i)-(v) FDAM nível 2: K&S §3.4                                                                                                                                        |
| **RL Adapt Layer - SA-Risk**        | **L2**    | **AGENT**  | s_t^SA, risk score            | a_t^SA (limites pos.)      | (i)-(v) FDAM nível 2                                                                                                                                                  |
| **Alpha Signal Layer - SA-Macro**   | **L2**    | **MODULE** | Fatores FF5                   | Factor score α_i           | (i)-(v) FDAM nível 2: K&S §4.2, eq. 364                                                                                                                               |
| **RL Adapt Layer - SA-Macro**       | **L2**    | **AGENT**  | s_t^SA, factor score          | a_t^SA (exposição, timing) | (i)-(v) FDAM nível 2                                                                                                                                                  |

**Proposição de Invariância de Escala (FDAM):** _Os cinco critérios de classificação Módulo/Agente do FDAM produzem o mesmo resultado independentemente do nível de abstração em que são aplicados. Formalmente: ∀ c ∈ C, FDAM(c, L1) = FDAM(c, L2) onde C é o conjunto de componentes e L1, L2 são os dois níveis de abstração._

# 7\. Módulo 4 - Análise Teórica e Proposições P1-P5

O Módulo 4 deriva cinco proposições formais sobre o comportamento do sistema MATTS. As proposições P1-P4 são derivadas usando teoria de Lyapunov (Khalil 2002; Berkenkamp et al. 2017; NeurIPS 2025). A proposição P5 é nova na v4.0 e usa XP-MARL como prior empírico.

### P1 - Invariância de Lyapunov do RDM

Existe uma função de Lyapunov V*RDM(π_t) tal que V_RDM(π_t) ≤ V_RDM(π*{t-1}) em expectativa, garantindo que os parâmetros HMM-GAS permanecem na região de operação estável após perturbações de regime. Derivada via teoria de Lyapunov estocástica para sistemas GAS.

### P2 - Acoplamento de Pequeno Ganho (RDM → Orquestrador)

O acoplamento entre a saída do RDM (π_t) e a política do orquestrador satisfaz o teorema do pequeno ganho: ||G_RDM|| · ||G_orch|| < 1, garantindo que erros de classificação de regime do RDM não se amplificam na política do orquestrador. Derivada via teoria de controle de input-output (Vidyasagar 2002).

### P3 - Monotonia de Política (HAML)

Cada iteração de atualização sequencial do HARL estritamente não piora o retorno conjunto J(π). Formalmente: J(π^{k+1}) ≥ J(π^k). Esta proposição é herdada diretamente do Fundamental Theorem of HAML (Zhong et al. JMLR 2024 \[17\]), Theorem 14.

### P4 - CVaR Regime-Condicionado é Coerente

A função de recompensa r*t = CVaR*α(R_t | regime_t) − δ(k_t) é uma medida coerente de risco para cada regime fixo, satisfazendo sub-aditividade, homogeneidade, monotonicidade e invariância de translação (Artzner et al. 1999).

### P5 \[v4\] - Equilíbrio de Stackelberg com Prioridade Aprendida

**\[V4-4 - nova em v4.0\]** O sistema MATTS converge para um equilíbrio de Stackelberg (S\*, w\*\_t, a\*\_t) em que: (a) o orquestrador é o líder que maximiza J*orch antecipando a resposta ótima dos sub-agentes dado w_t; (b) cada sub-agente é um seguidor em Nash equilibrium dado w_t fixo (garantido por P3 via HAML); (c) as prioridades de atuação são aprendidas via XP-MARL e convergem para uma atribuição estável. A prova usa a estrutura bi-stage do XP-MARL (Xu et al. 2024 \[19\]) como esqueleto e as garantias Nash do HAML (Zhong et al. 2024 \[17\]) para a parte do seguidor. Formalmente: no equilíbrio de Stackelberg, o orquestrador resolve argmax*{w_t} J_orch(w_t, BR(w_t)), onde BR(w_t) é a melhor resposta conjunta dos sub-agentes dado w_t, garantida por HAML em Nash.

# 8\. Baselines e Plano Experimental

O MATTS v4.0 é comparado contra 16 baselines. Os baselines B15 e B16 são novos na v4.0 e isolam especificamente as contribuições dos upgrades de estado da arte.

| **#**          | **Baseline**                                      | **Algoritmo**                                       | **Referência**                        | **O que isola**                                                                          |
| -------------- | ------------------------------------------------- | --------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------------------- |
| **B1**         | Buy-and-Hold                                      | Portfolio igual-ponderado, rebalanceamento anual    | -                                     | Custo de oportunidade mínimo                                                             |
| **B2**         | Markowitz MVO                                     | Otimização média-variância com covariância amostral | Markowitz (1952)                      | Modelo estático sem regime                                                               |
| **B3**         | Ledoit-Wolf Shrinkage                             | MVO com matriz de covariância shrinkage             | Ledoit & Wolf (2004)                  | Robustez de estimação vs. MATTS                                                          |
| **B4**         | Risk Parity                                       | Pesos inversamente proporcionais à volatilidade     | Roncalli (2017)                       | Diversificação sem RL                                                                    |
| **B5**         | **Alpha Combos estático (K&S §3.20)**             | OLS weights fixos, sem adaptação de regime          | Kakushadze & Serur (2018)             | ★ Isolamento direto da contribuição do orquestrador HAPPO                                |
| **B6**         | Multifactor estático (K&S §3.6)                   | Value+Momentum+Vol com pesos fixos                  | Kakushadze & Serur (2018)             | Alpha Signal Layers sem RL Adaptation                                                    |
| **B7**         | DRL single-agent (PPO)                            | PPO single agent sobre portfólio completo           | Schulman et al. (2017)                | Ganho de MARL sobre DRL monolítico                                                       |
| **B8**         | MADDPG                                            | CTDE com MADDPG                                     | Lowe et al. (2017)                    | HAPPO vs. MADDPG (sem garantias formais)                                                 |
| **B9**         | MAPPO                                             | MAPPO homogêneo (parameter sharing)                 | Yu et al. (2022)                      | Heterogeneidade HARL vs. homogeneidade MAPPO                                             |
| **B10**        | GPM - GNN+RL estático                             | GAT estático + RL (baseline GNN sem regime)         | Shi et al. (2022)                     | Contribuição do TGN dinâmico vs. GAT estático                                            |
| **B11**        | MSPM modular                                      | Arquitetura modular sem regime detection            | Huang & Tanaka (2022)                 | Contribuição do RDM sobre sistema modular sem regimes                                    |
| **B12**        | **GAT-MARL (SOTA)**                               | GNN+MARL sem hierarquia explícita                   | Chen et al. (2025)                    | Contribuição de FDAM hierárquico + regimes vs. SOTA plano                                |
| **B13**        | Ablation: MATTS RL-puro (sem Alpha Signal Layers) | Apenas RL Adaptation Layer, sem sinal clássico      | HAPPO/HASAC puro                      | Isolamento da contribuição do Alpha Signal Layer (eficiência amostral)                   |
| **B14**        | Ablation: MATTS Alpha fixo (sem RL Adaptation)    | Alpha Signal Layers com pesos fixos, sem HAPPO      | K&S §3.20 + §3.6 combinados           | Isolamento da contribuição do RL Adaptation Layer (adaptação de regime)                  |
| **B15 \[v4\]** | Ablation: HAPPO vs. HASAC                         | MATTS completo com HAPPO vs. HASAC no orquestrador  | Zhong JMLR 2024; Liu ICLR 2024        | Isolamento: on-policy (HAPPO) vs. off-policy max entropy (HASAC) para ação contínua w_t  |
| **B16 \[v4\]** | Ablation: TGN vs. GAT estático                    | MATTS com TGN no DyFO vs. GAT estático (ROLAND)     | Rossi et al. (2020) vs. ROLAND (2022) | Isolamento: grafos contínuos (TGN) vs. snapshots discretos (ROLAND) sobre mesmo pipeline |

### 8.1 Protocolo de Validação

Walk-forward 60/20/20 (treino/validação/teste) em janelas rolantes de 252 dias úteis. Cinco datasets: S&P 500 constituintes (2000-2024), MSCI World ETFs, commodities futures, cripto top-20 por capitalização, Fama-French 5 fatores. Métricas: Deflated Sharpe Ratio (López de Prado 2018 \[28\]), CVaR-95%, max drawdown, F1 de classificação de regime, Shapley values dos sub-agentes. 500 simulações bootstrap por configuração.

# 9\. Hipóteses e Alvos Quantitativos

A Tabela 4 apresenta as 10 hipóteses testáveis do MATTS v4.0. As hipóteses H3, H4, H9 e H10 são novas na v4.0.

| **#**         | **Hipótese**                                                                                                | **Métrica**                                                                        | **Limiar de aceitação**                                                                           |
| ------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **H1**        | MATTS v4 supera Buy-and-Hold, Markowitz e Risk Parity em Deflated Sharpe                                    | Deflated SR (Módulo 5)                                                             | Deflated SR(MATTS) > Deflated SR(melhor baseline) em ≥ 75% das janelas walk-forward               |
| **H2**        | HAPPO e HASAC superam MAPPO, MADDPG e PPO single-agent                                                      | Retorno cumulativo; CVaR-95%; max drawdown                                         | HAPPO/HASAC ≥ MAPPO em retorno em ≥ 80% dos cenários; CVaR ≥ 5pp melhor                           |
| **H3 \[v4\]** | HASAC converge mais rápido que HAPPO em espaços de ação contínuos (w_t ∈ \[0,1\]^K)                         | Episódios até convergência; variance da política                                   | HASAC atinge 95% do retorno de HAPPO em ≤ 70% dos episódios                                       |
| **H4 \[v4\]** | TGN supera ROLAND e GAT estático na qualidade do embedding de regime                                        | R² de predição de regime; Sharpe condicional por regime                            | Ablation B16: Sharpe condicional com TGN ≥ Sharpe com ROLAND em ≥ 70% das janelas                 |
| **H5**        | Arquitetura híbrida supera K&S Alpha Combos estático (B5)                                                   | Deflated SR; CVaR; drawdown                                                        | Deflated SR(MATTS) ≥ Deflated SR(B5) em ≥ 80% das janelas walk-forward                            |
| **H6**        | RL Adaptation Layer isolado contribui positivamente sobre Alpha Signal fixo                                 | Comparação B14 vs. MATTS completo                                                  | Deflated SR(MATTS completo) > Deflated SR(B14) em ≥ 70% das janelas                               |
| **H7**        | MATTS completo supera MATTS RL-puro em eficiência amostral                                                  | Episódios até 90% do retorno de convergência                                       | Ablation B13: MATTS completo converge ≥ 30% mais rápido que RL-puro                               |
| **H8**        | EWC previne catastrophic forgetting em mudanças de regime simuladas                                         | Retorno médio pós-mudança; F1 de regime                                            | MATTS com EWC perde ≤ 15% do Sharpe pré-mudança; baseline sem EWC perde ≥ 40%                     |
| **H9 \[v4\]** | XP-MARL com prioridade aprendida supera hierarquia com prioridade fixa                                      | Estabilidade de treinamento (variância de retorno); episódios até convergência     | Variância de retorno com prioridade aprendida ≤ 50% da variância com prioridade fixa              |
| **H10**       | Equilíbrio de Stackelberg é atingido - orquestrador converge para política estável dado sub-agentes em Nash | Norma do gradiente do orquestrador; estacionariedade das políticas dos sub-agentes | \|∇J_orch\| < ε em ≥ 95% dos timesteps após convergência; sub-agentes em Nash verificado via HARL |

# 10\. Contribuições Científicas

## Contribuições Teóricas

- **C1 - Proposições P1-P5 do Módulo 4:** primeira derivação formal de estabilidade Lyapunov + equilíbrio de Stackelberg para sistema MARL hierárquico financeiro. P5 é ancorada em prior empírico XP-MARL.
- **C2 - FDAM recursivo com invariância de escala:** prova de que os cinco critérios classificam corretamente em dois níveis de abstração independentes.
- **C3 - Comparação HAPPO vs. HASAC em portfólio financeiro com espaço contínuo:** primeiro benchmark sistemático das duas variantes do HARL em contexto financeiro real (ablation B15).
- **C4 - TGN vs. modelos de snapshot para correlações dinâmicas de ativos:** primeiro benchmark de grafos temporais de eventos contínuos vs. snapshots discretos em portfólio quantitativo (ablation B16).
- **C5 - Arquitetura híbrida Alpha Signal + HARL:** primeiro sistema que combina sinais clássicos provados (K&S 2018) com HARL de última geração, com ablation isolando contribuição de cada layer.

## Contribuições Tecnológicas

- **C6 - Sistema MATTS v4.0:** implementação open-source em Ray/RLlib com TGN, HAPPO, HASAC, EWC, Curriculum e 16 baselines integrados.
- **C7 - Registro INPI:** registro de software MATTS v4.0 como propriedade intelectual universitária.

# 11\. Cronograma - 48 Meses

| **Período** | **Fase**        | **Atividades**                                                                                                                    | **Entregável**                                                                         | **Mudança v4**                                                           |
| ----------- | --------------- | --------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **M1-4**    | Setup           | RDM (HMM+GAS+TVTP), ambiente FinRL-Meta, Alpha Signal Layers SA-Trend+MeanRev                                                     | RDM funcional com saída π_t e H(π_t); Alpha Signal Layers testados em dados históricos | Mantida                                                                  |
| **M3-8**    | **DyFO \[v4\]** | Implementação TGN sobre grafo financeiro dinâmico. Comparação TGN vs. ROLAND vs. GAT (Ablation B16)                               | DyFO com TGN produzindo e_t; tabela ablation B16                                       | **NOVA: TGN substitui ROLAND; ablation B16 adicionada**                  |
| **M5-12**   | **Agentes**     | Implementar HAPPO e HASAC (HARL JMLR 2024). Validar convergência em ambiente controlado. Ablation B15 (HAPPO vs. HASAC)           | 2 agentes cooperativos convergindo; ablation B15                                       | **NOVA: HASAC adicionado; ablation B15**                                 |
| **M8-16**   | **Hierarquia**  | Orquestrador com XP-MARL (prioridade aprendida). Integrar sub-agentes híbridos completos. Testar Stackelberg em ambiente simulado | Hierarquia funcional. Evidência preliminar de P5                                       | **NOVA: XP-MARL integrado; P5 derivada com referência a Xu et al. 2024** |
| **M12-20**  | Continual RL    | EWC + Curriculum 4 estágios + replay estratificado. Simular mudanças de regime abruptas (crise, bull, bear, lateral)              | Agentes com EWC; comparação B13+B14; hipótese H8                                       | Mantida                                                                  |
| **M16-28**  | MATTS completo  | Integrar todos os 5 módulos + 5 agentes. Walk-forward 60/20/20 em 3 datasets. 16 baselines completos                              | Sistema MATTS v4.0 integrado; tabelas experimentais preliminares                       | NOVO: 16 baselines (B15+B16 adicionados)                                 |
| **M24-34**  | **Teoria**      | Provas P1-P5 do Módulo 4 em LaTeX. P5 (Stackelberg) com referência a XP-MARL como prior empírico. FDAM recursivo formalizado      | Capítulo teórico completo; proposições P1-P5 revisadas pelo orientador                 | **NOVA: P5 fundamentada em XP-MARL**                                     |
| **M30-40**  | Validação       | 500 bootstrap; deflated SR; Shapley values de sub-agentes; ablations B13-B16 completos                                            | Paper experimental submetido; resultados de todos os H1-H10                            | NOVO: H3, H4, H9, H10 adicionados                                        |
| **M38-48**  | Escrita         | Tese completa; repositório reprodutível; INPI registro MATTS v4.0                                                                 | Tese defendida; 3 papers publicados/submetidos                                         | -                                                                        |

# 12\. Limitações Documentadas

### L1 - Inductive Bias dos Alpha Signal Layers

O Alpha Signal Layer de cada sub-agente assume que momentum, reversão à média, fatores macro e baixa volatilidade permanecem como fontes de alpha. O RL Adaptation Layer não pode descobrir estratégias radicalmente diferentes fora desses paradigmas. Esta é uma limitação consciente e documentada - não uma falha: o MATTS é um sistema de adaptação de sinais provados, não de descoberta de sinais novos. Contingência: o ablation B13 (RL puro) serve como alternativa se os sinais clássicos se tornarem sistemicamente não-preditivos.

### L2 - TGN em Dados de Alta Frequência

O TGN é validado para eventos assíncronos, mas a granularidade do DyFO permanece diária. Dados intradiários (LOB, tick-by-tick) requerem infraestrutura adicional não prevista no escopo do doutorado. Documentado como R&D industrial pós-PhD.

### L3 - Lacunas Comerciais

Seis lacunas comerciais permanecem fora do escopo: (i) dados de tick reais, (ii) simulação realista de LOB, (iii) infraestrutura de produção de baixa latência, (iv) compliance regulatório automatizado, (v) análise de capacidade de estratégia, (vi) detecção de regime OOD em tempo real. Todas documentadas como trabalho futuro de P&D industrial pós-defesa.

# 13\. Infraestrutura Computacional

4× NVIDIA A100 80GB (treinamento HAPPO/HASAC paralelo com Ray/RLlib), 1TB NVMe (datasets históricos + checkpoints), 64-core CPU (simulações bootstrap e processamento de eventos TGN). Stack: Python 3.11, PyTorch 2.x, Ray 2.x/RLlib, HARL (github.com/PKU-MARL/HARL), PyG (PyTorch Geometric para TGN), DVC para versionamento de experimentos, MLflow para tracking. Reprodutibilidade garantida via containers Docker + seeds fixas + DVC pipeline.

# 14\. Referências

As referências marcadas como \[v4\] são adições ou atualizações da v4.0 em relação à v3.0.

| **#**             | **Referência completa**                                                                                                                                                                                  | **Componente MATTS v4**                                                             |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **\[1\]**         | Hamilton, J.D. (1989). A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle. Econometrica, 57(2), 357-384.                                                        | RDM - HMM fundamento                                                                |
| **\[2\]**         | Filardo, A.J. (1994). Business-Cycle Phases and Their Transitional Dynamics. Journal of Business & Economic Statistics, 12(3), 299-308.                                                                  | RDM - TVTP                                                                          |
| **\[3\]**         | Creal, D., Koopman, S.J., & Lucas, A. (2013). Generalized Autoregressive Score Models with Applications. Journal of Applied Econometrics, 28(5), 777-795.                                                | RDM - GAS                                                                           |
| **\[4\]**         | Gorgi, P., Koopman, S.J., & Li, M. (2019). Forecasting Economic Time Series Using Score-Driven Dynamic Models with Mixed-Data Sampling. Journal of Applied Econometrics.                                 | RDM - GAS validação                                                                 |
| **\[5\]**         | Guidolin, M., & Timmermann, A. (2007). Asset Allocation under Multivariate Regime Switching. Journal of Economic Dynamics and Control, 31(11), 3503-3544.                                                | Motivação, Pilar 1                                                                  |
| **\[6\]**         | Nystrup, P., et al. (2018). Dynamic Portfolio Optimization Across Hidden Market Regimes. Quantitative Finance, 18(1), 83-95.                                                                             | RDM + portfólio                                                                     |
| **\[7\] \[v4\]**  | Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal Graph Networks for Deep Learning on Dynamic Graphs. ICML 2020 Workshop on Graph Representation Learning. | **DyFO - TGN (substitui ROLAND)**                                                   |
| **\[8\]**         | Veličković, P., et al. (2018). Graph Attention Networks. ICLR 2018.                                                                                                                                      | DyFO - GAT base                                                                     |
| **\[9\]**         | Shi, Y., et al. (2022). GPM: Graph-Based Portfolio Management via Deep Reinforcement Learning. Expert Systems with Applications.                                                                         | DyFO - baseline B10                                                                 |
| **\[10\]**        | Jiang, Z., Xu, D., & Liang, J. (2017). A Deep Reinforcement Learning Framework for the Financial Portfolio Management Problem. arXiv:1706.10059.                                                         | Baseline RL financeiro                                                              |
| **\[11\]**        | Wang, Z., et al. (2021). DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management. AAAI 2021.                                                                    | Baseline DRL portfólio                                                              |
| **\[12\]**        | Chen, X., et al. (2025). GAT-MARL: A Multi-Agent Reinforcement Learning Approach for Portfolio Optimization Using Graph Attention Networks. Scientific Reports.                                          | Baseline SOTA B12                                                                   |
| **\[13\]**        | Huang, X., & Tanaka, K. (2022). MSPM: Multi-Strategy Portfolio Management via Deep Reinforcement Learning. PLOS ONE.                                                                                     | Baseline modular B11                                                                |
| **\[14\]**        | Li, X., Tam, H.K., & Yeung, C.H. (2024). Graph Reinforcement Learning for Portfolio Management. Expert Systems with Applications.                                                                        | Baseline GNN+RL                                                                     |
| **\[15\]**        | Lowe, R., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments. NeurIPS 2017.                                                                                          | CTDE - paradigma arquitetural                                                       |
| **\[16\]**        | Yu, C., et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games. NeurIPS 2022.                                                                                               | Baseline MAPPO B9                                                                   |
| **\[17\] \[v4\]** | Zhong, Y., Kuba, J.G., Feng, X., Hu, S., Ji, J., & Yang, Y. (2024). Heterogeneous-Agent Reinforcement Learning. Journal of Machine Learning Research, 25(32), 1-67.                                      | **HARL framework - referência definitiva (substitui Kuba et al. 2022 conferência)** |
| **\[18\] \[v4\]** | Liu, J., Zhong, Y., Hu, S., Fu, H., Fu, Q., Chang, X., & Yang, Y. (2024). Maximum Entropy Heterogeneous-Agent Reinforcement Learning. ICLR 2024 (Spotlight).                                             | **HASAC - variante max-entropy do HARL**                                            |
| **\[19\] \[v4\]** | Xu, J., Sobhy, O., & Alrifaee, B. (2024). XP-MARL: Auxiliary Prioritization in Multi-Agent Reinforcement Learning to Address Non-Stationarity. arXiv:2409.11852.                                         | **Estrutura bi-stage Stackelberg - prior empírico para P5 do Módulo 4**             |
| **\[20\]**        | Sutton, R.S., Precup, D., & Singh, S. (1999). Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in RL. Artificial Intelligence, 112(1-2), 181-211.                                        | Orquestrador - framework de options                                                 |
| **\[21\]**        | Kuba, J.G., et al. (2022). Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning. ICLR 2022.                                                                                            | HAPPO - paper de conferência (complementado por \[17\])                             |
| **\[22\]**        | Berkenkamp, F., et al. (2017). Safe Model-Based Reinforcement Learning with Stability Guarantees. NeurIPS 2017.                                                                                          | Módulo 4 - Lyapunov+RL fundamento                                                   |
| **\[23\]**        | NeurIPS 2025. Certifying Stability of Reinforcement Learning Policies Using Generalized Lyapunov Functions. NeurIPS 2025.                                                                                | Módulo 4 - Lyapunov SOTA                                                            |
| **\[24\]**        | Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. PNAS, 114(13), 3521-3526.                                                                                         | M4 - EWC                                                                            |
| **\[25\]**        | Bengio, Y., et al. (2009). Curriculum Learning. ICML 2009.                                                                                                                                               | M3 - Curriculum Learning                                                            |
| **\[26\]**        | Rockafellar, R.T., & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. Journal of Risk, 2(3), 21-41.                                                                                        | Reward CVaR                                                                         |
| **\[27\]**        | Almgren, R., & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. Journal of Risk, 3(2), 5-39.                                                                                              | Módulo 5 - custo de transação                                                       |
| **\[28\]**        | López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.                                                                                                                                | Módulo 5 - protocolo experimental                                                   |
| **\[29\]**        | Liu, X., et al. (2022). FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial RL. NeurIPS 2022 Workshop.                                                                              | Módulo 5 - benchmark FinRL-Meta                                                     |
| **\[30\]**        | Kakushadze, Z., & Serur, J.A. (2018). 151 Trading Strategies. SSRN:3247865.                                                                                                                              | Alpha Signal Layers - §3.1, §3.6, §3.9, §3.20, §4.2, §4.6                           |
| **\[31\]**        | Jegadeesh, N., & Titman, S. (1993). Returns to Buying Winners and Selling Losers. Journal of Finance, 48(1), 65-91.                                                                                      | SA-Trend - warm-start parâmetros canônicos                                          |
| **\[32\]**        | Fama, E.F., & French, K.R. (2015). A Five-Factor Asset Pricing Model. Journal of Financial Economics, 116(1), 1-22.                                                                                      | SA-Macro - modelo de fatores                                                        |

# 15\. Guia de Leitura - MATTS v4.0

**Como usar:** leia N1 antes de N2, e N2 antes de N3. Os itens com **★v4** são adições ou atualizações da v4.0 não presentes no guia anterior. Sincronize a leitura com o cronograma da Seção 11. Os 20 obrigatórios da lista final são os mínimos para a qualificação.

| **#**  | **Pilar**                        | **Componente MATTS v4**          | **RSL**  | **Contribuição central**                                    |
| ------ | -------------------------------- | -------------------------------- | -------- | ----------------------------------------------------------- |
| **1**  | Finanças Quantitativas Clássica  | Baselines, motivação             | Q3       | Justifica por que regimes + RL superam modelos estáticos    |
| **2**  | Séries Temporais e Regimes       | RDM - HMM, GAS, TVTP             | Q1, Q9   | Detecção de regimes com quantificação de incerteza H(π_t)   |
| **3**  | Risco e Portfólio                | Reward CVaR, Módulo 4            | Q11      | CVaR regime-condicionado como função de recompensa coerente |
| **4**  | RL - Fundamentos                 | Todos os agentes                 | Q2-Q4    | Base formal para HAPPO/HASAC e análise de convergência      |
| **5**  | **Deep RL e MARL / HARL \[v4\]** | Orquestrador + Sub-Agentes       | Q3, Q4   | HARL JMLR 2024 + HASAC + XP-MARL - núcleo das contribuições |
| **6**  | **Graph Neural Networks \[v4\]** | DyFO - TGN, GAT                  | Q5       | TGN: grafos de eventos contínuos vs. snapshots discretos    |
| **7**  | Controle e Estabilidade          | Módulo 4 - Lyapunov, Stackelberg | Q6       | Provas P1-P5; P5 fundamentada em XP-MARL                    |
| **8**  | Continual e Curriculum Learning  | M3 + M4 - EWC, Curriculum        | Q10      | Prevenção de catastrophic forgetting em HARL financeiro     |
| **9**  | Estratégias Quantitativas        | Alpha Signal Layers              | Q3, V4-1 | Sinais clássicos K&S que alimentam os sub-agentes híbridos  |
| **10** | Metodologia Experimental         | Módulo 5 - validação             | Q8       | Anti-leakage, deflated SR, 16 baselines, reprodutibilidade  |

**Pilar 1 - Finanças Quantitativas Clássica**

_Motivação empírica - por que modelos estáticos falham e por que regimes importam. Baselines diretos do MATTS._

| **Tipo** | **Dif.** | **Referência**                                                                            | **O que ler**           | **Por que é essencial**                                                                                    | **Conecta com** |
| -------- | -------- | ----------------------------------------------------------------------------------------- | ----------------------- | ---------------------------------------------------------------------------------------------------------- | --------------- |
| **TB**   | **N1**   | Campbell, Lo & MacKinlay - Econometrics of Financial Markets (1997)                       | Caps. 1-2, 5, 9         | Fatos estilizados que motivam o RDM. Mostra por que retornos não são i.i.d.                                | Pilar 2         |
| **TB**   | **N1**   | Cochrane - Asset Pricing (2005)                                                           | Caps. 1-4, 20           | SDF framework + predictability. Base teórica do SA-Macro e do baseline Multifactor.                        | Pilar 9         |
| **TB**   | **N2**   | Ilmanen - Expected Returns (2011)                                                         | Caps. 4-7, 12           | Prêmios de risco por classe de ativo - base para SA-Macro e ontologia do DyFO.                             | Pilares 6, 9    |
| **ART**  | **N2**   | Guidolin & Timmermann (2007) - Asset Allocation under Multivariate Regime Switching. JEDC | Completo                | ★ obrigatório. Demonstra empiricamente que regimes alteram a alocação ótima. Motivação central do projeto. | Pilar 2, 3      |
| **ART**  | **N3**   | Ang & Bekaert (2002) - Int. Asset Allocation with Regime Shifts. RFS                      | Seções 1-3 + resultados | Correlações entre ativos mudam por regime - motiva o grafo dinâmico TGN.                                   | Pilares 2, 6    |
| **SV**   | **N2**   | Harvey et al. (2016) - Cross-Section of Expected Returns. RFS                             | Seções 1-3, Conclusão   | Múltiplos testes em fatores - base para deflated SR e protocolo experimental.                              | Pilar 10        |

**Pilar 2 - Séries Temporais Financeiras e Detecção de Regimes**

_Fundamento direto do Módulo 1 (RDM): HMM, GAS, TVTP e alternativas. Sem alteração na v4._

| **Tipo** | **Dif.** | **Referência**                                                        | **O que ler**            | **Por que é essencial**                                                      | **Conecta com** |
| -------- | -------- | --------------------------------------------------------------------- | ------------------------ | ---------------------------------------------------------------------------- | --------------- |
| **TB**   | **N1**   | Hamilton - Time Series Analysis (1994)                                | Caps. 1-4, 22            | ★ obrigatório. Fundamento do HMM financeiro. Derive o filtro forward.        | Base do RDM     |
| **TB**   | **N1**   | Durbin & Koopman - Time Series Analysis by State Space Methods (2012) | Caps. 1-4, 6, 10         | Filtragem de Kalman e extensões não-Gaussianas - algoritmos centrais do RDM. | RDM - filtragem |
| **ART**  | **N1**   | Hamilton (1989) - Econometrica. 57(2):357-384                         | Completo                 | ★ obrigatório. Paper original do HMM financeiro. Derive cada equação na mão. | Base do RDM     |
| **ART**  | **N2**   | Filardo (1994) - JBES. 12(3):299-308                                  | Completo                 | ★ obrigatório. TVTP - extensão direta de Hamilton 1989.                      | RDM - TVTP      |
| **ART**  | **N2**   | Creal, Koopman & Lucas (2013) - JAE. 28(5):777-795                    | Seções 1-4 + Apêndice A  | ★ obrigatório. Define GAS - coração do RDM v4.0.                             | RDM - GAS       |
| **ART**  | **N2**   | Gorgi, Koopman & Li (2019) - JAE                                      | Seções 1-3               | Validação empírica do GAS em dados financeiros reais.                        | RSL Q1          |
| **ART**  | **N2**   | Nystrup et al. (2018) - Quant Finance                                 | Completo                 | ★ obrigatório. Ponte entre RDM e decisão de portfólio.                       | Pilares 1, 3    |
| **ART**  | **N3**   | Adams & MacKay (2007) - BOCPD. arXiv:0710.3742                        | Algoritmo + experimentos | Alternativa ao HMM revisada na RSL Q9.                                       | RSL Q9          |

**Pilar 3 - Otimização de Portfólio e Gestão de Risco**

_Fundamento do reward CVaR regime-condicionado e dos baselines clássicos. Sem alteração na v4._

| **Tipo** | **Dif.** | **Referência**                                                         | **O que ler**   | **Por que é essencial**                                                                     | **Conecta com** |
| -------- | -------- | ---------------------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------- | --------------- |
| **TB**   | **N2**   | McNeil, Frey & Embrechts - Quantitative Risk Management (2ª ed., 2015) | Caps. 1-2, 6, 8 | ★ obrigatório. CVaR como medida coerente - base matemática do Módulo 4.                     | Reward CVaR     |
| **ART**  | **N1**   | Rockafellar & Uryasev (2000) - J. Risk. 2(3):21-41                     | Completo        | ★ obrigatório. Define CVaR e sua otimizabilidade via LP. Derive a formulação do reward r_t. | Reward CVaR     |
| **ART**  | **N2**   | Artzner et al. (1999) - Coherent Measures of Risk. Math. Finance       | Completo        | Prova que CVaR é coerente e VaR não - justificativa teórica da escolha.                     | Pilar 3         |
| **ART**  | **N2**   | Almgren & Chriss (2001) - J. Risk. 3(2):5-39                           | Completo        | ★ obrigatório. Modelo de impacto de mercado do Módulo 5. Derive δ(k_t).                     | Módulo 5        |
| **ART**  | **N2**   | Ledoit & Wolf (2004) - J. Portfolio Management                         | Completo        | Shrinkage de covariância - baseline B3 do MATTS.                                            | Baselines       |

**Pilar 4 - Aprendizado por Reforço - Fundamentos**

_Base formal obrigatória antes de qualquer trabalho com HARL, HAPPO ou HASAC._

| **Tipo** | **Dif.** | **Referência**                                                          | **O que ler**       | **Por que é essencial**                                                                             | **Conecta com**  |
| -------- | -------- | ----------------------------------------------------------------------- | ------------------- | --------------------------------------------------------------------------------------------------- | ---------------- |
| **TB**   | **N1**   | Sutton & Barto - Reinforcement Learning: An Introduction (2ª ed., 2018) | Caps. 1-7, 9-10, 13 | ★ obrigatório completo. Derive TD update, policy gradient e convergência do Q-learning.             | Todos os agentes |
| **TB**   | **N2**   | Bertsekas - RL and Optimal Control (2019)                               | Caps. 1-3, 5        | Formalização matemática rigorosa - necessário para as provas do Módulo 4.                           | Pilar 7          |
| **TB**   | **N2**   | Puterman - Markov Decision Processes (1994)                             | Caps. 1-4, 6, 8     | Referência formal de MDP para provas de suficiência do FDAM.                                        | FDAM, Pilar 7    |
| **ART**  | **N2**   | Schulman et al. (2017) - PPO. arXiv:1707.06347                          | Completo            | ★ obrigatório. Ancestral direto do HAPPO. Derive o clipping objective.                              | Pilar 5          |
| **ART**  | **N3**   | Haarnoja et al. (2018) - SAC. ICML                                      | Completo            | SAC single-agent - base do HASAC. Entender entropy regularization aqui antes de ler HARL JMLR 2024. | HASAC - Pilar 5  |

**Pilar 5 - Deep RL, MARL, HARL e Hierarquia de Políticas**

_NÚCLEO DA CONTRIBUIÇÃO. Três adições v4: HARL JMLR 2024, HASAC, XP-MARL._

| **Tipo** | **Dif.**   | **Referência**                                          | **O que ler**   | **Por que é essencial**                                                                                       | **Conecta com**    |
| -------- | ---------- | ------------------------------------------------------- | --------------- | ------------------------------------------------------------------------------------------------------------- | ------------------ |
| **TB**   | **N2**     | Shoham & Leyton-Brown - Multiagent Systems (2009)       | Caps. 3-5, 7    | ★ obrigatório. Nash, Stackelberg, learning in games - leia antes de qualquer paper MARL.                      | Stackelberg, FDAM  |
| **ART**  | **N1**     | Lowe et al. (2017) - MADDPG. NeurIPS                    | Completo        | ★ obrigatório. Introduz CTDE - paradigma arquitetural do MATTS.                                               | Orquestrador       |
| **ART**  | **N2**     | Yu et al. (2022) - MAPPO. NeurIPS                       | Completo        | ★ obrigatório. MAPPO supera MADDPG/MATD3 - justificativa empírica para HARL.                                  | Baseline B9        |
| **ART**  | **N2**     | Kuba et al. (2022) - HAPPO. ICLR                        | Completo        | Paper de conferência do HAPPO - ler como introdução antes da versão JMLR 2024.                                | HARL - entrada     |
| **ART**  | **N2 ★v4** | Zhong, Kuba et al. (2024) - HARL. JMLR 25(32):1-67      | Completo (67pp) | ★ obrigatório v4. Versão definitiva do HARL: HAML theorem, HAPPO, HATD3. Referência primária do orquestrador. | HARL JMLR 2024     |
| **ART**  | **N3 ★v4** | Liu et al. (2024) - HASAC. ICLR 2024 Spotlight          | Completo        | ★ obrigatório v4. Maximum entropy HARL para espaços contínuos. Coerência com M6 (H(π_t)).                     | HASAC ablation B15 |
| **ART**  | **N3 ★v4** | Xu, Sobhy & Alrifaee (2024) - XP-MARL. arXiv:2409.11852 | Completo        | ★ obrigatório v4. Prior empírico da estrutura bi-stage Stackelberg. Prioridade aprendida via MARL auxiliar.   | P5, Módulo 4       |
| **ART**  | **N2**     | Sutton, Precup & Singh (1999) - Options. AI Journal     | Completo        | ★ obrigatório. Framework de options - base da hierarquia policy-over-agents.                                  | Orquestrador       |
| **ART**  | **N3**     | Ma et al. (2021) - Feudal MARL. arXiv                   | Completo        | Hierarquia feudal em MARL - referência comparativa para P5.                                                   | P5 - Módulo 4      |
| **SV**   | **N2**     | Gronauer & Diepold (2022) - MARL Survey. AI Review      | Completo        | Survey de MARL até 2022 - panorama para posicionar HARL.                                                      | RSL Q4             |

**Pilar 6 - Graph Neural Networks e Grafos Temporais**

_DyFO atualizado na v4: TGN substitui ROLAND. Ablation B16 compara os dois._

| **Tipo** | **Dif.**   | **Referência**                                  | **O que ler** | **Por que é essencial**                                                                                          | **Conecta com** |
| -------- | ---------- | ----------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------- | --------------- |
| **TB**   | **N1**     | Hamilton - Graph Representation Learning (2020) | Caps. 1-4     | ★ obrigatório. GCN, GraphSAGE, mecanismos de atenção - base obrigatória antes de GAT e TGN.                      | DyFO - base GNN |
| **ART**  | **N1**     | Veličković et al. (2018) - GAT. ICLR            | Completo      | ★ obrigatório. Mecanismo de atenção por aresta - componente de readout do TGN no DyFO.                           | DyFO - GAT      |
| **ART**  | **N2 ★v4** | Rossi et al. (2020) - TGN. ICML Workshop on GRL | Completo      | ★ obrigatório v4. Temporal Graph Networks com memória por nó para eventos assíncronos. Substitui ROLAND no DyFO. | DyFO - TGN      |
| **ART**  | **N2 ★v4** | You et al. (2022) - ROLAND. KDD                 | Completo      | ★ obrigatório v4. Leia ROLAND para entender o que TGN supera - baseline do ablation B16.                         | Ablation B16    |
| **ART**  | **N2**     | Shi et al. (2022) - GPM. Expert Systems         | Completo      | ★ obrigatório como baseline B10. GAT estático + RL - referência direta para medir ganho do TGN.                  | Baseline B10    |
| **ART**  | **N3**     | Pareja et al. (2020) - EvolveGCN. AAAI          | Completo      | Alternativa dinâmica ao TGN - necessário para RSL Q5.                                                            | RSL Q5          |

**Pilar 7 - Teoria de Controle, Estabilidade e Jogos Hierárquicos**

_P5 (Stackelberg) atualizada na v4 com XP-MARL como prior empírico. Mais rigorosa que v3._

| **Tipo** | **Dif.**   | **Referência**                                                         | **O que ler**                           | **Por que é essencial**                                                                                | **Conecta com**   |
| -------- | ---------- | ---------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------ | ----------------- |
| **TB**   | **N1**     | Khalil - Nonlinear Systems (3ª ed., 2002)                              | Caps. 1-6, 9                            | ★ obrigatório. Lyapunov stability theorem - referência direta de P1-P3. Derive cada teorema.           | Módulo 4 - P1-P3  |
| **TB**   | **N2**     | Vidyasagar - Nonlinear Systems Analysis (2ª ed., 2002)                 | Caps. 5-6                               | Teorema do pequeno ganho - fundamento de P2 (acoplamento RDM-orquestrador).                            | Módulo 4 - P2     |
| **TB**   | **N2**     | Basar & Olsder - Dynamic Noncooperative Game Theory (2ª ed., 1999)     | Caps. 1-3, 5, 7                         | ★ obrigatório. Jogos de Stackelberg dinâmicos - referência formal de P5.                               | Módulo 4 - P5     |
| **ART**  | **N2**     | Berkenkamp et al. (2017) - Safe MBRL. NeurIPS                          | Completo                                | ★ obrigatório. Primeira conexão rigorosa Lyapunov + RL. P1-P4 estendem este trabalho.                  | Módulo 4 - âncora |
| **ART**  | **N3**     | NeurIPS 2025 - Certifying Stability via Generalized Lyapunov Functions | Completo                                | ★ obrigatório. SOTA em certificação de estabilidade RL. P1-P4 devem se posicionar frente a este paper. | Módulo 4 - SOTA   |
| **ART**  | **N3 ★v4** | Xu, Sobhy & Alrifaee (2024) - XP-MARL. arXiv:2409.11852                | Seção de análise teórica + experimentos | ★ obrigatório v4. Prior empírico de P5: estrutura bi-stage com prioridade aprendida.                   | P5 - Módulo 4     |

**Pilar 8 - Continual Learning e Curriculum Learning**

_EWC + Curriculum 4 estágios. Sem alteração na v4. AEWC documentado como contingência._

| **Tipo** | **Dif.** | **Referência**                                                            | **O que ler** | **Por que é essencial**                                                                | **Conecta com** |
| -------- | -------- | ------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------- | --------------- |
| **ART**  | **N1**   | Kirkpatrick et al. (2017) - EWC. PNAS 114(13):3521                        | Completo      | ★ obrigatório. Paper original do EWC. Derive a penalidade de Fisher.                   | M4 - EWC        |
| **ART**  | **N2**   | Bengio et al. (2009) - Curriculum Learning. ICML                          | Completo      | ★ obrigatório. Os 4 estágios do currículo do MATTS derivam dos princípios deste paper. | M3 - Curriculum |
| **ART**  | **N2**   | Zenke, Poole & Ganguli (2017) - Synaptic Intelligence. ICML               | Completo      | Alternativa mais eficiente ao EWC - necessário para RSL Q10.                           | RSL Q10         |
| **ART**  | **N2**   | Rolnick et al. (2019) - Experience Replay for Continual Learning. NeurIPS | Completo      | Replay estratificado por regime - complemento ao EWC no MATTS.                         | M4 - replay     |
| **SV**   | **N3**   | De Lange et al. (2022) - Continual Learning Survey. TPAMI                 | Seções 3-5    | Survey para posicionar EWC no panorama atual e identificar AEWC.                       | RSL Q10         |

**Pilar 9 - Estratégias Quantitativas e Alpha Combination**

_Base dos Alpha Signal Layers e dos baselines K&S. Sem alteração na v4._

| **Tipo** | **Dif.** | **Referência**                                                   | **O que ler**                              | **Por que é essencial**                                                                                 | **Conecta com**      |
| -------- | -------- | ---------------------------------------------------------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------- | -------------------- |
| **TB**   | **N1**   | Kakushadze & Serur (2018) - 151 Trading Strategies. SSRN:3247865 | §3.1, §3.6, §3.9, §3.18, §3.20, §4.2, §4.6 | ★ obrigatório. Equações dos Alpha Signal Layers de cada sub-agente. Implemente as fórmulas da Tabela 1. | Alpha Signal Layers  |
| **TB**   | **N2**   | Grinold & Kahn - Active Portfolio Management (2ª ed., 1999)      | Caps. 6-8, 14                              | Information Coefficient e Fundamental Law - base para avaliar qualidade dos Alpha Signal Layers.        | Alpha Signal quality |
| **ART**  | **N1**   | Jegadeesh & Titman (1993) - J. Finance 48(1):65                  | Completo                                   | ★ obrigatório. Momentum canônico S=12m - warm-start do SA-Trend.                                        | SA-Trend warm-start  |
| **ART**  | **N2**   | Fama & French (2015) - J. Fin. Econ. 116(1):1                    | Seções 1-3                                 | Modelo FF5 - base do Alpha Signal Layer do SA-Macro (eq. 364 de K&S).                                   | SA-Macro signal      |
| **ART**  | **N3**   | Frazzini & Pedersen (2014) - Betting Against Beta. J. Fin. Econ. | Seções 1-4                                 | Low-volatility anomaly - base do Alpha Signal Layer do SA-Risk (K&S §3.4).                              | SA-Risk signal       |

**Pilar 10 - Metodologia Experimental em ML Financeiro**

_Protocolo anti-leakage, deflated SR, 16 baselines. Sem alteração na v4._

| **Tipo** | **Dif.** | **Referência**                                                            | **O que ler**     | **Por que é essencial**                                                                           | **Conecta com**        |
| -------- | -------- | ------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------- | ---------------------- |
| **TB**   | **N1**   | López de Prado - Advances in Financial ML (2018)                          | Caps. 1-7, 14, 16 | ★ obrigatório completo. Cap. 16 (Deflated SR) é leitura mandatória antes de qualquer experimento. | Módulo 5               |
| **ART**  | **N2**   | Bailey & López de Prado (2014) - Deflated Sharpe Ratio. J. Portfolio Mgmt | Completo          | ★ obrigatório. Derive a fórmula - o número de configurações testadas entra no denominador.        | Módulo 5 - métrica     |
| **ART**  | **N1**   | Liu et al. (2022) - FinRL-Meta. NeurIPS Workshop                          | Completo          | ★ obrigatório. Framework de benchmarking padronizado. Leia antes de implementar o Módulo 5.       | Módulo 5 - benchmark   |
| **ART**  | **N2**   | White (2000) - Reality Check for Data Snooping. Econometrica              | Seções 1-3        | Reality Check bootstrap - fundamento dos 500 bootstrap do protocolo experimental.                 | Módulo 5 - Monte Carlo |
| **ART**  | **N2**   | Harvey & Liu (2015) - Backtesting. J. Portfolio Mgmt                      | Completo          | Taxonomia de erros de backtesting - checklist para o protocolo do Módulo 5.                       | Módulo 5 - validação   |

# 15.1 Roteiro de Leitura - 48 Meses

Sincronizado com o cronograma da Seção 11. Colunas marcadas em verde são adições da v4.

| **Período** | **Fase**                         | **Pilares em foco** | **Leituras-chave**                                                         | **Marco de domínio**                                                                   | **v4 novo**                         |
| ----------- | -------------------------------- | ------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ----------------------------------- |
| **M1-4**    | **Base Absoluta**                | P1, P4, P10         | Hamilton (1994) Caps. 22; Sutton & Barto completo; López de Prado completo | HMM em Python; backtest com deflated SR; derivar Bellman na mão                        | -                                   |
| **M3-6**    | **Regimes e Risco**              | P2, P3              | Hamilton (1989); Filardo; Creal et al.; Rockafellar & Uryasev              | RDM funcional (HMM+TVTP+GAS) com π_t e H(π_t); formulação do reward CVaR               | -                                   |
| **M5-8**    | **RL Fundamentos**               | P4, P9              | Sutton & Barto Caps. 9-13; Schulman PPO; K&S §3.1,§3.9                     | Replicar benchmark RL de portfólio; Alpha Signal Layers SA-Trend e SA-MeanRev          | -                                   |
| **M5-10**   | **GNN + TGN \[v4\]**             | P6                  | Hamilton GRL Caps. 1-4; Veličković GAT; Rossi TGN; ROLAND                  | TGN sobre grafo de 50 ativos; ablation B16 preliminar: TGN vs. ROLAND vs. GAT          | **TGN + ablation B16 novos**        |
| **M7-14**   | **HARL \[v4\]**                  | P5                  | Shoham & LB Caps. 3-5; Lowe MADDPG; Yu MAPPO; Zhong HARL JMLR 2024         | Derivar Fundamental Theorem HAML; implementar CTDE com HAPPO e HASAC                   | **HARL JMLR 2024 + HASAC novos**    |
| **M10-16**  | **XP-MARL + Stackelberg \[v4\]** | P5, P7              | Basar & Olsder Caps. 1-3, 5; Xu et al. XP-MARL 2024                        | Implementar hierarquia bi-stage com prioridade aprendida; esboço de P5                 | **XP-MARL como prior de P5 - novo** |
| **M12-20**  | **Continual RL**                 | P8                  | Kirkpatrick EWC; Bengio Curriculum; Rolnick replay                         | EWC sobre agente HAPPO treinado; demonstrar redução de forgetting em mudança de regime | -                                   |
| **M16-28**  | **Integração MATTS v4**          | P5, P6, P7          | Zhong HARL 2024; Berkenkamp; NeurIPS 2025 Lyapunov                         | Sistema completo integrado; tabelas experimentais preliminares; 16 baselines           | **B15 + B16 novos**                 |
| **M24-34**  | **Análise Teórica**              | P7                  | Khalil Caps. 5-6; Vidyasagar Cap. 6; NeurIPS 2025; Xu XP-MARL              | Provas P1-P5 em LaTeX; P5 com XP-MARL; FDAM recursivo formalizado                      | **P5 com XP-MARL - novo**           |
| **M32-46**  | **Validação + Escrita**          | P10                 | López de Prado Cap. 16; FinRL-Meta; Harvey & Liu                           | Todas as tabelas H1-H10; ablations B13-B16; paper experimental submetido               | **H3, H4, H9, H10 novos**           |

# 15.2 Lista de 20 Obrigatórios - Mínimo para Qualificação

**Critério de domínio:** não é ter lido - é conseguir executar o item "Derivar/Implementar" sem consultar nenhuma fonte. Os itens marcados **★v4** são novos ou substituídos na v4.0 em relação ao guia anterior.

| **#**  | **Pilar** | **Referência**                             | **Tipo** | **Derivar / Implementar**                                                                             |
| ------ | --------- | ------------------------------------------ | -------- | ----------------------------------------------------------------------------------------------------- |
| **1**  | **P1**    | Guidolin & Timmermann (2007)               | ART      | Replicar Tabela 3 - alocação ótima por regime em dois datasets                                        |
| **2**  | **P2**    | Hamilton (1989) - Econometrica             | ART      | Derivar filtro forward completo; implementar HMM 2-regime em Python                                   |
| **3**  | **P2**    | Filardo (1994) - JBES                      | ART      | Estender HMM com TVTP; comparar log-likelihood com Hamilton básico                                    |
| **4**  | **P2**    | Creal, Koopman & Lucas (2013)              | ART      | Derivar mecanismo GAS; implementar score update de σ_t                                                |
| **5**  | **P2**    | Nystrup et al. (2018)                      | ART      | Replicar otimização de portfólio com HMM sobre dados Fama-French                                      |
| **6**  | **P3**    | Rockafellar & Uryasev (2000)               | ART      | Derivar formulação LP do CVaR; codificar r_t = CVaR(R_t\|regime) − δ(k_t)                             |
| **7**  | **P3**    | Almgren & Chriss (2001)                    | ART      | Derivar modelo de custo; implementar δ(k_t) para 3 níveis de liquidez                                 |
| **8**  | **P4**    | Sutton & Barto (2018) Caps. 1-13           | TB       | Derivar policy gradient theorem na lousa sem livro; escrever REINFORCE com baseline                   |
| **9**  | **P4**    | Schulman et al. PPO (2017)                 | ART      | Derivar clipping objective; provar melhoria monótona single-agent em 3 passos                         |
| **10** | **P5**    | Shoham & LB (2009) Caps. 3-5               | TB       | Derivar Nash e Stackelberg equilibrium formalmente; identificar diferença estrutural                  |
| **11** | **P5**    | Lowe et al. MADDPG (2017)                  | ART      | Implementar CTDE básico com 2 agentes em ambiente cooperativo                                         |
| **12** | **P5**    | **Zhong et al. HARL JMLR 2024 ★v4**        | ART      | Derivar Fundamental Theorem of HAML em 4 passos; identificar onde a sequencialidade é usada           |
| **13** | **P5**    | **Liu et al. HASAC ICLR 2024 ★v4**         | ART      | Derivar objetivo de maximum entropy do HASAC; comparar com HAPPO em espaço contínuo sintético         |
| **14** | **P5**    | **Xu et al. XP-MARL arXiv:2409.11852 ★v4** | ART      | Implementar estrutura bi-stage com prioridade fixa; esboçar extensão com prioridade aprendida         |
| **15** | **P6**    | Veličković et al. GAT (2018)               | ART      | Implementar GAT sobre grafo financeiro de 50 ativos; verificar atenção por aresta                     |
| **16** | **P6**    | **Rossi et al. TGN (2020) ★v4**            | ART      | Implementar TGN sobre stream de eventos financeiros diários; medir latência de atualização vs. ROLAND |
| **17** | **P7**    | Khalil (2002) Caps. 5-6                    | TB       | Derivar Lyapunov stability theorem; construir V(x) candidata para sistema dinâmico simples            |
| **18** | **P7**    | Berkenkamp et al. (2017)                   | ART      | Derivar certificado de estabilidade; adaptar para sistema acoplado RDM-orquestrador                   |
| **19** | **P8**    | Kirkpatrick et al. EWC (2017)              | ART      | Derivar penalidade de Fisher; implementar EWC sobre HAPPO treinado em mudança de regime simulada      |
| **20** | **P10**   | López de Prado (2018) completo             | TB       | Calcular deflated SR com N=16 configurações; identificar look-ahead bias em backtest de exemplo       |

# 15.3 Critérios de Domínio por Pilar

_Você domina um pilar quando consegue executar os itens abaixo sem consultar nenhuma fonte. Leitura com sublinhado não é domínio._

| **P2 - Regimes** | Implementar o filtro forward de um HMM com TVTP do zero em 2 horas, produzir π_t e H(π_t) sobre dados reais e plotar a sequência de regimes filtrada. |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |

| **P3 - CVaR** | Escrever a formulação do reward r_t com todos os parâmetros, justificar a coerência do CVaR vs. VaR, e implementar o cálculo CVaR-95% sobre uma janela rolante em 30 minutos. |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

| **P4 - RL** | Derivar o policy gradient theorem na lousa, sem livro, partindo da definição de J(θ). Escrever o algoritmo REINFORCE com baseline do zero. |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------ |

| **P5 - HARL \[v4\]** | Derivar o Fundamental Theorem of HAML em 4 passos. Explicar por que a atualização sequencial garante monotonia que a atualização simultânea não garante. Diferenciar HAPPO de HASAC em 2 frases. |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

| **P5 - XP-MARL \[v4\]** | Definir o problema bi-stage do XP-MARL formalmente. Identificar onde o orquestrador do MATTS é o agente de alta prioridade. Escrever a condição de equilíbrio de Stackelberg informalmente. |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

| **P6 - TGN \[v4\]** | Implementar o módulo de memória do TGN sobre um grafo financeiro de 10 ativos com 3 tipos de evento. Comparar o embedding e_t gerado com o de GAT estático em 2 timesteps de crise simulada. |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |

| **P7 - Lyapunov** | Dado um sistema dinâmico x\_{t+1} = f(x_t), construir uma função candidata V(x) e verificar as condições de decréscimo. Explicar o que a condição de pequeno ganho diz sobre o acoplamento RDM-orquestrador. |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |

| **P10 - Experimental** | Calcular o deflated Sharpe Ratio de uma estratégia hipotética dado N=16 configurações testadas. Identificar um look-ahead bias em um backtest de exemplo e corrigi-lo. |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |