## Entradas brutas do mercado

- Retornos R_t: N ativos, tempo = (diário, hora, minuto)
- Macroeconomiza z_t: VIX, spread, curva
- Eventos financeiros: earnings, Fed, Rating
- Fatores FF5 + Preços (OLS c/ rolling window = 252): O sinal que sai é o normalizado - o retorno de cada ativo que não é explicado por nenhum dos cinco fatores. O RL Adaptation Layer então aprende a ajustar as exposições-alvo e o timing de rotação entre fatores de acordo com o regime - por exemplo, aumentando exposição a HML em regimes de recuperação econômica e reduzindo em ciclos de crescimento acelerado.

![](data:image/png;base64,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)

- 1. MKT - retorno do mercado acima da taxa livre de risco (prêmio de risco de mercado),
  - SMB (Small Minus Big) - retorno de empresas pequenas menos empresas grandes, capturando o prêmio de tamanho
  - HML (High Minus Low) - retorno de empresas com alto book-to-market menos as de baixo, capturando o prêmio de valor
  - RMW (Robust Minus Weak) - retorno de empresas com alta lucratividade operacional menos as de baixa, adicionado no modelo de 5 fatores de Fama & French (2015)
  - CMA (Conservative Minus Aggressive) - retorno de empresas com baixo investimento em ativos menos as de alto investimento, também do modelo de 2015

# M1

## RDM - Detecta o regime de mercado

- Inputs: Retornos R_t dos N ativos, Variáveis macro z_t, VIX, spread crédito, inclinação da curva, CPI, crescimento PIB
- Processamento: Filtro forward Hamilton atualiza as propriedades filtradas de rgime. GAS recalibra mi_k e \\Sigma_k via score de Fisher. TVTP ajusta a matriz de transição P(z_t) via logit.
- Outputs:
  - Vetor de probabilidades de regime pi_t como probabilidade de transição de um regime x em t para um regime y em t+1.
  - Entropia epistêmica H(pi_t) com score de confiabilidade

Ex M1: K=3 · π_t = (0.72, 0.21, 0.07) → regime bull provável · H(π_t) = 0.84 (baixa incerteza)

# M2

## DyFO (TGN) atualiza o grafo financeiro

#### Arquitetura

A arquitetura é composta por quatro módulos encadeados:

- O módulo de **memória** mantém um vetor de estado por nó, atualizado a cada evento que envolve aquele nó - esse é o componente mais crítico, pois permite que o modelo se lembre do histórico de longo prazo de cada entidade.
- O módulo de **mensagens** computa, a cada evento, uma mensagem que carrega informação da interação (memória dos dois nós, tempo decorrido e features da aresta).
- O **agregador de mensagens** resolve o caso em que múltiplos eventos ocorrem para o mesmo nó num mesmo batch, combinando-os numa mensagem consolidada.
- Por fim, o módulo de **embedding** gera a representação atual do nó usando a memória e os vizinhos no grafo temporal - com variantes que vão de uma simples projeção de identidade até atenção multi-head sobre a vizinhança temporal.

Um desafio de treinamento importante é que os módulos de memória não recebem gradiente diretamente. A solução proposta é o Raw Message Store: ao processar um batch, o modelo primeiro atualiza as memórias usando mensagens de batches anteriores (armazenadas), depois prediz as interações do batch atual, e por último armazena as mensagens do batch atual para uso futuro. Isso evita data leakage e permite que os gradientes fluam para os módulos de memória.

- Inputs:
  - Grafo G= (V, E) do dia anterior com memórias m_i(t⁻)
  - Eventos financeiros assíncronos do dia: earnings, decisões Fed, rebaixamentos de rating, M&A
- Processamento: Para cada evento e=(u,v,t_e,f_e): módulo de mensagem computa msg_u e msg_v. Módulo de memória atualiza m_u(t_e⁺) e m_v(t_e⁺) via GRU. Ao final do dia, GAT computa z_i(t) sobre os estados de memória atuais.
- Output:
  - Embeddings por nó **z_i(t)** ∈ ℝ⁶⁴ para cada ativo i
  - Embedding do grafo **e_t** = Σ w_i^cap · z_i(t) ∈ ℝ⁶⁴

Ex M2: Apple divulga earnings acima do esperado → m_AAPL atualizado imediatamente → z_AAPL(t) reflete correlação dinâmica com fornecedores TSMC, FOXC

## M3 - State Constructor

### Alpha Signal Layers calculam scores fechados

- Inputs:
  - SA-Trend: retornos r\_{i,t}^(τ) em janelas múltiplas
  - SA-MeanRev: retornos cross-section + identidade setorial
  - SA-Risk: σ_i, β_i, MDD_i em janela 252d
  - SA-Macro: r\_{i,t} e fatores FF5 em janela 252d
- Processamento: Equações fechadas K&S, sem parâmetros aprendidos. Saída determinística dado o input. Parametrizado pelos hiperparâmetros do dia anterior (τ_lb, W, ζ, T_est) decididos pelo RL Adaptation Layer ontem.
- Outputs:
  - **α¹_t** = momentum cross-sec. Normalizado
  - **α²_t** = z-score de reversão setorial
  - **α³_t** = score low-vol composto
  - **α⁴_t** = alpha residual FF5 (OLS)

Ex: α¹*t = (r*{i,τ} − r̄*τ) / σ*{i,τ} · · · sem gradiente, sem recompensa, output puro K&S

### State Constructor monta o vetor global s_t

- Inputs:
  - e_t ∈ ℝ⁶⁴ (TGN)
  - π_t ∈ Δ² (RDM, K=3)
  - H(π_t) ∈ ℝ (entropia epistêmica)
  - α¹α²α³α⁴ ∈ ℝ⁴ (Alpha Layers)
  - x*t ∈ ℝ¹⁵⁴ (retornos×3 janelas, vol×3, volume, bid-ask, w*{t-1})
- Processamento: Concatenação determinística. Cada dimensão normalizada por média e desvio padrão do período de treino (sem look-ahead). Invariante: mesma operação sempre produz o mesmo s_t dado as mesmas entradas.
- Outputs: **s_t** ∈ ℝ²²⁶ Vetor unificado enviado ao orquestrador e a cada sub-agente (na versão local s_t^(k))

Ex : s_t = \[ e_t(64) ‖ π_t(3) ‖ H(1) ‖ α¹α²α³α⁴(4) ‖ x_t(154) \] · dim total = 226

### Orquestrador decide w_t e o comunica (XP-MARL)

- Inputs:
  - s_t ∈ ℝ²²⁶ (estado global completo)
  - Inclui π_t e H(π_t) → orquestrador conhece o regime e sua incerteza antes de decidir
- Processamento:
  - Ator MLP \[226→512→256→128→4\] com softmax final. HASAC usa distribuição de Dirichlet.
  - Atualização sequencial HARL: orquestrador atualiza primeiro (líder Stackelberg).
  - w_t propagado via XP-MARL antes dos sub-agentes agirem.
- Outputs:
  - **w_t** = (w₁, w₂, w₃, w₄) ∈ Δ³
  - w_k = fração do capital alocada ao sub-agente k
  - Comunicado a todos os 4 sub-agentes antes deles agirem

Ex: regime bull (π₁=0.72) → w_t = (0.45, 0.15, 0.20, 0.20) → 45% capital para SA-Trend (momentum domina)

### RL Adaptation Layers decidem os hiperparâmetros

- Inputs:
  - Estado local s_t^(k) = \[π_t ‖ H(π_t) ‖ α^k_t ‖ x_t^(k)\]
  - Peso recebido w_k do orquestrador
  - Score do Alpha Signal Layer α^k_t
- Processamento:
  - Cada sub-agente é um seguidor Stackelberg: dado w_k fixo, otimiza seus hiperparâmetros via HAML (atualização sequencial, garantia Nash).
  - EWC penaliza afastamento dos parâmetros do regime anterior.
- Outputs:
  - SA-Trend: (τ_lb, S\*, ξ) → janela e tamanho de posição
  - SA-MeanRev: (W, ζ, ξ) → janela e z-score limiar
  - SA-Risk: (ℓ*max, τ*σ) → limite de posição por ativo
  - SA-Macro: (T_est, β_target) → exposição a fatores

Ex: H(π_t)=0.84 baixo → SA-Trend aumenta τ_lb (lookback longo, confiança no momentum) · SA-Risk reduz ℓ_max (posições menores em bull)

## M4

### Portfolio Executor consolida e envia as ordens

- Inputs:
  - Posições-alvo de cada sub-agente (determinadas pelos hiperparâmetros a_t^(k) aplicados ao score α^k_t)
  - Pesos w_t do orquestrador
  - Posição corrente do portfólio w\_{t-1}
  - Dados de liquidez: volume médio 63d, bid-ask spread
- Processamento:
  - Consolidação ponderada por w_k.
  - Cálculo da variação de posição: Δw*i = w*{i,t} − w\_{i,t-1}.
  - Clipping por limite de posição ℓ_max.
  - Verificação de lote mínimo.
  - Estimativa de custo de transação (Almgren-Chriss).
- Outputs:
  - **k_t** = vetor de quantidades de ordem por ativo (comprar / vender / manter)
  - Custo estimado δ(k_t) para o Reward Constructor

k*{i,t} = clip\[(w*{i,t}·C*t − P*{i,t}) / P^ask*{i,t}, −ℓ_max, ℓ_max\] · 𝟙\[|k*{i,t}| > k_min\]

## M5

### Reward Constructor fecha o loop de aprendizado

- Inputs:
  - Retornos realizados r_t do portfólio
  - Regime vigente s_t = k (do RDM)
  - Custo de transação realizado δ(k_t)
- Processamento:
  - CVaR calculado sobre janela rolante de retornos, condicionado ao regime k.
  - Se Jarque-Bera rejeita normalidade: cauda Pareto Generalizada.
  - Recompensa regime-condicionada propagada a todos os agentes (orquestrador + 4 sub-agentes).
- Outputs:
  - **r_t** = CVaR_α(R_t^w | s_t=k) − δ(k_t) ∈ ℝ
  - Armazenado no replay buffer estratificado por regime B_k
  - Alimenta a atualização HAPPO/HASAC+EWC no próximo batch

Loop completo: t+1 → novos R*t → RDM atualiza π*{t+1} → DyFO processa novos eventos → ciclo recomeça