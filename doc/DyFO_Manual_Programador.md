**MATTS v4.0**

**Manual do Programador**

**Módulo 2 - DyFO com TGN**

_Dynamic Financial Ontology - Temporal Graph Networks_

Versão 4.0 - Março 2026

_Baseado em: Rossi et al. (2020) - TGN; Ablation B16 previsto_

# 0\. Diagnóstico Crítico do DyFO v3 → v4

Esta seção apresenta a avaliação crítica do Módulo 2 frente ao estado da arte em otimização de portfólio com grafos temporais. O diagnóstico identifica o que faz sentido, o que precisa ser corrigido e o que deve ser complementado.

## 0.1 O que faz sentido - pontos validados pela literatura

_VALIDADO: A escolha do TGN sobre snapshots discretos (ROLAND) é corroborada por múltiplos trabalhos de 2024-2025. A literatura convérge em que grafos estáticos falham em períodos de mudança de regime. TAGN (MDPI 2026), STGAT (2025), FSTGAT (2025) e GAP-TGN (ICLR 2026 Workshop) demonstram consistentemente que grafos dinâmicos superam snapshots em ambientes financeiros não-estacionários._

_VALIDADO: Nodos como ativos e arestas como relações (correlação, setor, cadeia de fornecimento) é o design dominante na literatura. Chen et al. (2025) / GAT-MARL usa exatamente quatro tipos de aresta: correlação rólante, similaridade fundamental, afiliação setorial e cadeia de fornecimento. O MATTS usa design compatível._

_VALIDADO: Memória por nó com GRU é o mecanismo SOTA para CTDGs. A literatura 2024 confirma que TGN-attn (com memória) supera modelos sem memória em +4% de precisão e que o problema de staleness é mitigado pela agregação de vizinhos._

_VALIDADO: Usar GAT como readout sobre os estados de memória é correto. Múltiplos trabalhos usam exatamente este pipeline: TGN (encoder) + GAT (readout por aresta interpretável). Korangi et al. (2024) usa GAT com 30 anos de dados de mid-caps e supera MVO e Risk Parity._

_VALIDADO: Granularidade diária com eventos assíncronos é viável. O paper GAP-TGN (ICLR 2026 Workshop) usa exatamente esse design com dados de congô congressional em horizonte diário. TAGN usa dados diários do S&P 500 (2018-2024)._

## 0.2 O que precisa ser CORRIGIDO - lacunas críticas

_LACUNA CRÍTICA 1 - Grafo homogêneo vs. heterogêneo: A proposta atual trata todos os tipos de aresta de forma homogênea. A literatura 2024-2025 demonstra claramente que grafos heterogêneos (com tipos de aresta distintos e métricas de agregação diferentes por tipo) superam grafos homogêneos. Chen et al. (2025) e THGNN (CIKM 2022) mostram que correlação de preço e relação de cadeia de fornecimento têm semânticas fundamentalmente diferentes. O DyFO precisa implementar arestas tipadas com mecanismos de atenção por tipo._

_LACUNA CRÍTICA 2 - Construção da aresta de correlação: Correlação de Pearson janelada é ruidosa e não-estacionária. O estado da arte usa DCC-GARCH (Engle 2002) para correlações dinâmicas entre ativos ou Distance Correlation com TMFG (Triangulated Maximally Filtered Graph) para sparsificação. Usar Pearson bruta gera grafos densos com muitas arestas espurias, levando a over-smoothing._

_LACUNA CRÍTICA 3 - Staleness em ativos com baixa liquidez: O mecanismo de memória do TGN tem staleness problem documentado: nodos que ficam inativos por períodos longos acumulam memória desatualizada. Para ativos menos líquidos (small-caps, ETFs de nícho), isso é especialmente problemático. A literatura (GAP-TGN, ICLR 2026) identifica este como o principal desafio do TGN em finanças. A proposta não documenta mitigação explícita._

_LACUNA CRÍTICA 4 - Features de nó estão zeradas: A proposta MATTS documenta que v_i = 0 (vetor zero para todos os nódos). Isso segue a configuração original do TGN em Wikipedia/Reddit, mas em finanças é subótimo. Trabalhos 2024 (STGAT, DASF-Net, paper de Headlines to Holdings) mostram ganhos significativos ao incluir features fundamentais (P/E, beta, volatilidade histórica, setor) como features iniciais dos nódos._

## 0.3 O que deve ser COMPLEMENTADO

**COMPLEMENTO 1 - Tipos de evento financeiro:** A proposta menciona earnings reports, decisões Fed e rebaixamentos como exemplos, mas não especifica o vetor de features de cada tipo de evento. A literatura (TAGN, THGNN) usa vetores de features diferenciados por tipo: eventos point-in-time (earnings, Fed) vs. eventos de fluxo (co-movimento diário). Esta especificação precisa ser formalizada no design do módulo.

**COMPLEMENTO 2 - Leitura do grafo pelo MARL:** A proposta usa o embedding TGN como e_t ∈ R^d, mas não especifica como o embedding do grafo completo é gerado a partir dos embeddings individuais dos nódos. O readout global (pooling sobre todos os nódos, ou atenção diferenciada para os K ativos do portfólio) é um detalhe crítico que afeta diretamente o que o orquestrador vê.

**COMPLEMENTO 3 - Integração com regime:** Nenhum trabalho 2024-2025 integra TGN + detecção de regime HMM em pipeline end-to-end. O MATTS faz isso via State Constructor (e_t concatenado com pi_t), mas a literatura sugere que condicionar o próprio TGN no regime (usar pi_t como feature de nó durante a agregação) pode ser superior. Isso é uma contribuição original que vale explorar na ablation B16.

**COMPLEMENTO 4 - Sparsificação do grafo:** Para universos de 100+ ativos, um grafo totalmente conectado tem O(N^2) arestas, gerando over-smoothing e custo computacional proibitivo. DASF-Net (2025) usa heat-kernel diffusion para sparsificação, TMFG (Korangi 2024) para filtro de correlação. A proposta não documenta estratégia de sparsificação.

## 0.4 Tabela-Resumo do Diagnóstico

| **Aspecto**                 | **Status**         | **Recomendação**                              |
| --------------------------- | ------------------ | --------------------------------------------- |
| TGN sobre snapshots         | ✅ Validado        | Manter. Suportado por 8+ papers 2024-2025     |
| Memória por nó com GRU      | ✅ Validado        | Manter TGN-attn com 1 camada                  |
| GAT como readout            | ✅ Validado        | Manter. Especificar readout global            |
| Grafo homogêneo             | ⚠️ Lacuna          | Implementar arestas tipadas (4 tipos mínimo)  |
| Correlação Pearson          | ⚠️ Lacuna          | Substituir por DCC-GARCH ou Distance Corr.    |
| Features de nó zeradas      | ⚠️ Lacuna          | Adicionar features fundamentais (6 min.)      |
| Staleness mitigation        | ⚠️ Não documentado | Documentar e implementar proxy de atualização |
| Tipos de evento c/ features | ➕ Falta spec      | Definir vetor por tipo de evento              |
| Readout global e_t          | ➕ Falta spec      | Definir pooling ou atenção global             |
| Sparsificação do grafo      | ➕ Falta spec      | Usar TMFG ou limiar DCC adaptativo            |

# 1\. Visão Geral do Módulo

O DyFO (Dynamic Financial Ontology) é o Módulo 2 do MATTS v4.0. Sua responsabilidade é única e bem-delimitada: receber um fluxo de eventos financeiros com timestamps e produzir, a cada passo de decisão t, um vetor e_t ∈ R^d que representa as relações dinâmicas entre ativos do portfólio.

O DyFO é classificado como MODULE no FDAM: sem política aprendida em loop de recompensa, sem interação direta com o ambiente financeiro, stateless entre episódios (memória zerada no início de cada epísodo de treinamento).

| **Propriedade**       | **Valor**                                               |
| --------------------- | ------------------------------------------------------- |
| Tipo FDAM             | MODULE (não-agente)                                     |
| Entrada               | Grafo G = (V, E), stream de eventos e_i(t)              |
| Saída                 | Embedding e_t ∈ R^d por passo de decisão                |
| Arquitetura interna   | TGN-attn: memória GRU + 1 camada GAT                    |
| Dimensão de memória   | 172 (alinhado com features de aresta LIWC)              |
| Dimensão de embedding | 100 (padrão MATTS)                                      |
| Ablation associado    | B16: TGN vs. ROLAND vs. GAT estático                    |
| Hipótese associada    | H4: Sharpe condicional TGN ≥ ROLAND em ≥70% das janelas |
| Stack                 | PyTorch 2.x + PyTorch Geometric (PyG) 2.x               |

## 1.1 Posição na Arquitetura MATTS

O DyFO opera entre a entrada bruta de dados de mercado e o State Constructor. Seu output e_t é um dos cinco componentes do estado aumentado s_t:

s_t = \[ e_t | pi_t | H(pi_t) | alpha_signals_t | x_t \]

e_t ocupa as primeiras d=100 dimensões de s_t. O orquestrador e todos os sub-agentes consomem s_t sem saber qual parte veio do DyFO - o acoplamento é apenas via State Constructor.

# 2\. Design do Grafo Financeiro

## 2.1 Nós

Cada ativo do portfólio é um nó. O universo de nós é fixo por dataset durante o treinamento (inductive setting: novos ativos podem ser adicionados sem re-treino, pois o TGN suporta nós não vistos via memória zero).

| **Dataset**           | **Nós (ativos)**      | **Tipo**        |
| --------------------- | --------------------- | --------------- |
| S&P 500 constituintes | ~500 (filtrado top K) | Ações           |
| MSCI World ETFs       | ~50 ETFs              | ETF             |
| Commodities futures   | ~30 contratos         | Futuros         |
| Cripto top-20         | 20 tokens             | Cripto          |
| Fama-French 5 fatores | 5 portfolios de fator | Fator sintético |

## 2.2 Features de Nó (v_i) - Atualização Recomendada

_A proposta original usa v_i = 0 (vetor zero). A literatura 2024-2025 demonstra ganhos ao incluir features fundamentais. Recomenda-se o seguinte vetor mínimo de 8 dimensões:_

| **Feature**     | **Descrição**                              | **Fonte**           | **Dim.** |
| --------------- | ------------------------------------------ | ------------------- | -------- |
| retorno_log_21d | Retorno log acumulado 21 dias              | Preço diário        | 1        |
| vol_hist_21d    | Volatilidade histórica 21 dias             | Preço diário        | 1        |
| beta_mercado    | Beta vs. índice de referência (janela 63d) | Preço diário        | 1        |
| setor_one_hot   | Codificação GICS Sector                    | Metadados estáticos | 11       |
| market_cap_norm | Log(market cap) normalizado                | Fundamental diário  | 1        |
| drawdown_atual  | Drawdown corrente desde última máxima      | Preço diário        | 1        |
| regime_prob     | pi_t do RDM (regime atual)                 | Saída Módulo 1      | K        |
| volume_norm     | Volume rel. à média 21d (liquidez proxy)   | Volume diário       | 1        |

_regime_prob é o acoplamento formal com o Módulo 1 (RDM). Ao injetar pi_t como feature de nó, o TGN pode aprender padrões de correlação condicionados no regime durante a agregação - uma contribuição original do MATTS que nenhum trabalho 2024-2025 implementou._

## 2.3 Tipos de Aresta (Grafo Heterogêneo)

O DyFO implementa um grafo heterogêneo com 4 tipos de aresta. Cada tipo tem features distintas e contribui com informação semanticamente diferente para a memória dos nós:

| **Tipo**               | **Código** | **Construção**                           | **Feature dim.**         | **Frequência** |
| ---------------------- | ---------- | ---------------------------------------- | ------------------------ | -------------- |
| Correlação dinâmica    | CORR       | DCC-GARCH rho\_{ij}(t)                   | 3 (rho, CI_low, CI_high) | Diária         |
| Setor compartilhado    | SECT       | GICS Sector == GICS Sector               | 1 (binária)              | Estática       |
| Cadeia de fornecimento | SUPL       | FactSet Supply Chain (ou OpenCorporates) | 1 (força do vínculo)     | Trimestral     |
| Co-movimento de fator  | FACT       | \|loading FF5_i - FF5_j\| < threshold    | 5 (loadings)             | Diária         |

_Para sparsificação das arestas CORR, usar TMFG (Triangulated Maximally Filtered Graph) ou manter apenas arestas com |rho| > 0.3 e p-value < 0.05. Para universos de 500 ativos, um grafo CORR totalmente conectado tem 124.750 arestas - use sparsificação._

## 2.4 Tipos de Evento (stream de eventos)

Um evento é qualquer ocorrência que deve atualizar imediatamente a memória de um ou dois nós. A seguir o catálogo mínimo de eventos do DyFO:

| **Tipo**           | **Nós afetados**   | **Trigger**                    | **Feature vector f_e**                                 |
| ------------------ | ------------------ | ------------------------------ | ------------------------------------------------------ |
| PRICE_UPDATE       | 1 nó (ativo)       | Fechamento diário              | \[delta_ret, vol_1d, volume_norm\]                     |
| EARNINGS_REPORT    | 1 nó (empresa)     | Data de divulgação             | \[surprise_EPS, revenue_beat, guidance_delta\]         |
| FED_DECISION       | Todos os nós       | Decisão FOMC                   | \[delta_rate, dot_plot_revision, statement_sentiment\] |
| CREDIT_DOWNGRADE   | 1 nó (emissor)     | S&P/Moody's/Fitch notch change | \[notch_delta, outlook_code, sector_contagion\]        |
| CORP_ACTION        | 1 nó (empresa)     | M&A, spin-off, split           | \[event_type_code, deal_value_norm, premium\]          |
| CORRELATION_UPDATE | 2 nós (par)        | Re-estimação DCC-GARCH diária  | \[rho_new, delta_rho, significance\]                   |
| MACRO_RELEASE      | K nós (regime-dep) | CPI, NFP, PMI                  | \[surprise, revision, volatility_impact\]              |

_Atenção à hierarquia de impacto: FED_DECISION afeta TODOS os nós simultaneamente. Isso cria um batch com N eventos no mesmo timestamp. Use o agregador de mensagens 'mean' (não 'last') para esses eventos sistematicamente sincronizados._

# 3\. Arquitetura Interna do TGN no DyFO

## 3.1 Pipeline Completo

O pipeline do DyFO processa um stream de eventos e produz, ao final de cada passo de decisão (tipicamente cada dia útil), o vetor e_t. O pipeline tem 6 estágios:

- Ingestion: eventos do dia são ordenados por timestamp e enfileirados
- Message Function: para cada evento, computa mensagem m_i(t) = msg(s_i(t-), s_j(t-), delta_t, f_e(t))
- Message Aggregation: agrega mensagens por nó dentro do batch (mean ou last)
- Memory Update: atualiza s_i(t) = GRU(m_bar_i(t), s_i(t-))
- Graph Embedding (GAT): z_i(t) = GAT(s_i(t), {s_j(t) : j em vizinhos de i}, {phi(t - t_j)})
- Global Readout: e_t = readout({z_i(t) : i em portfolio_K})

## 3.2 Message Function

A função de mensagem usa a formulação de identidade (concatenação) do paper original, com extensão para grafos heterogêneos:

m_i(t) = \[s_i(t-) || s_j(t-) || phi(delta_t) || f_e(t) || edge_type_embedding\]

phi(delta_t) = Time2Vec(delta_t) # dim=100

edge_type_embedding # dim=16, uma por tipo de aresta

_Para eventos de nó único (EARNINGS_REPORT, CREDIT_DOWNGRADE): m_i(t) = \[s_i(t-) || phi(t) || f_e(t)\]. S_j é zerado nesses casos._

## 3.3 Memory Updater

s_i(t) = GRU(m_bar_i(t), s_i(t-))

Parâmetros: GRU com hidden_size = memory_dim = 172. Uma instância de GRU compartilhada para todos os nós (weight sharing). Inicialização: s_i(0) = 0 para todos os nós.

## 3.4 Temporal Graph Attention (embedding)

Usa 1 camada de atenção temporal (TGN-attn com L=1, conforme ablation do paper original que mostra que 1 camada com memória supera 2 camadas sem memória):

h_i^(0)(t) = s_i(t) + v_i(t) # features de nó + memória

q(t) = h_i^(0)(t) || phi(0) # query = nó alvo

K = V = \[h*j^(0)(t) || e*{ij}(t_k) || phi(t - t_k)\] # vizinhos temporais

h_tilde_i(t) = MultiHeadAttention(q, K, V) # num_heads=2

z_i(t) = MLP(h_i^(0)(t) || h_tilde_i(t)) # embedding final dim=100

## 3.5 Global Readout

O readout converte os N embeddings de nó em um único vetor e_t que representa o estado do grafo completo:

| **Estratégia**         | **Fórmula**                           | **Quando usar**                                           |
| ---------------------- | ------------------------------------- | --------------------------------------------------------- |
| Mean pooling (default) | e_t = mean({z_i(t) : i em portfolio}) | Portfólios com ativos equiponderáveis                     |
| Weighted by market cap | e_t = sum(w_i \* z_i(t)) / sum(w_i)   | Portfólios ponderados por capitalização                   |
| Attention readout      | e_t = softmax(q_orch \* Z^T) \* Z     | Quando o orquestrador deve focar em subconjunto de ativos |

_Para o ablation B16, usar mean pooling como default. A comparação TGN vs. ROLAND deve usar o mesmo readout._

## 3.6 Raw Message Store e Training Strategy

O mecanismo crítico para treinar os módulos de memória via backprop:

- Ao processar batch t: buscar raw messages armazenadas de batches anteriores
- Computar mensagens, agregar, atualizar memória com essas raw messages
- Usar memória atualizada para computar embeddings e calcular perda
- Armazenar raw messages do batch atual para uso no batch t+1
- Gradiente flui: perda -> embeddings -> memória -> raw messages de t-1

_Tamanho de batch crítico: o paper original recomenda batch_size=200. Para dados financeiros diários com ~252 pregoes/ano, isso corresponde a aproximadamente 1 ano de histórico por batch. Reduza para 64-128 se os eventos forem esparsos (datasets de crise)._

# 4\. Implementação: Guia Passo a Passo

## 4.1 Dependências

\# requirements para o DyFO

torch>=2.0.0

torch_geometric>=2.4.0 # PyG com suporte TGN

torch_geometric_temporal>=0.54 # opcional: modelos adicionais

statsmodels>=0.14.0 # DCC-GARCH via arch package

arch>=6.0.0 # DCC-GARCH

pandas>=2.0.0

numpy>=1.24.0

networkx>=3.1 # opcional: visualização e TMFG

## 4.2 Estrutura de Arquivos

matts/

modules/

dyfo/

\__init_\_.py

graph_builder.py # construção e atualização do grafo

event_stream.py # parser e normalização de eventos

tgn_encoder.py # TGN core: memória, msg, agg, embedding

readout.py # readout global -> e_t

edge_features.py # DCC-GARCH, TMFG, edge type embeddings

node_features.py # features de nó (beta, vol, setor)

dyfo_module.py # interface pública do módulo

rdm/ # Módulo 1 (HMM-GAS-TVTP)

state_constructor.py # concat e_t | pi_t | H(pi_t) | alpha | x

## 4.3 Interface Pública (dyfo_module.py)

O DyFO expõe uma interface simples para o State Constructor:

class DyFOModule(nn.Module):

def \__init_\_(self, config: DyFOConfig):

\# config.memory_dim = 172

\# config.embedding_dim = 100

\# config.num_heads = 2

\# config.edge_types = \['CORR','SECT','SUPL','FACT'\]

\# config.event_types = \['PRICE_UPDATE','EARNINGS_REPORT',...\]

...

def forward(self, events: List\[FinancialEvent\]) -> torch.Tensor:

\# Recebe lista de eventos do dia

\# Retorna e_t de shape (embedding_dim,)

...

def reset_memory(self):

\# Zera memoria de todos os nos

\# Chamar no inicio de cada episodio de treino

...

def save_memory_checkpoint(self, path: str):

\# Salva estado da memoria para inferencia

...

def load_memory_checkpoint(self, path: str):

\# Carrega estado da memoria (ex: inicio do periodo de teste)

...

## 4.4 Estrutura de um FinancialEvent

@dataclass

class FinancialEvent:

event_type: str # 'PRICE_UPDATE', 'EARNINGS_REPORT', etc.

timestamp: float # Unix timestamp ou dias desde epoch

source_node: int # indice do ativo (nodo i)

target_node: int # indice do ativo j (-1 se node-only event)

edge_type: str # 'CORR', 'SECT', 'SUPL', 'FACT'

features: torch.Tensor # vetor f_e (dim variavel por tipo)

node_features: torch.Tensor # features atualizadas do nodo i

## 4.5 Construção das Arestas de Correlação (DCC-GARCH)

Este é o componente mais crítico de construção do grafo. Use a biblioteca arch:

from arch.univariate import GARCH

from arch.multivariate import DCC

def compute_dcc_correlations(returns_df, window=252):

\# returns_df: DataFrame (T x N) de retornos diários

\# Estima GARCH(1,1) por ativo

residuals = \[\]

for col in returns_df.columns:

model = GARCH(returns_df\[col\])

res = model.fit(disp='off')

residuals.append(res.std_resid)

\# Estima DCC sobre resíduos padronizados

dcc_model = DCC(pd.DataFrame(residuals).T)

dcc_fit = dcc_model.fit(disp='off')

\# Retorna matriz de correlacao dinamica Ht (T x N x N)

return dcc_fit.conditional_correlation

_Para N > 100 ativos, DCC-GARCH completo fica computacionalmente custoso. Use DCC-GARCH em pares (rolling window 63d) e depois TMFG para sparsificação. Alternativa mais rápida: Distance Correlation com janela rólante de 21d._

## 4.6 Tratamento de Staleness

Para ativos que ficam sem eventos por mais de T dias (ex: small-caps, ETFs de nícho), implementar proxy de atualização:

STALENESS_THRESHOLD = 5 # dias sem evento

def inject_staleness_proxy(asset_id, current_time, last_event_time):

\# Se ativo sem evento por > threshold dias

if current_time - last_event_time > STALENESS_THRESHOLD:

\# Injeta evento PRICE_UPDATE sintetico com features zeradas

\# Isso forca o TGN a 'olhar' para os vizinhos no GAT

synthetic_event = FinancialEvent(

event_type='PRICE_UPDATE',

timestamp=current_time,

source_node=asset_id,

target_node=-1,

features=torch.zeros(3) # sem nova informacao

)

return synthetic_event

return None

_Não injetar eventos sintéticos durante o período de teste sem injectá-los também durante o treino. Inconsistência treinamento/inferência é a causa mais comum de look-ahead bias em modelos TGN financeiros._

# 5\. Treinamento e Integração com o Pipeline MATTS

## 5.1 Objetivo de Treinamento do DyFO

O DyFO não tem objetivo de treinamento próprio (FDAM: MODULE). Seus parâmetros são otimizados via backprop pelo gradiente da perda do sistema completo (orquestrador + sub-agentes). O DyFO recebe gradientes de:

- Perda CVaR do orquestrador (via State Constructor)
- Perda dos sub-agentes híbridos (via State Constructor)

Isso significa que os pesos do TGN (GRU, atenção, MLP) são aprendidos de forma a produzir representações de grafo que maximizem o retorno ajustado ao risco do portfólio - não apenas que prevêjam links futuros. Esta é uma diferença fundamental em relação ao TGN original.

## 5.2 Pré-treino Opcional (Self-Supervised)

Para acelerar a convergência, é recomendado um estágio de pré-treino do TGN em tarefa de previsão de link (qual par de ativos terá alta correlação amanhã):

\# Pre-training em link prediction

loss_pretrain = BCE(

p(edge_ij | z_i(t), z_j(t)), # prob de alta correlacao

label_ij_t1 # correlacao > threshold amanha

)

_O pré-treino não é obrigatório mas reduz o número de episódios HARL necessários para convergência. Curriculum Learning (estágio 1) deve incluir o DyFO pré-treinado._

## 5.3 Protocolo Walk-Forward

O DyFO segue o protocolo walk-forward 60/20/20 do MATTS:

- Treino (60%): TGN aprende padrões de correlação + backprop da perda do sistema
- Validação (20%): memoria inicializada com o estado final do treino (NÃO zerada)
- Teste (20%): idem validação. Memória continua evoluindo durante o teste (design TGN)

_ERRO COMUM: zerar a memória do TGN no início do período de validação/teste. Isso invalida o design de eventos contínuos e cria um look-ahead bias implícito. A memória deve ser herdada do período anterior._

## 5.4 Hiperparâmetros do DyFO

| **Hiperparâmetro**      | **Valor padrão** | **Justificativa**                           | **Range de busca** |
| ----------------------- | ---------------- | ------------------------------------------- | ------------------ |
| memory_dim              | 172              | Alinhado com features LIWC (paper original) | 128, 172, 256      |
| embedding_dim           | 100              | Padrão MATTS                                | 64, 100, 128       |
| num_attention_heads     | 2                | Paper original TGN                          | 1, 2, 4            |
| num_neighbors           | 10               | TGN-attn default (most recent)              | 5, 10, 20          |
| time_encoding_dim       | 100              | Time2Vec padrão                             | 64, 100            |
| batch_size_events       | 200              | Trade-off velocidade/granularidade          | 64, 128, 200       |
| dropout                 | 0.1              | Padrão MATTS                                | 0.0, 0.1, 0.2      |
| staleness_threshold     | 5 dias           | Recomendação baseada em liquidez            | 3, 5, 10           |
| corr_sparsify_threshold | 0.3 (\|rho\|)    | Remover correlações espurias                | 0.2, 0.3, 0.4      |

# 6\. Ablation B16: TGN vs. ROLAND vs. GAT Estático

## 6.1 Design do Ablation

O ablation B16 isola a contribuição do TGN (eventos contínuos) versus modelos de baseline do DyFO. Os três modelos devem usar o mesmo pipeline MATTS, diferindo apenas no Módulo 2:

| **Variante**        | **Módulo 2**                        | **Granularidade**          | **Memória**         |
| ------------------- | ----------------------------------- | -------------------------- | ------------------- |
| DyFO-TGN (proposta) | TGN-attn 1L, memória GRU            | Contínua (evento-a-evento) | Por nó, persistente |
| DyFO-ROLAND         | EvolveGCN-H sobre snapshots         | Discreta (mensal)          | Não (stateless)     |
| DyFO-GAT-Static     | GAT estático sobre correlação média | Não (estático)             | Não                 |

## 6.2 Métricas do Ablation

| **Métrica**                           | **Hipótese (H4)**                                     | **Como medir**                                 |
| ------------------------------------- | ----------------------------------------------------- | ---------------------------------------------- |
| Sharpe condicional por regime         | TGN >= ROLAND em 70% das janelas                      | Walk-forward 60/20/20, separado por regime     |
| R² de previsão de regime              | TGN embedding melhora classificação de regime         | Regressao logistica pi_t ~ \[e_t, e_t-ROLAND\] |
| Latencia de atualização de correlação | TGN captura eventos Fed no dia; ROLAND espera 30 dias | Medir delta de e_t antes/apos evento Fed       |
| Drawdown maximo em crises             | TGN reduz drawdown em regimes de alta volatilidade    | CVaR-95% por período de crise identificado     |

## 6.3 Pseudo-Código do Ablation

for variant in \['TGN', 'ROLAND', 'GAT_STATIC'\]:

\# 1. Inicializa pipeline MATTS com variante do M2

matts = MATTSSystem(dyfo_variant=variant)

\# 2. Walk-forward em 5 datasets

for dataset in \['SP500', 'MSCI', 'COMM', 'CRYPTO', 'FF5'\]:

results\[variant\]\[dataset\] = \[\]

for window in walk_forward_windows(dataset, split=(0.6,0.2,0.2)):

train, val, test = window

matts.train(train, reset_memory=True)

sharpe = matts.evaluate(test, reset_memory=False) # herda memoria

regime = rdm.classify(test)

results\[variant\]\[dataset\].append((sharpe, regime))

\# 3. Calcular Sharpe condicional por regime

for regime_k in range(K):

sharpe_k = \[r\[0\] for r in results if r\[1\] == regime_k\]

print(f'{variant} Sharpe regime-{regime_k}: {mean(sharpe_k):.3f}')

# 7\. Checklist de Implementação

## 7.1 Checklist de Construção do Grafo

- \[ \] Nós definidos: um por ativo no universo
- \[ \] Features de nó implementadas (mínimo 8 dim conforme Seção 2.2)
- \[ \] regime_prob injetado como feature de nó (acoplamento M1 -> M2)
- \[ \] 4 tipos de aresta implementados: CORR, SECT, SUPL, FACT
- \[ \] DCC-GARCH implementado para arestas CORR (não usar Pearson simples)
- \[ \] TMFG ou limiar aplicado para sparsificação (|rho| > 0.3)
- \[ \] Cadeia de fornecimento carregada de fonte externa (FactSet ou OpenCorporates)

## 7.2 Checklist do TGN

- \[ \] Memória GRU: memory_dim = 172, inicialização zero
- \[ \] Message function: concatenação com Time2Vec para delta_t
- \[ \] Edge type embedding: dim=16 por tipo, aprendido
- \[ \] Agregador: 'last' (default) ou 'mean' para eventos sincronizados
- \[ \] GAT: 1 camada, 2 cabecas, 10 vizinhos mais recentes
- \[ \] Raw Message Store implementado corretamente (sem leakage)
- \[ \] Staleness proxy: injetar PRICE_UPDATE sintético após 5 dias sem evento

## 7.3 Checklist de Treinamento

- \[ \] batch_size = 200 eventos
- \[ \] Memória NÃO zerada entre treino/validação/teste (walk-forward)
- \[ \] Memória zerada no início de cada episódio de TREINO
- \[ \] Gradiente flui para GRU via Raw Message Store
- \[ \] Pré-treino self-supervised opcional implementado
- \[ \] Checkpoints de memória salvos ao final de cada fold walk-forward

## 7.4 Checklist do Ablation B16

- \[ \] Três variantes implementadas: TGN, ROLAND, GAT_STATIC
- \[ \] Mesmo pipeline MATTS para as três variantes (só M2 difere)
- \[ \] Sharpe condicional por regime calculado separadamente
- \[ \] Latencia de atualização medida para eventos Fed/FOMC
- \[ \] 500 bootstraps sobre os resultados do ablation
- \[ \] H4 testada: TGN >= ROLAND em 70% das janelas walk-forward

# 8\. Referências Diretas

| **Referência**                        | **Relância para o DyFO**                                               |
| ------------------------------------- | ---------------------------------------------------------------------- |
| Rossi et al. (2020) - TGN             | Arquitetura base: memória, msg, agg, embedding, Raw Message Store      |
| Korangi et al. (2024) - GAT Portfolio | Validação GAT em portfólio de 30 anos; usa Distance Correlation + TMFG |
| TAGN (MDPI 2026)                      | Grafo multi-escala: supply chain + co-holdings + high-freq; GAT-GRU    |
| Chen et al. (2025) - GAT-MARL         | 4 tipos de aresta heterogênea; baseline B12 do MATTS                   |
| STGAT (MDPI 2025)                     | STL decomposition + GAT para portfólio; baseline adicional             |
| GAP-TGN (ICLR 2026 Workshop)          | TGN em finanças + problema de staleness documentado + walk-forward     |
| FinDKG (2024)                         | Knowledge graph dinâmico com LLM; 12 tipos de entidade, 15 relações    |
| Engle (2002) - DCC-GARCH              | Correlações dinâmicas: alternativa robusta à correlação de Pearson     |
| Veličković et al. (2018) - GAT        | Readout com atenção por aresta interpretável                           |
| You et al. (2022) - ROLAND            | Baseline B16: snapshots mensais com EvolveGCN                          |

_- Fim do Manual -_