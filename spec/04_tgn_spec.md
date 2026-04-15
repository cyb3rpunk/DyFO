# 04 — Temporal Encoder Spec

> Fonte de verdade para a próxima evolução do DyFO:
> 1. `Relation-aware heterogeneous TGN` inspirado em TeSa/CTRL
> 2. `Temporal KG` como braço de ablação interpretável

---

## Objetivo desta fase

Substituir o status atual de "TGN heterogêneo com aresta tipada mínima" por uma arquitetura que:

- respeite a heterogeneidade semântica entre `CORR`, `SECT`, `SUPL`, `FACT` e eventos sistêmicos
- preserve o regime de eventos contínuos do DyFO
- mantenha o universo fixo em **30 ações**
- reutilize o protocolo de avaliação de [run_bootstrap_eval_v5.py](../scripts/run_bootstrap_eval_v5.py:1) como referência
- **não edite** o runner `v5` já validado

---

## Escopo congelado

### Universo de ativos
- O experimento permanece em **30 ações**
- Não há expansão para Nasdaq-100, S&P 500 completo ou universos >30 nesta fase
- `C(30,2) = 435` pares continua sendo a unidade de avaliação por dia

### Runner e protocolo
- [run_bootstrap_eval_v5.py](../scripts/run_bootstrap_eval_v5.py:1) é o **ponto de partida conceitual**
- O protocolo walk-forward, o binomial test de H4 e o bootstrap por janela devem ser preservados
- Cada nova variante deve ter **script próprio**
- É proibido editar o `run_bootstrap_eval_v5.py` nesta fase

### Estratégia de implementação
- Primeiro implementar **BL-17 relation-aware heterogeneous TGN**
- Depois implementar **BL-18 Temporal KG**
- O braço BL-18 é complementar e comparativo, não substituto do BL-17

---

## Problema no TGN atual

O encoder atual já processa eventos heterogêneos, mas ainda comprime demais a semântica relacional:

1. A `message function` mistura todos os tipos de relação no mesmo espaço latente.
2. A agregação é global por nó, sem separação explícita por relação.
3. A camada estrutural usa majoritariamente `edge_type_emb`, sem explorar de forma forte as features econômicas da aresta.
4. Eventos sistêmicos e eventos bilaterais compartilham o mesmo caminho de atualização.
5. O modelo é bom para predição, mas ainda fraco para auditoria causal e explicabilidade por relação.

---

## BL-17 — Relation-aware Heterogeneous TGN

### Intuição

Inspirado em TeSa/CTRL, o encoder passa a ter dois níveis de composição:

1. **Intra-relação:** agrega mensagens apenas dentro de cada relação/evento semântico
2. **Inter-relação:** funde as representações produzidas por cada relação com atenção semântica

O objetivo é impedir que `CORR`, `FACT`, `SECT`, `SUPL`, `PRICE_UPDATE`, `FED_DECISION` e `MACRO_RELEASE`
compitam no mesmo canal sem distinção.

### Nome canônico da variante

`ra_htgn`

Este id deve ser usado como `model_variant` novo, sem sobrescrever `tgn`.

### Arquitetura alvo

```
eventos do dia
   ↓
encoder de mensagem por relação/evento
   ↓
memória específica por nó
   ↓
agregação intra-relação
   ↓
atenção inter-relação / semantic fusion
   ↓
embedding temporal relation-aware
   ↓
readout global e_t
```

### Mudanças obrigatórias

#### 1. Message function relation-aware
- Separar o espaço de mensagens por grupo semântico
- Recomendação mínima:
  - `node_event`: PRICE_UPDATE, EARNINGS_REPORT, CREDIT_DOWNGRADE, CORP_ACTION
  - `system_event`: FED_DECISION, MACRO_RELEASE
  - `pair_relation`: CORRELATION_UPDATE
  - `static_relation`: SECT, SUPL, FACT no embedding estrutural
- Cada grupo deve ter pelo menos:
  - projeção própria
  - bias/normalização própria
  - embedding de tipo próprio

#### 2. Agregação intra-relação
- Para cada nó `i`, agregar separadamente:
  - `m_i^node`
  - `m_i^system`
  - `m_i^pair`
- O agregador por grupo pode ser `mean` como default
- Para `FED_DECISION`, a regra de determinismo com `mean` permanece obrigatória

#### 3. Fusão inter-relação
- Introduzir um módulo de atenção semântica:

```text
alpha_i^r = softmax(score(h_i^r))
m_i^fusion = Σ_r alpha_i^r h_i^r
```

- As atenções por relação devem ser exportáveis para análise posterior
- Essa fusão é a principal herança conceitual de TeSa

#### 4. Memória e tempo
- Manter memória por nó
- Manter `delta_t` contínuo
- Opcional desejável:
  - uma porta de intensidade/event-rate inspirada em CTRL/DyRep/Hawkes
  - prioridade para `system_event` em dias de choque macro

#### 5. Embedding estrutural relation-aware
- A camada de embedding não deve consumir apenas `edge_type_emb`
- Deve consumir também features reais de aresta quando disponíveis:
  - `CORR`: `rho`, `delta_rho`, `significance`
  - `FACT`: vetor de distância de loadings
  - `SUPL`: força do vínculo
  - `SECT`: flag binária
- Se necessário, normalizar cada família em espaço comum por projeção linear

### Contratos de implementação

#### Arquivos novos preferenciais
- `dyfo/core/relation_aware_tgn.py`
- `dyfo/core/relation_semantic_attention.py`
- `scripts/run_bootstrap_eval_ra_htgn.py`

#### Arquivos a ajustar com cuidado
- `dyfo/core/model_variants.py`
- `scripts/train_link_prediction.py`
- `dyfo/config.py`

#### Regra de compatibilidade
- O caminho `model_variant="tgn"` deve continuar intacto
- O runner v5 continua comparando apenas `tgn`, `roland`, `gat_static`
- O novo runner deve espelhar a lógica do v5 e adicionar `ra_htgn`

### Critérios de aceite BL-17

1. `ra_htgn` roda end-to-end no mesmo dataset de 30 ações
2. O protocolo walk-forward é equivalente ao v5
3. O script novo não altera resultados do v5
4. O modelo salva pesos/diagnósticos de atenção por relação
5. Há comparação contra `tgn`, `roland` e `gat_static`

---

## BL-18 — Temporal KG Ablation

### Objetivo

Criar um braço mais interpretável, no qual o mercado é descrito como uma coleção de fatos temporais,
em vez de apenas um fluxo neural de mensagens.

### Nome canônico da variante

`temporal_kg`

### Formalização alvo

Representar conhecimento temporal como quádruplas:

```text
(head, relation, tail, timestamp)
```

ou quíntuplas quando houver atributos do fato:

```text
(head, relation, tail, timestamp, attributes)
```

### Mapeamento inicial do domínio financeiro

- `(AAPL, in_sector, Technology, t)`
- `(AAPL, correlated_with, MSFT, t, {rho, delta_rho})`
- `(XOM, exposed_to_macro, fed_funds_rate, t, {surprise_z, change})`
- `(JPM, affected_by, earnings_report, t, {surprise_eps})`
- `(NVDA, similar_factor_profile, AVGO, t, {ff5_distance})`

### Design do braço TKG

#### Entidades
- ativos
- setores
- fatores macro
- tipos de evento canônicos

#### Relações mínimas
- `correlated_with`
- `in_sector`
- `supply_link`
- `similar_factor_profile`
- `affected_by_event`
- `exposed_to_macro`

#### Tarefas do braço
- forecast de fatos temporais
- score de plausibilidade de arestas futuras
- geração de trilhas interpretáveis por relação

### Estratégia de modelagem

Não tentar reproduzir toda a literatura TKG de uma vez.
O braço BL-18 deve começar simples:

1. conversão do event stream para fatos temporais
2. baseline neural temporal simples
3. export de explicações por relação e timestamp

Modelos elegíveis:
- RE-Net style autoregressivo
- encoder recorrente relacional simples
- rule-enhanced temporal scorer

### Arquivos novos preferenciais
- `dyfo/core/temporal_kg.py`
- `dyfo/core/temporal_kg_adapter.py`
- `scripts/run_bootstrap_eval_temporal_kg.py`

### Regra de isolamento
- O braço `temporal_kg` não pode degradar nem substituir `tgn` ou `ra_htgn`
- O pipeline TKG deve ter serialização e avaliação próprias

### Critérios de aceite BL-18

1. Conversão determinística de eventos DyFO para fatos temporais
2. Runner novo inspirado no v5, sem editar o arquivo atual
3. Métricas preditivas comparáveis ao pipeline existente
4. Saída interpretável por relação, entidade e timestamp

---

## Comparação esperada entre braços

| Variante | Força principal | Fraqueza principal | Papel no paper |
|---------|------------------|--------------------|----------------|
| `tgn` | baseline validado | pouca semântica relacional | baseline histórico |
| `ra_htgn` | melhor predição relacional | maior complexidade | candidato principal |
| `temporal_kg` | melhor interpretabilidade | menor flexibilidade neural | ablação interpretável |

---

## Avaliação experimental

### Regra geral

Todos os braços novos devem preservar a filosofia do `v5`:

- múltiplas janelas walk-forward
- H4 em nível de janela
- bootstrap por janela
- testes preditivos sem pseudorreplicação

### Scripts previstos

- `scripts/run_bootstrap_eval_v5.py`
  - permanece congelado
  - referência do protocolo
- `scripts/run_bootstrap_eval_ra_htgn.py`
  - replica a lógica do v5 e adiciona `ra_htgn`
- `scripts/run_bootstrap_eval_temporal_kg.py`
  - replica a lógica do v5 para o braço interpretável

### Comparações mínimas

#### BL-17
- `ra_htgn` vs `tgn`
- `ra_htgn` vs `roland`
- `ra_htgn` vs `gat_static`

#### BL-18
- `temporal_kg` vs `tgn`
- `temporal_kg` vs `ra_htgn`

### Métricas mínimas
- `r_squared`
- `spearman`
- `mae`
- `sharpe_proxy`
- win-rate por janela
- atenção semântica por relação para `ra_htgn`
- trilhas/fatos explicativos para `temporal_kg`

---

## Ordem recomendada de implementação

1. Adicionar `model_variant="ra_htgn"` sem tocar no caminho atual `tgn`
2. Extrair blocos reutilizáveis do protocolo v5 para um runner novo ou duplicado
3. Rodar smoke test com 1 janela
4. Rodar walk-forward curto
5. Congelar BL-17
6. Implementar conversor `event stream -> temporal facts`
7. Adicionar `model_variant="temporal_kg"` em pipeline isolado
8. Rodar ablação interpretável

---

## Não objetivos desta fase

- aumentar o universo além de 30 ações
- substituir a hipótese H4 por outro endpoint primário
- refatorar o runner v5 validado
- unificar BL-17 e BL-18 em um único modelo monolítico

---

## Definição de pronto

O SDD desta fase está satisfeito quando:

1. existe uma implementação `ra_htgn` isolada do `tgn`
2. existe um runner novo derivado conceitualmente do `v5`
3. existe um braço `temporal_kg` para ablação interpretável
4. o universo continua em 30 ações
5. o `run_bootstrap_eval_v5.py` não foi editado
