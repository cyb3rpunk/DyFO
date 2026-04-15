# 03 — Catálogo de Eventos

> Especificação completa de todos os tipos de evento, vetores de features e regras de disparo.
> Esta é a fonte de verdade para implementação e extensão do event stream.

---

## Estrutura de um Evento

```python
@dataclass
class FinancialEvent:
    timestamp: pd.Timestamp      # UTC, tz-aware OBRIGATÓRIO
    event_type: str              # ver catálogo abaixo
    src_node: int                # índice do nó fonte
    dst_node: int | None         # índice do nó destino (None para eventos de 1 nó)
    edge_type: str | None        # tipo de aresta ("CORR", "SECT", "SUPL", "FACT", None)
    features: torch.Tensor       # vetor de features específico do tipo
```

**Regra de timestamp:** todos os timestamps devem ser `tz-aware` (UTC).
Usar `pd.Timestamp(..., tz='UTC')` ou `.tz_localize('UTC')`.
**Nunca** comparar `tz-naive` com `tz-aware` — causa `TypeError` silencioso.

---

## Catálogo de Eventos

### PRICE_UPDATE
```
Trigger:    Fechamento diário de mercado
Nós:        1 nó (o ativo)
Edge type:  None
Features:   [delta_ret, vol_1d, volume_norm]   dim=3
```
- `delta_ret`: retorno logarítmico do dia (log(P_t / P_{t-1}))
- `vol_1d`: volatilidade intraday estimada (high-low range normalizado)
- `volume_norm`: volume relativo à média 21d (0=mínimo, 1=máximo do período)

**Frequência:** ~252 por ativo por ano. Evento mais frequente do sistema.

---

### EARNINGS_REPORT
```
Trigger:    Data de divulgação de resultados trimestrais
Nós:        1 nó (a empresa)
Edge type:  None
Features:   [surprise_EPS, revenue_beat, guidance_delta]   dim=3
```
- `surprise_EPS`: (EPS_real - EPS_estimado) / |EPS_estimado|, clampado em [-3, 3]
- `revenue_beat`: (Receita_real - Receita_estimada) / |Receita_estimada|
- `guidance_delta`: revisão do guidance (positivo=up, negativo=down, 0=neutro/ausente)

**Fonte:** `yfinance_adapter.get_earnings_dates()` — ~4 por ativo por ano.

---

### FED_DECISION
```
Trigger:    Decisão do FOMC (Federal Reserve)
Nós:        TODOS os N nós (evento sistêmico)
Edge type:  None
Features:   [delta_rate, dot_plot_revision, statement_sentiment]   dim=3
```
- `delta_rate`: mudança em basis points (ex: +25 → 0.25, sem mudança → 0.0)
- `dot_plot_revision`: revisão da mediana do dot plot para o ano corrente
- `statement_sentiment`: score de sentimento do comunicado (-1=hawkish, +1=dovish)

**REGRA CRÍTICA:** Este evento cria N mensagens no mesmo timestamp.
O agregador **deve ser `mean`** para garantir determinismo.

**Frequência:** ~8 decisões por ano.

---

### CREDIT_DOWNGRADE
```
Trigger:    Rebaixamento de rating por S&P, Moody's ou Fitch
Nós:        1 nó (o emissor)
Edge type:  None
Features:   [notch_delta, outlook_code, sector_contagion]   dim=3
```
- `notch_delta`: número de notches de rebaixamento (negativo) ou upgrade (positivo)
- `outlook_code`: -1=negativo, 0=estável, +1=positivo
- `sector_contagion`: fração de ativos do mesmo setor no portfólio (proxy de contágio)

---

### CORP_ACTION
```
Trigger:    M&A, spin-off, split, dividendo especial
Nós:        1 nó (a empresa)
Edge type:  None
Features:   [event_type_code, deal_value_norm, premium]   dim=3
```
- `event_type_code`: 1=M&A, 2=spin-off, 3=split, 4=dividendo_especial
- `deal_value_norm`: valor do deal / market_cap do ativo (0 se não aplicável)
- `premium`: prêmio pago sobre o preço de mercado (0 se não aplicável)

**Fonte:** `yfinance_adapter.get_corporate_actions()`.

---

### CORRELATION_UPDATE
```
Trigger:    Re-estimação diária do DCC-GARCH
Nós:        2 nós (par de ativos)
Edge type:  "CORR"
Features:   [rho_new, delta_rho, significance]   dim=3
```
- `rho_new`: nova correlação DCC-GARCH ρ_ij(t)
- `delta_rho`: variação em relação ao dia anterior (ρ_ij(t) - ρ_ij(t-1))
- `significance`: 1 se |ρ| > threshold e p-value < 0.05, 0 caso contrário

**Frequência:** C(N,2) eventos por dia útil (435 para 30 ativos).

---

### MACRO_RELEASE
```
Trigger:    Divulgação de indicador macroeconômico (CPI, NFP, PMI, GDP)
Nós:        K nós (dependente do regime — ativos mais sensíveis ao macro)
Edge type:  None
Features:   [surprise, revision, volatility_impact]   dim=3
```
- `surprise`: (realizado - esperado) / desvio_padrão_histórico
- `revision`: revisão do dado anterior (0 se não houver)
- `volatility_impact`: percentual de aumento do VIX nas 2h após o release

**Fonte:** `fred_adapter.py` — séries FRED com timestamps de divulgação.

---

## Hierarquia de Impacto

```
FED_DECISION    → todos os N nós    (sistêmico máximo)
MACRO_RELEASE   → K nós selecionados (sistêmico parcial)
CREDIT_DOWNGRADE → 1 nó + contágio setorial
EARNINGS_REPORT  → 1 nó
CORP_ACTION      → 1 nó
CORRELATION_UPDATE → par (i,j)
PRICE_UPDATE     → 1 nó (rotineiro)
```

---

## Regras de Implementação

1. **Ordering:** eventos no mesmo timestamp devem ser ordenados por hierarquia de impacto
   (FED > MACRO > CREDIT > EARNINGS > CORP > CORR > PRICE)
2. **tz-awareness:** todo timestamp deve ser UTC tz-aware — usar `.tz_localize('UTC')` se necessário
3. **Normalização:** todos os features devem ser normalizados **antes** de entrar no modelo
   (StandardScaler fit no train set, transform em val/test)
4. **Missing data:** features ausentes → 0.0 (não NaN, não remover o evento)
5. **Batch FED:** ao processar FED_DECISION, criar N eventos paralelos e agregar com `mean`

---

## Extensão: adicionando novo tipo de evento

Checklist:
- [ ] Definir trigger, nós afetados, edge_type e vetor de features (dim fixo)
- [ ] Adicionar ao `DyFOConfig.event_types` em [dyfo/config.py](../dyfo/config.py)
- [ ] Implementar coleta no adapter correspondente em [dyfo/data/](../dyfo/data/)
- [ ] Adicionar ao catálogo acima
- [ ] Adicionar teste em [tests/test_smoke.py](../tests/test_smoke.py)
