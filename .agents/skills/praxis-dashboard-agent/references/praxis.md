# Praxis — referência específica do dashboard

Leia junto com o núcleo genérico (`SKILL.md`). Contém os caminhos, contratos e
decisões específicos do dashboard do **Praxis** (downstream read-only do PORTA).

## Caminhos

- `workspace/dashboard/{index.html,dashboard.js,dashboard.css,blocks.json,m8_view.json}`
- `scripts/praxis/dashboard_m8.py` (provedor determinístico de `m8_view.json`)
- `scripts/praxis/check_view.py` (M8.9 change monitor: `content_hash` contratos vs view)
- `scripts/praxis/backtest_timeseries.py` + `contracts/backtest_timeseries.json`
- `contracts/backtest_report.json` (agregados por allocator/fold/overall)
- `contracts/cells_manifest.json` (lineage por célula)
- `workspace/signals/live-alerts.json` (alertas), `workspace/positions/live.json` (posições)
- `reports/PRAXIS_VIEW_CHANGES.md` (M8.9 trilha de auditoria append-only)

## Monitor de mudanças (M8.9, PM89)

Fluxo concreto de orquestração (ver núcleo genérico "Monitor de mudanças"):

1. `python scripts/praxis/check_view.py` → `0` = sync; `1` = divergência.
2. Se divergir e a mudança for legítima: `python scripts/praxis/check_view.py
   --write-trail` (registra a regeneração planejada), depois
   `python scripts/praxis/dashboard_m8.py` (regenera `m8_view.json`; **não toca
   contratos/PORTA**).
3. `python scripts/praxis/run_tests.py` → deve passar.
4. `python scripts/praxis/check_view.py --write-trail` (registra a validação
   pós-regeneração, append-only em `reports/PRAXIS_VIEW_CHANGES.md`).
5. Commit com mensagem citando `PM89`.

**Fail-closed:** divergência persistente (contrato ausente/incompatível) → não
comitar; reportar erro.

## Contratos (sempre via `load_json_contract`)

- `m8_view.json`: `{schema_version, content_hash, non_evidential, panels, gaps}`;
  panels: table, gauge, radar, status, graph, timeseries, exposure, diagnostic, risk.
- `backtest_timeseries.json`: pontos `{period_ts, w, regime_p, return, status,
  fold, gap}` por allocator; `non_evidential`.
- Painel sem fonte → `SEM FONTE` com reason + `decision_ref` (fail-closed).

## Decisões D-M8-x (resumo)

- **D-M8-1** grafo = nós allocators, arestas = distância L1 dos pesos médios.
- **D-M8-2** cores: grafo estético; painéis semânticos rotulados (sem "lucro/perda").
- **D-M8-3** grafo decorativo no MVP; interativo = futuro.
- **D-M8-4** seletor allocator+fold (P1); filtros combinados = futuro.
- **D-M8-5** radar = métricas agregadas rotuladas; exposição/diagnóstico = M8.3.
- **D-M8-6** gauge = max_drawdown; risco (drawdown corrente) = M8.3.
- **D-M8-7** P1 = tabela+gauge+radar+status; conjunto completo = M8.2/M8.3.
- **D-M8-8** sem contrato temporal no MVP; M8.2 criou `backtest_timeseries.json`.
- **D-M8-9** skill local inicialmente; promoção = M8.4.
- **D-M8-10** animação: híbrido (D) agora — eventos auditados OU `DECORATIVO/SEM
  FONTE`; M8.3 = A (procedural decorativa, sem semântica).
- **D-M8.2-1..4** e **D-M8.3-1..4**: defaults aprovados (contrato aditivo,
  interação hover/seletor, exposição/diagnóstico/risco, animação decorativa).

## Feature spec

- `.specs/features/praxis-dashboard-m8/` (spec/context/design/tasks/test_plan,
  incl. `m82_*`, `m83_*`, `m84_*`).

## Fronteiras

- PORTA read-only (PRAXIS-01/60); zero dados vivos/credenciais/ordens
  (PRAXIS-110/111/32); `non_evidential`; determinismo; fail-closed.
