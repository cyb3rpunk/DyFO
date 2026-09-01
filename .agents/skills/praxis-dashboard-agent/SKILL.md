---
name: praxis-dashboard-agent
description: >
  Orienta agentes a construir e manter dashboards com integridade científica:
  renderizar somente dados de contratos existentes, fail-closed, lineage por
  célula, non_evidential e determinismo. Use SEMPRE que a tarefa envolver o
  dashboard do Praxis (blocos, m8_view, SVG/Canvas do dashboard,
  workspace/dashboard), ou qualquer dashboard de dados de pesquisa com as
  mesmas disciplinas — mesmo que o usuário não diga "dashboard" explicitamente.
  NÃO use para apps/backends fora de pesquisa científica nem para o PORTA
  (dados/resultados).
---

# Dashboard Agent (integridade científica)

Guia para construir e manter dashboards de dados de pesquisa **sem inventar
dados**, sem rede/credenciais/ordens e sem claims financeiros. Para detalhes
específicos do projeto **Praxis** (caminhos, contratos, decisões de escopo do
dashboard), leia `references/praxis.md`.

## Regras inegociáveis

1. **Fonte primeiro.** Todo painel parte de um contrato existente carregado via
   `load_json_contract`. Se não há fonte, pare e marque `SEM FONTE` — nunca
   invente dado nem placeholder silencioso.
2. **Referência visual ≠ schema.** Vídeo/imagem é direção estética (paleta,
   densidade, composição). Cores/nós/clusters NÃO ganham significado sem fonte
   ou decisão registrada do usuário. Nunca decida "dourado = lucro" por conta
   própria.
3. **Lineage e fail-closed.** Toda célula visual carrega lineage
   (`LineageRecord`); ausência/schema incompatível/hash divergente → falha
   fechada.
4. **`non_evidential`.** Painéis derivados de backtest/simulação carregam a flag;
   proibido claim de alpha/retorno/evidência.
5. **Determinismo.** Render com os mesmos contratos produz o mesmo DOM/estado.
6. **Segurança.** Use DOM APIs (`textContent`/`createElement`) para dados de
   contrato; evite `innerHTML` com dados não canônicos (XSS).
7. **Zero rede/credenciais/ordens.** Nada de `requests/urllib/socket` nem
   endpoints de ordem.

## Fluxo de trabalho

1. Leia a spec/design/context do projeto e o manifesto de view se existir
   (para o Praxis: `references/praxis.md`).
2. Mapeie o painel pedido: visual → fonte existente → contrato → lacuna.
3. Sem fonte verificável → `SEM FONTE` com reason + `decision_ref`; se o
   significado depende de decisão não registrada, **pergunte ao usuário** antes.
4. Render determinístico (nativo, sem deps novas) e registre lineage/fail-closed.
5. Verifique com a suíte de testes do projeto.
6. Não commite com decisão pendente ou fonte ausente.

## Monitor de mudanças (M8.9, PM89-10..16)

Ao orquestrar o dashboard, verifique se a view está sincronizada com os contratos
fonte (o caminho concreto do monitor está em `references/`):

1. Rode o **change monitor** (check-view). `0` = view sincronizada; `≠0` =
   divergência (contrato mudou / hash divergente) → fail-closed.
2. Se divergir e a mudança for legítima, rode `check_view.py --write-trail` para
   registrar a regeneração planejada e depois o **gerador da view** (regenera a
   view **sem tocar contratos/PORTA**).
3. Rode a **suíte de testes** do projeto → deve passar.
4. Rode `check_view.py --write-trail` novamente para registrar a validação
   pós-regeneração na **trilha de auditoria** (append-only).
5. Commit com mensagem citando `PM89`.

**Fail-closed:** se a divergência persistir após a regeneração (ex.: contrato
fonte ausente/incompatível), **não** comite a view — reporte o erro.
**Nunca** mude um contrato fonte; a orquestração só regenera a view.

## Reuso

- `load_json_contract`/`canonical_json`/`content_hash` do projeto.
- Blocos/JS existentes (fetchJson, fail) — padrão aditivo.
- `LineageRecord`/`render_cell` para hover/status com lineage.
- Não crie script/asset auxiliar novo sem necessidade comprovada.

## Quando perguntar ao usuário

- O significado de um elemento visual não tem fonte nem decisão registrada.
- Há múltiplas alternativas válidas de escopo (ex.: criar contrato novo vs deixar
  painel fora do MVP).

## Quando recusar implementação

- Dado vivo, rede, credenciais, ordens, claim de alpha, alteração de contrato
  validado, ou dado ausente sem decisão registrada.

## Casos de avaliação (evals)

**Gatilho (deve disparar):**
- "adicionar um gauge de max_drawdown ao dashboard"
- "o dashboard não mostra o resultado do backtest"
- "seguir o estilo do vídeo no dashboard"
- "o painel de radar do dashboard está errado"

**Não deve disparar:**
- "criar um dashboard React do projeto X" (fora de pesquisa científica)
- "rodar o backtest do framework" (backend/backtest, sem UI)
- "alterar contrato do PORTA" (dados/resultados)

**Caso de uso — sem fonte:**
- Input: "o grafo temporal deveria mostrar a série por período".
- Esperado: identificar que não há contrato temporal → marcar `SEM FONTE` com
  reason + `decision_ref` e **perguntar ao usuário**; nunca inventar a série.

**Critérios de verificação dos evals:** os casos acima são verificados
localmente de forma **estrutural** (a descrição/frontmatter cobre os gatilhos
positivos e exclui os negativos; o caso "sem fonte" tem regra explícita de
fail-closed + pergunta). Execução comportamental com subagentes (com/sem a
skill) é validação contínua planejada para o futuro (orquestração M8.9).

## Critérios objetivos de sucesso

- Fonte verificável por painel; determinismo (2 renders idênticos); zero
  claim/alpha; zero placeholder silencioso; lineage presente; suíte PASS;
  `non_evidential` propagado.
