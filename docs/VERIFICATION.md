# DyFO — Verificação Científica (VERIFICATION)

> **Wave 3 — passe de verificação & revisão científica** sobre os 6 documentos Wave-2
> (`docs/{OVERVIEW,ARCHITECTURE,DATA,MODELS,EXPERIMENTS,TESTING}.md`) e o relatório de
> discrepâncias Wave-2 (`DOCS_MASTER/_orchestration/report_wave2_DyFO.md`).
> Metodologia: *implementation-verification-pass*. Este arquivo é o **único** escrito nesta
> onda; nenhum código, `.specs`, `doc/`, `results/` ou os 6 docs Wave-2 foi modificado.
> Toda afirmação abaixo foi checada contra `caminho:linha` de código ou contra o artefato
> bruto (JSON/txt), citados inline. Data da auditoria: 2026-07-08.

---

## 1. Veredito-resumo

**As 6 docs Wave-2 são de alta fidelidade.** Das 18 discrepâncias catalogadas, **18/18 são
CONFIRMADAS** contra código/artefatos (nenhuma REFUTADA; 2 recebem ajuste de ênfase). Todos
os números-manchete (0.824, 2.615, p=0.0018, p<0.0001, 5/9, 4/9, EWMA 0.9905 > TGAT 0.8871,
scaling 0.7674/0.8972/0.8943, probe −5.35/−4.36) foram reproduzidos **exatamente** a partir
dos artefatos brutos. A separação entre R² de link prediction e o null do forecast de ρ é
mantida corretamente, e as setas DyFO→ORION (BL-09/DI-1) estão corretamente marcadas como
PLANEJADAS.

Esta onda **acrescenta 5 achados NOVOS** que as docs Wave-2 não registraram — o mais
importante (**N1**) é um **look-ahead na construção dos alvos DCC-GARCH** que qualifica a
interpretação de *todos* os R² absolutos (0.824 e 0.99 inclusive).

**Status geral:** `verified` com `caveat` (N1) e uma **must-fix de segurança** (chave FRED
versionada — item 17/§6).

**Top 3 por severidade:**
1. **CRÍTICA (ciência):** "H4 CONFIRMADA (p=0.0018)" (`STATE.md:3`) é cientificamente
   **insustentável como afirmação corrente** — conflaciona um resultado TGN v0.9,
   single-split, não-replicado, com a hipótese TGAT de ≥70% de janelas walk-forward (§4).
2. **ALTA (segurança):** chave FRED hardcoded (`config.py:91`) está **versionada e no
   histórico git** desde o commit `1add91f` (§6) — vazamento real de credencial.
3. **ALTA (metodologia, NOVO):** alvos DCC-GARCH ajustados no **sample inteiro** (§7, N1) →
   R² absolutos não são estritamente out-of-sample.

---

## 2. Tabela de adjudicação (18 discrepâncias)

| # | Discrepância | Veredito | Sev. | Evidência (checada nesta onda) |
|---|---|---|---|---|
| 1 | H4 p-value: dois testes distintos, ambos era TGN | **CONFIRMADA** | Crítica | `doc/EXPERIMENT_LOG.md:512` (p=0.0018, block bootstrap 10k/blocos-5d, **TGN vs ROLAND** Sharpe GMVP, 30 ativos, v0.9) e `:543-546` (p<0.0001, Wilcoxon+DM+Holm, **TGN** vs baselines, **erro preditivo**). `ROADMAP.md:81` rotula "(TGAT > Baselines)" = erro de atribuição (modelo e métrica) |
| 2 | H4 financeira não replicou | **CONFIRMADA** | Crítica | `EXPERIMENT_LOG.md:557-559` (v1.0 Sharpe P=0.595, CVaR P=0.289, n.s.); era TGAT: `rev2_20260418_130703` `primary_comparison` = 5/9, Wilcoxon **p=0.71484**; `rev3_20260420_141237` = 4/9, **p=0.58984**. Critério ≥70% nunca atingido (máx 55,6%) |
| 3 | rho_conditioning sob rótulo "tgat" | **CONFIRMADA** | Alta | `run_spy_vix_covid_compare.py:235` `use_rho_conditioning=True` sob `model_variant="tgat"` (`:227`), imprime "TGAT test metrics" (`:238`) sem qualificar; JSONs de stress não registram o flag. Default False em `link_prediction.py:237`, `train_link_prediction.py:281`, runner rev3 (não passa). Inventário de contaminação em §5 |
| 4 | BL-18: código confirma ROADMAP, STATE obsoleto | **CONFIRMADA** | Baixa | `temporal_kg.py` (395 l.), `temporal_kg_adapter.py` (197 l.), `run_bootstrap_eval_temporal_kg.py` (445 l.), `tests/test_temporal_kg.py`. `STATE.md:91-97` = checklist vazio "aguardando BL-17" (obsoleto). Ressalva: temporal_kg ausente de `metrics_by_variant` em todos os sumários auditados; só link_pred (melhor R²=0.7124) |
| 5 | Runner canônico = rev3 | **CONFIRMADA** | Baixa | `.specs/codebase/TESTING.md:12` (rev2 "✅ Ativo", doc de 2026-04-19) vs `ROADMAP.md:68` (rev3). Só rev3 tem `--seeds` (`:950-956`); artefatos rev3 até 05-02. Header do rev3 ainda diz "Revision 2" (`:1`, copiado do rev2) |
| 6 | "10k bootstrap, block_size=10" sem lastro | **CONFIRMADA** | Média | Reivindicado em `.specs/codebase/TESTING.md:45-46` (e `:4,:58`). Defaults reais 2000/5 (`rev3:114-115`, `rev2:111-112`); artefatos 2000-5000/5 (`run_config`). Único 10k real = v0.9 single-split, **blocos de 5** (`EXPERIMENT_LOG.md:487`), não 10 |
| 7 | Manchete v1.0 = composto de runs distintos | **CONFIRMADA** | Alta | R²=**0.8240663912452635** em `link_pred_tgat_s42_20260421_032602/results.json` (50 tk) — cujo **próprio** Sharpe=1.5357, MDD=5.55%, turnover=0.1230. Sharpe≈2.615 = média de 2 runs **30 tk** (2.618114 + 2.611735)/2 = 2.6149, R²≈0.76. MDD 12.4% e turnover 0.085 **inexistentes** em qualquer artefato |
| 8 | EWMA supera TGAT no harness canônico | **CONFIRMADA** | Alta | `rev3_20260501_200449` `descriptive_summary.mean_window_metrics`: ewma R²=**0.99048** > tgat **0.88706** ≈ persistence **0.87832**. Mesma tarefa/alvo (`r_squared` nas mesmas 9 janelas, esquema idêntico) → **enquadramento Wave-2 é justo** (§4.b) |
| 9 | BL-22 "50 ótimo" sem suporte estatístico | **CONFIRMADA** | Média | `scaling_summary.json`: Wilcoxon **NaN**, DM **p=1.0** (runs únicos); R² 50 (0.89719) ≈ 100 (0.89429); em 100tk `rev3_20260501_190349` tgat R²=**−4.4728**. Nota extra: no Sharpe o 30tk é o maior (2.198) — "50 ótimo" só vale para R² |
| 10 | TMFG não implementado (BL-26) | **CONFIRMADA** | Média | `ticker_registry` rotula 100→"tmfg"; runner emite warning "uses threshold — results may differ from spec" (`rev3:748-753`) |
| 11 | 5 seeds anunciados, máx 3 observados | **CONFIRMADA** | Média | `abl_full_tgat_20260421_114006` `run_config.seeds=[42,123,456]`; nenhum dir `s789`/`s2024`. `MULTISEED_SEEDS=[42,123,456,789,2024]` definido (`rev3:118`) mas default `[42]` (`:117`). (Existem dirs s43/s44 avulsos, não os anunciados) |
| 12 | BL-27: checkpoints pré-fix incompatíveis; 0.824 pré/contemporâneo ao fix | **CONFIRMADA** | Alta | `edge_dim=self._et_dim` (`tgat_encoder.py:255`), `edge_attr` (`:379`); `STATE.md:18-19` "todos os experimentos precisam ser re-rodados"; TASK.md 2026-04-21; run do 0.824 = `20260421_032602` (mesma data). BL-30 re-ablação pendente |
| 13 | dyfo_module.py é específico do TGN | **CONFIRMADA** | Baixa | `dyfo_module.py:24` `import TGNEncoder`, `:58` `self.encoder = TGNEncoder(...)`; staleness só no caminho TGN |
| 14 | SUPL sempre vazio | **CONFIRMADA** | Baixa | Código ok (`edge_features.py:362-383`); sem CSV em `data/`; `train_link_prediction.py:191` `supply_chain_edges=[]`. Lacuna de dados (BL-10) |
| 15 | Docstrings desatualizadas | **CONFIRMADA** | Baixa | `node_features.py:3` "18-dim" vs fórmula/código 20 (`:6,:8`); `tgat_encoder.py:29` e comentário `:215` "20" vs `num_neighbors=10` (`config.py:34`) |
| 16 | config default "tgn" vs primário tgat | **CONFIRMADA** | Baixa | `config.py:24` `model_variant: str = "tgn"` |
| 17 | FRED key hardcoded | **CONFIRMADA (+histórico git)** | Alta | `config.py:91` = `"7a786abc97ebd22946d8763e4d9130bf"`; introduzida no commit **`1add91f`** (2026-04-13, antes era `""`); `dyfo/config.py` **é rastreado** → chave permanente no histórico. §6 |
| 18 | ra_htgn: avaliação instável/escassa | **CONFIRMADA** | Baixa | 2 runs: R²=0.75752 (`s42_20260421_110155`) e **−0.75640** (`s42_20260421_112649`); ausente do protocolo bootstrap |

**Ajustes de ênfase (não rebaixamentos):** item 7 e item 8 são tratados como severidade
**Alta** por serem os que mais afetam a leitura científica da manchete/tese (Wave-2 já os
descreve corretamente; apenas elevamos a severidade). Nenhum item foi REFUTADO.

---

## 3. Auditoria de citações (28 conferidas; requisito ≥15)

Todas reproduzidas do artefato/linha **nesta onda**:

| # | Afirmação | Fonte | OK |
|---|---|---|---|
| 1 | `embedding_dim=100` | `config.py:30` | ✅ |
| 2 | `model_variant="tgn"` default | `config.py:24` | ✅ |
| 3 | `node_feature_dim=20` | `config.py:68` | ✅ |
| 4 | FRED key hardcoded (32 hex) | `config.py:91` | ✅ |
| 5 | 8 séries FRED (DFF…MANEMP) | `config.py:92-103` | ✅ |
| 6 | **R²=0.824** exato = 0.8240663912452635 | `link_pred_tgat_s42_20260421_032602/results.json` | ✅ |
| 7 | mesmo run: Sharpe 1.5357/MDD 5.55%/turnover 0.1230 | idem (`metrics.*`) | ✅ |
| 8 | Sharpe 2.618114 (30 tk, R²=0.7597) | `link_pred_tgat_s42_20260416_171937/results.json` | ✅ |
| 9 | Sharpe 2.611735 (30 tk, R²=0.7576) | `link_pred_tgat_s42_20260417_090713/results.json` | ✅ |
| 10 | ewma 0.99048 / tgat 0.88706 / pers 0.87832 | `rev3_20260501_200449/…summary.json` | ✅ |
| 11 | p=0.0018 (TGN vs ROLAND Sharpe, v0.9) | `EXPERIMENT_LOG.md:512` | ✅ |
| 12 | p<0.0001 (TGN preditivo, v1.0) | `EXPERIMENT_LOG.md:543-546` | ✅ |
| 13 | Sharpe P=0.595 / CVaR 0.289 (v1.0 n.s.) | `EXPERIMENT_LOG.md:557-559` | ✅ |
| 14 | tgat vs tgn 5/9, Wilcoxon p=0.71484 | `rev2_20260418_130703` `primary_comparison` | ✅ |
| 15 | tgat vs tgn 4/9, p=0.58984 | `rev3_20260420_141237` `primary_comparison` | ✅ |
| 16 | probe CALM/BREAK −5.3515 / −4.3561; pers/EWMA 0.9901/0.9894 | `results/probe_results.txt` | ✅ |
| 17 | scaling 0.7674/0.8972/0.8943; Wilcoxon NaN, DM 1.0 | `scaling_summary.json` | ✅ |
| 18 | 100tk colapso tgat R²=−4.4728 | `rev3_20260501_190349/…summary.json` | ✅ |
| 19 | edge_dim fix (BL-27) | `tgat_encoder.py:255,379` | ✅ |
| 20 | SUPL vazio | `train_link_prediction.py:191` | ✅ |
| 21 | `--seeds` help "42 123 456 789 2024" | `rev3:950-956` | ✅ |
| 22 | rho_conditioning True sob "tgat" | `run_spy_vix_covid_compare.py:227,235,238` | ✅ |
| 23 | probe `USE_RHO_CONDITIONING=False` | `run_diagnostic_probe.py:19-20` | ✅ |
| 24 | defaults bootstrap 2000/5 | `rev3:114-115` | ✅ |
| 25 | 32 testes coletados | `pytest --collect-only` (2.83s) | ✅ |
| 26 | ORION: M6 ACTIVE; `graph_embedding=None`; L-1 | `ORION ROADMAP.md:29-31`, `constructor.py:62,74`, `docs/MODELS.md:144` | ✅ |
| 27 | ra_htgn 0.7575/−0.7564; temporal_kg best 0.7124 | `link_pred_ra_htgn_s42_*`, `link_pred_temporal_kg_s42_*` | ✅ |
| 28 | `.specs`/`results`/`doc` ignorados; 3 scripts probe untracked; sem `.github/` | `git check-ignore`, `git ls-files`, `git status` | ✅ |

Nenhuma citação Wave-2 falhou na verificação. Pequenos desvios de linha (a doc cita
faixas como `:249-256`, `:747-753`; a linha exata é `:255`, `:748`) estão **dentro** das
faixas citadas — sem impacto.

---

## 4. H4 — deep-dive (itens 1-2, 8) e formulação honesta recomendada

### 4.a A pergunta central: "H4 CONFIRMADA (p=0.0018)" (`STATE.md:3`) é sustentável?

**Não como afirmação corrente.** O p=0.0018 falha em **quatro** dimensões simultâneas em
relação à H4 que o projeto de fato pré-registrou (`PROJECT.md:49-51`: *"Sharpe e MDD do
**TGAT** superiores aos baselines em **≥70% das janelas walk-forward**"*):

1. **Modelo errado.** p=0.0018 é **TGN vs ROLAND** (`EXPERIMENT_LOG.md:499,512`), não TGAT.
   O TGN foi depois **rebaixado a baseline** (`ROADMAP.md:66`).
2. **Protocolo errado.** v0.9 **abandonou** o walk-forward multi-janela por custo/rede e
   **pivotou para bootstrap de bloco numa única split 60/20/20**
   (`EXPERIMENT_LOG.md:470-472,480`). Ou seja, o p=0.0018 **não** mede "≥70% das janelas" —
   mede P(TGN≤ROLAND) numa única janela reamostrada.
3. **Não replicou.** A validação robusta v1.0 (mesmo TGN, Wilcoxon+DM+Holm) deu **Sharpe
   P=0.595** e **CVaR P=0.289** — ambos n.s. (`EXPERIMENT_LOG.md:557-559`), com conclusão
   explícita "mais acurácia preditiva não garante maior Sharpe" (`:562`).
4. **Na era TGAT** (o modelo que H4 realmente nomeia) o win rate máximo é **5/9 ≈ 55,6%**
   (Wilcoxon p=0.715), depois **4/9** (p=0.590) — **nunca** ≥70%.

O que o p=0.0018 tem a favor: é um resultado real, corretamente reproduzido, de que **numa
janela específica** o TGN de correlação bateu o ROLAND em Sharpe GMVP com forte
significância de bootstrap. Isso não é fraude — é um resultado **exploratório, de outra era
de modelo e outro protocolo**, apresentado como se fosse a confirmação da hipótese
financeira TGAT.

### 4.b Item 8 — EWMA > TGAT é comparação justa?

**Sim, é a mesma tarefa.** Em `rev3_20260501_200449` as três variantes (persistence, ewma,
tgat) são avaliadas no **mesmo alvo** (`r_squared`), nas **mesmas 9 janelas**, com esquema
de métricas idêntico (`loss, mse, mae, r_squared, spearman, …, sharpe_proxy`). Não há troca
de target: ewma R²=0.99048, tgat R²=0.88706, persistence R²=0.87832 medem literalmente a
reconstrução de ρ nível-a-nível. O enquadramento de `EXPERIMENTS.md §4.1` ("null replicado
no harness canônico") é **correto e justo**. (Detalhe corroborante: o MAE do tgat, 0.0361,
é ~4× o do ewma/persistence, ~0.009 — o TGAT perde por margem larga na métrica de erro.)

### 4.c Formulação honesta recomendada (para substituir `STATE.md:3` e `ROADMAP.md:47,81`)

> **H4 (redação financeira TGAT, ≥70% das janelas): NÃO confirmada.** O que a evidência
> suporta é uma hipótese mais fraca e distinta — **superioridade *preditiva* do encoder
> temporal sobre baselines de snapshot, na era TGN**: TGN vs ROLAND/GAT-Static em erro de
> correlação, p<0.0001 (Wilcoxon+DM+Holm, v1.0, `EXPERIMENT_LOG.md:543-546`). A vantagem
> **financeira** (Sharpe/CVaR) **não** se sustentou (v1.0 Sharpe P=0.595; era TGAT
> ≤5/9 janelas, n.s.). O p=0.0018 é um resultado **histórico TGN v0.9, single-split,
> não replicado** — deve ser citado como tal, nunca como confirmação corrente da H4 TGAT.

Recomenda-se também que, se a dissertação mantiver a comparação preditiva, ela use o rótulo
"**superioridade preditiva (era TGN)**", separando-a explicitamente da tese financeira.

---

## 5. Inventário de contaminação (rho_conditioning — item 3)

`scripts/run_spy_vix_covid_compare.py` treina a série "tgat" com
`use_rho_conditioning=True` (`:235`) **sem rebatizar** a variante (`model_variant="tgat"`,
`:227`) e sem registrar o flag nos JSONs. Artefatos **derivados** desse script
(potencialmente contaminados por vazamento de ρ_t no decoder):

**Escritos pelo script (`:25-29,615-619,672,679,762-764`):**
- `figures/stress_event_compare_<PAR>.{pdf,png}` e `figures/stress_event_compare/…`
  (8 pares: SPY_VIX, SPY_GLD, SPY_TLT, **SPY_BTC_USD**, QQQ_BTC_USD, GLD_BTC_USD, XLE_SPY,
  XLK_SPY);
- `results/stress_event_compare/<PAR>_metrics.json` (8) + `<PAR>_predictions.csv` +
  `<PAR>_tgat_preds.csv`;
- `results/stress_event_compare/stress_event_summary.json` e `stress_event_report.md`.

**A série "tgat" desses 8 pares está sob suspeita de conditioning** — deve ser
re-gerada com o flag em False ou rotulada `tgat_plus_rho` antes de qualquer uso na tese.
O padrão honesto existe e é o correto: `run_smoketest_covid_tgat_plus_rho.py` (variante
separada `tgat_plus_rho`, docstring `:9-13`).

**Referência por materiais do paper — checado:** `doc/samplepaper.tex` inclui somente
`figures/dyfo_framework.png` (`:133`), `figures/evaluation_pipeline.png` (comentado, `:500`)
e `figures/wilcoxon_pvalues_dotplot.png` (`:662`); `doc/paper_methodology_results.md` não
inclui figura de stress. **Nenhuma figura/JSON contaminado é referenciado pelo paper atual**
— a contaminação está **contida** aos artefatos de stress event e não vazou para o
manuscrito. (Boa notícia; manter assim.)

---

## 6. Nota de segurança (item 17) — must-fix

- **Achado:** `DataConfig.fred_api_key` (`config.py:91`) traz uma chave FRED de 32 hex
  como **default versionado**. O adapter tem o caminho correto (`.env`/env var,
  `fred_adapter.py:_get_api_key`), mas o default hardcoded sempre vaza.
- **Exposição no histórico git:** introduzida no commit **`1add91f`** ("Inclusão de testes
  estatísticos", 2026-04-13, autor Igor Battazza) — antes o campo era `""`. Como
  `dyfo/config.py` **é rastreado** (`git ls-files` positivo), a credencial está **gravada
  permanentemente no histórico** e não desaparece removendo-a apenas do HEAD.
- **Classificação:** exposição de credencial em VCS (CWE-798, *Use of Hard-coded
  Credentials*). Severidade **Alta** (não bloqueia execução, mas expõe segredo).
- **Remediação (recomendada, NÃO aplicada):** (i) **rotacionar** a chave FRED junto ao
  provedor; (ii) trocar o default por `""` ou `os.environ.get("FRED_API_KEY", "")`;
  (iii) purgar do histórico (`git filter-repo`/BFG) se o repositório for/for a ser público.

---

## 7. Achados NOVOS de metodologia (N1–N5)

Itens não registrados pelas 6 docs Wave-2.

### N1 — Look-ahead na construção dos alvos DCC-GARCH (severidade **Alta**)
`compute_dcc_garch_correlations` (`edge_features.py:182-325`) ajusta:
- GARCH(1,1) por ativo sobre a **série inteira** (`:227,240` — sem janela rolante/expansiva);
- parâmetros DCC(a,b) por MLE sobre **todos** os resíduos padronizados (`:287-290`), com
  `Q_bar` = correlação incondicional do **sample completo** (`:287`);
- recursão forward de `R_t` sobre o sample inteiro (`:299`).

Em `prepare_data` (`train_link_prediction.py:99-248`) esse `corr_by_date` é computado **uma
única vez** sobre `[start,end]` e reusado como **feature** (`corr_today`, `:582`) **e alvo**
(`corr_tomorrow`, `:562`) em **todas** as janelas walk-forward. Consequência: os alvos ρ(t+1)
das janelas de **teste** foram construídos com parâmetros que "viram" o período de teste
(inclusive o futuro). **Impacto:**
- **Não** enviesa comparações *relativas* (todos os modelos partilham o mesmo alvo) → as
  conclusões de horse-race (EWMA>TGAT, win rates, null do probe) **permanecem válidas**;
- **Infla** os R² *absolutos*: 0.824 (link prediction) e ~0.99 (persistence/EWMA) não são
  forecast estritamente causal. A régua persistence≈0.99 vem em parte deste alvo suavizado
  e não-causal.
- **Recomendação:** citar 0.824 sempre com o caveat "alvo DCC-GARCH ajustado no sample
  completo — não é R² out-of-sample estrito". As docs Wave-2 devem acrescentar esta ressalva
  onde citam 0.824/0.99 (OVERVIEW §5, EXPERIMENTS §3/§4.1).

### N2 — Overclaims internos em PROJECT.md (severidade Média)
- `PROJECT.md:76` marca "Regime conditioning `regime_prob` (M1→M2) **✅**" na tabela de
  contribuições, mas BL-09 está pendente e os slots são **zero-filled** (OVERVIEW §1 já
  registra o pendente, mas não sinaliza o ✅ enganoso da tabela).
- `PROJECT.md:77` marca "Validação **10k-bootstrap walk-forward ✅**"; o bootstrap
  walk-forward real usa **2000/5** — o "10k" só existiu no single-split v0.9 (não
  walk-forward). Reforça a discrepância do item 6.

### N3 — Default do runner canônico é 30 tickers, não 50 (severidade Baixa)
`DEFAULT_N_TICKERS = 30` (`rev3:116`) contradiz o invariante "universo padrão 50 ações"
(`ARCHITECTURE.md §1`, `.specs/codebase/ARCHITECTURE.md`). Os runs de 50 tk exigiram
`--n_tickers 50` explícito. Nuance de precisão, não erro de resultado.

### N4 — Estimativas pontuais e CIs single-seed (severidade Baixa/Média)
O `descriptive_summary` usado para EWMA vs TGAT e as comparações Wilcoxon repousa em
**seed 42** (`run_config.seeds=[42]` em `rev3_20260501_200449`); o bootstrap de bloco usa
`seed=42+window_idx` fixo (`rev3:329`). Reprodutível, porém a robustez a seeds do horse-race
canônico **não** foi exercida (multi-seed só existe no braço de ablação `abl_full_tgat`, e
mesmo lá 3 seeds — item 11). Não invalida o null (a margem EWMA−TGAT é grande), mas os
p-values de janela (n=9) são de baixa potência.

### N5 — Protocolo walk-forward do paper está comentado (severidade Baixa)
A descrição das 9 janelas 500/125/125 em `samplepaper.tex:155` está **comentada** (linha
inicia com `%`). Se o manuscrito for compilado como está, a seção de métodos pode **não**
enunciar o protocolo. Verificar antes da submissão.

**Sanidade do que NÃO é leakage (confirmado OK):**
- `build_windows` (`run_bootstrap_eval_v5.py:277-299`): train/val/test são fatias
  **consecutivas e disjuntas**; `cursor += step_size` (125 = test_size) → janelas
  **não sobrepostas**; sem overlap train↔eval dentro da janela.
- O runner **evita** DM agregado quando janelas se sobrepõem: `if step_days >= test_days`
  (`rev3:498`), senão pula com warning (`:513`) — cuidado metodológico correto.
- `gat_static` calcula o grafo estático só nas **datas de treino** ("preventing look-ahead
  leakage", `gat_static_baseline.py:188`).
- Testes estatísticos: Wilcoxon one-sided "greater", DM (HAC), Holm-Bonferroni presentes e
  aplicados (`rev3:471-513`); `primary_comparison` reporta binomial + Wilcoxon + Holm
  coerentemente. Sem anomalia.

---

## 8. Separação R² × null e setas PLANEJADAS (tarefa 6) — OK

- **Separação mantida.** As docs preservam o null do probe (R²≈−5.35/−4.36 vs persistência
  0.99, `probe_results.txt`) **separado** do R²=0.824 de link prediction (OVERVIEW §4-5,
  EXPERIMENTS §3/§4.1/§6), com a regra "nunca maquiar o null com o 0.824" explícita
  (`EXPERIMENTS.md:238`). Verificado e correto. (N1 apenas *acrescenta* um caveat de causalidade
  a ambos os R², não altera a separação.)
- **Setas PLANEJADAS.** DyFO→ORION (BL-09/DI-1) marcadas como planejadas e confirmadas no
  repo ORION: `ROADMAP.md:29-31` (M6 ⏳ ACTIVE), `constructor.py:62` (`graph_embedding=None`,
  desligado, nunca chamado), `:74` (shape-check `(128,)` desatualizado), `docs/MODELS.md:144`
  (Lesson L-1, "no implementation in this repository"). Nenhum arco planejado é apresentado
  como implementado.

---

## 9. Correções que as docs Wave-2 precisam (NÃO aplicadas)

Registradas para a próxima onda de edição; **não** modificar as 6 docs Wave-2 aqui.

1. **Acrescentar caveat N1** (look-ahead DCC-GARCH) em OVERVIEW §5 e EXPERIMENTS §3/§4.1
   onde 0.824 e ~0.99 são citados: "alvo ajustado no sample completo → R² não é out-of-sample
   estrito; válido para comparação relativa, não como forecast causal".
2. **H4:** trocar toda referência a "H4 CONFIRMADA (p=0.0018)" pela formulação honesta do §4.c.
   As docs já decompõem os p-values corretamente; falta o veredito unificado "H4 financeira
   TGAT NÃO confirmada".
3. **Sinalizar os overclaims de PROJECT.md** (N2): `:76` regime_prob ✅ e `:77` "10k-bootstrap
   walk-forward ✅" são internamente falsos — anotar como discrepância documental.
4. **Segurança:** elevar a nota da chave FRED de "nota de rodapé" (DATA §1.2) para achado
   **must-fix** com a exposição no histórico git (commit `1add91f`).
5. **Contaminação:** DATA/EXPERIMENTS podem acrescentar que **nenhum** artefato contaminado
   é referenciado pelo paper (contenção confirmada) — e que o SPY_BTC_USD (citado como
   evidência de "tgat_event_window_win" em OVERVIEW §4) é justamente um dos 8 pares
   potencialmente condicionados: o caveat já existe em EXPERIMENTS §7, mas OVERVIEW §4 o
   cita sem o mesmo destaque.
6. **Nuance N3** (default 30 tk do runner) em TESTING §2.

---

## 10. Itens residuais não verificados

Mantidos como `[NÃO VERIFICADO]`, coerentes com o que as docs Wave-2 já sinalizam:

1. **Sharpe=2.615 / MDD=12.4% / Turnover=0.085** como números *únicos*: MDD e Turnover não
   existem em nenhum artefato (busca exaustiva); Sharpe só aproximável por média de 2 runs
   30-tk. Confirmado que **não há run único** com as 4 métricas.
2. **p=0.0018 e p<0.0001 contra JSON bruto**: fonte única `doc/EXPERIMENT_LOG.md` (fora do
   git); os runs `bootstrap_eval_20260412_170532` e `bootstrap_eval_v3_20260413_085728` não
   existem mais em `results/` (só preserva ≥2026-04-16). Não auditável além do log.
3. **Execução real com 5 seeds**: inexistente (item 11).
4. **calc_sigma.py "Paper claims σ=0.034/0.077"**: comentários hardcoded (`calc_sigma.py:83-84`),
   não recomputados.
5. **Causa do Spearman=nan do DyFO no probe**: predições ~constantes (modo
   "cross-sectional memorisation"); CSVs `probe_*_dyfo_preds.csv` não abertos.
6. **Notebooks colab_*.ipynb**, **audit_data_sources.py**, **mdd_turnover.log** linha-a-linha
   — não auditados.
7. **temporal_kg — contagem exata de runs**: MODELS §2.4 diz "4 runs"; localizei 2 com
   `results.json` (s42); melhor R²=0.7124 **confirmado**, contagem exata não. Ausência de
   temporal_kg em `metrics_by_variant` confirmada nos sumários inspecionados.
8. **Magnitude do N1 (leakage) sob correção causal**: não re-rodei o pipeline com DCC
   expansivo; a direção do efeito (infla R² absoluto) é certa, a magnitude não foi medida
   (proibido treinar nesta onda).

---

## 11. Status final

`verified` (18/18 discrepâncias confirmadas; 28 citações auditadas) **com**:
- 1 `caveat` metodológico NOVO de alto impacto interpretativo (**N1** — leakage DCC-GARCH);
- 1 **must-fix** de segurança (chave FRED versionada, §6);
- 1 correção científica obrigatória de redação (**H4**, §4.c) — não bloqueia a documentação,
  mas bloqueia qualquer alegação de "H4 confirmada" na dissertação/paper.

As 6 docs Wave-2 são fiéis e podem ser usadas como base; devem receber as 6 correções do §9
antes de virarem material de tese.
