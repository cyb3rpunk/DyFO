# DyFO - Manual de Execucao

Este manual descreve como preparar o ambiente e executar o pipeline do DyFO no repositorio atual.

## 1. Pre-requisitos

- Python 3.10+
- Conexao com internet (yfinance, FRED e Ken French Data Library)
- PowerShell (Windows)

Dependencias Python principais:
- torch, torch-geometric
- pandas, numpy, scipy
- yfinance, fredapi
- arch (DCC-GARCH)
- matplotlib, networkx

## 2. Preparacao do ambiente

No diretorio raiz do projeto, execute:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 3. Configuracao de variaveis de ambiente

Copie o arquivo de exemplo e preencha sua chave do FRED:

```powershell
Copy-Item .env.example .env
```

Edite o arquivo `.env` e configure:

```text
FRED_API_KEY=<sua_chave>
```

Sem a chave FRED, o pipeline ainda roda, mas parte dos eventos macro pode ficar indisponivel.

## 4. Fluxo rapido (fim-a-fim)

Depois de ativar o ambiente:

```powershell
python scripts/train_link_prediction.py
python scripts/plot_results.py
python scripts/visualize_dyfo.py
```

Resultados esperados:
- Nova pasta em `results/link_pred_YYYYMMDD_HHMMSS/`
- Arquivos principais:
  - `results.json`
  - `history.json`
  - `best_model.pt`
  - `training_results.png`
  - `dyfo_structure.png`

## 5. Fluxo recomendado (com validacoes)

### 5.1 Auditoria das fontes de dados

```powershell
python scripts/audit_data_sources.py
```

Esse script verifica:
- cobertura de precos/OHLCV
- earnings e corporate actions
- disponibilidade de series FRED
- status do pacote `arch` para DCC-GARCH

### 5.2 Smoke test do modulo

```powershell
python tests/test_smoke.py
```

### 5.3 Treinamento de link prediction

```powershell
python scripts/train_link_prediction.py
```

Configuracao padrao atual do script:
- Universo: 30 ativos S&P 500 (11 setores GICS)
- Janela temporal: 2020-01-01 ate 2024-12-31
- Modo: regressao de correlacao (`mode="regression"`)
- Correlacao: DCC-GARCH (`correlation_method="dcc_garch"`)
- Epocas: 10, com early stopping
- Gradient clipping: habilitado (`grad_clip_enabled=True`, `grad_clip_max_norm=1.0`)
- Scheduler: ReduceLROnPlateau habilitado (`factor=0.5`, `patience=2`, `min_lr=1e-6`)
- Split walk-forward: 60/20/20

### 5.4 Plot de metricas de treino

```powershell
python scripts/plot_results.py
```

Gera `training_results.png` no run mais recente.

### 5.5 Visualizacao de estrutura/ontologia

```powershell
python scripts/visualize_dyfo.py
```

Comandos uteis:

```powershell
python scripts/visualize_dyfo.py --run-dir results/link_pred_20260327_083335
python scripts/visualize_dyfo.py --save-path results/dyfo_structure_custom.png
```

## 6. Estrutura de saida

Cada execucao de treino cria um diretorio em `results/`:

- `best_model.pt`: melhor checkpoint por validacao
- `results.json`: metricas finais e parametros
- `history.json`: historico por epoca (train/val)
- `training_results.png`: curvas de treino
- `dyfo_structure.png`: comparacao visual ontologia vs grafo instancia (quando executado)

Logs adicionais podem ser gerados no diretorio `logs/`.

## 7. Solucao de problemas

### Erro: pacote `arch` ausente

```powershell
pip install arch
```

Sem `arch`, o metodo DCC-GARCH nao fica disponivel corretamente.

### Erro ao ativar `.venv` no PowerShell

Se houver bloqueio de politica de execucao:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Feche e reabra o terminal.

### FRED sem chave

Se `FRED_API_KEY` nao estiver definida, configure no arquivo `.env`.

### Falhas intermitentes em APIs externas

As fontes de mercado podem oscilar. Tente novamente em alguns minutos. O projeto ja possui retry em partes do pipeline.

## 8. Execucao de teste de integracao com dados reais

Para exercitar o pipeline completo em modo de integracao:

```powershell
python tests/test_real_data.py
```

Esse teste baixa dados reais e pode demorar mais que o smoke test.

## 9. Checklist rapido para reproducao

1. Ativar ambiente virtual.
2. Instalar dependencias.
3. Configurar `.env` com `FRED_API_KEY`.
4. Rodar `python scripts/audit_data_sources.py`.
5. Rodar `python scripts/train_link_prediction.py`.
6. Rodar `python scripts/plot_results.py`.
7. Rodar `python scripts/visualize_dyfo.py`.
8. Conferir artefatos em `results/<run>/`.
