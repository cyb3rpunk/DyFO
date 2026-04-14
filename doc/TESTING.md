# Como Executar os Testes do DyFO

Este guia descreve de forma simples os passos necessários para executar a suite de testes automatizados do projeto DyFO utilizando o ambiente virtual Python. 

O projeto é configurado para utilizar o framework **`pytest`**.

## 1. Ativando o Ambiente Virtual (.venv)

É fundamental sempre ativar o ambiente virtual antes de rodar os testes para garantir que as dependências corretas (isoladas) sejam utilizadas.

**No Windows (PowerShell/CMD):**
```powershell
.\.venv\Scripts\activate
```

**No Linux/macOS:**
```bash
source .venv/bin/activate
```

*(Você saberá que deu certo quando `(.venv)` aparecer no início da linha de comando do seu terminal.)*

## 2. Garantindo as Dependências

Com o ambiente virtual ativado, certifique-se de ter as dependências e o utilitário de testes instalados:

```powershell
# Instala as dependências do projeto
python -m pip install -r requirements.txt

# Instala o pytest
python -m pip install pytest

# IMPORTANTE: Instala o projeto DyFO em modo editável para que os imports funcionem
python -m pip install -e .
```

## 3. Comandos para Rodar os Testes

Abaixo estão os comandos recomendados para executar os testes. Lembre-se: eles devem ser executados na raiz do projeto (`d:\projetos\DyFO`), sempre com o `.venv` ativado.

### ▶️ Executar TODOS os testes
Esse comando varre a pasta `tests/` e executa todos os arquivos e testes que encontrar:
```powershell
pytest
```
*(Ou, de forma alternativa: `python -m pytest`)*

### ▶️ Executar os testes de um arquivo específico
Se você alterou um arquivo e quer testar apenas ele (ex: `test_real_data.py`):
```powershell
pytest tests\test_real_data.py
```

### ▶️ Executar de modo detalhado (Verbose)
Mostra quais testes específicos passaram (`PASSED`) ou falharam (`FAILED`), ao invés de exibir apenas pontos `..`:
```powershell
pytest -v
```

### ▶️ Exibir prints e logs ignorados
O `pytest` por padrão suprime a saída (`print()`) de testes que passaram com sucesso. Para forçar a visualização de tudo que está sendo "printado":
```powershell
pytest -s
```

*Dica: Você pode combinar os parâmetros, usando por exemplo `pytest -vs` para obter tanto o modo detalhado quanto as saídas do console.*

## 4. Como Executar o DyFO (Principais Scripts)

O DyFO é composto por alguns scripts de execução disponíveis dentro da pasta `scripts/`. Para rodar o modelo ou realizar auditoria de dados, você deve executá-los com o ambiente virtual `.venv` ativado.

### ▶️ Treinamento e Execução do Modelo Principal
Este é o script central do projeto. Ele treina o encoder *Temporal Graph Networks* junto do modelo analítico e prevê as correlações futuras dos ativos:
```powershell
python scripts\train_link_prediction.py
```
*(Isso rodará o pipeline completo de treinamento, salvando os resultados em logs localmente.)*

### ▶️ Checagem/Auditoria das Fontes de Dados
Para validar se as integrações com os provedores (como *Yahoo Finance*, *FRED*, etc) estão devidamente ativas:
```powershell
python scripts\audit_data_sources.py
```

### ▶️ Plotagem/Visualização dos Resultados
Para renderizar gráficos das saídas após um treinamento finalizado:
```powershell
python scripts\plot_results.py
```

> **Dica de Atalho:** Caso prefira não ativar o venv manualmente, você pode chamar o interpretador diretamente:
> `.\.venv\Scripts\python scripts\train_link_prediction.py`

> **Nota:** Certifique-se também de configurar a sua chave de API (por exemplo, a `FRED_API_KEY`) utilizando o arquivo `.env` na raiz do projeto, caso não o tenha feito!
