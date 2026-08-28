from pathlib import Path

vault_dir = Path("d:/Obsidian Vault/Doutorado/wiki")

# 1. Update DyFO.md
dyfo_md = vault_dir / "entities" / "DyFO.md"
dyfo_content = """---
title: "DyFO"
note_type: project
status: active
authority: source_repository
source_ids: []
source_commits:
  - "a36e68a"
  - "41cf9b8"
  - "3995173"
  - "8f1902c"
  - "c81ecaf"
  - "9352ff6"
  - "4962c58"
  - "f1ea023"
last_compiled: 2026-08-28T12:00:00Z
last_verified: 2026-08-28T12:00:00Z
confidence: high
human_review_required: false
tags:
  - "type/project"
  - "status/active"
  - "project/dyfo"
  - "theme/project-map"
---

# DyFO (Heterogeneous Temporal Graph Attention with Typed Edge Conditioning for Financial Co-movement Forecasting)

## Estado Atual (2026-08-28)

- **Branches ativas e consolidadas**:
  - `dev`: Base estabilizada com remediação integral de auditoria causal (REQ-D1, D2, D5, D6) e suite 46/46 GREEN.
  - `follow_on/portfolio_integration`: 
    - Implementação de `DyFOAdapter`, `StructuralGraphSnapshot`, `docs/ONTOLOGY_SCHEMA.md`, `PortaDataReader` (estritamente *read-only*), pipeline DRL causal.
    - Suíte de 6 figuras de alta resolução (300 DPI) para o BRACIS 2026.
    - Deck interativo de 11 slides (`doc/bracis_presentation_deck.html` com imagens em Base64, fórmulas vetoriais SVG e controle fullscreen) e roteiro detalhado de falas (`doc/BRACIS_PRESENTATION_NOTES.md`).
    - Remediações de auditoria pós-BRACIS (DeepSeek/Opencode GO):
      - **P0-1**: Default `correlation_method = "rolling_pearson"` garantindo causalidade estrita sem look-ahead de $\\bar{Q}$.
      - **P0-2**: Leitura do tensor real de regimes `S.npy` de PORTA com fallback contínuo softmax sobre `M.npy`.
      - **P1-1**: Mapeamento dinâmico de features por nome de coluna via `get_feature_columns()`.
      - **P1-2**: Suporte a carregamento de checkpoint treinado em `DyFOAdapter` com fallback determinístico.
      - **P1-3/P1-4/P1-5**: Ajuste rigoroso de nuances estatísticas (Sharpe $p=1.00$ vs colapso de simetria do Raw-DRL $1/N$, escala $N=18, 50, 100/104$) e contraste SVG $\\parallel$.
- **Suite de testes**: 46 testes passando (`pytest` 100% green), incluindo guards causais, integridade imutável *read-only* do PORTA e exportação estrutural.

## Papel no Framework do Doutorado (A Tríade)

1. **Ablação e Modelagem Estrutural Relacional (PORTA)**: Gerador de $\\mathbf{\\Sigma}_t$ estrutural e topologia decomposta ($A_\\text{CORR}, A_\\text{SECT}, A_\\text{SUPL}, A_\\text{FACT}$) via `DyFOAdapter.export_structural_graph()`.
2. **Consumo de Dados Curados (PORTA &rarr; DyFO)**: Leitura estritamente *read-only* (sem jamais alterar dados do PORTA) de tensores curados $X, R, M, S$, probabilidades de regime $\\pi_t$ e séries de retornos via `PortaDataReader`.
3. **Percepção Multimodal (ORION)**: Canal de embeddings relacionais temporais ($e_t \\in \\mathbb{R}^{100}$) para o `StateConstructor`.

## Invariante Absoluto (Lei de Não-Modificação do PORTA)

- Todos os acessos a `d:\\projetos\\PORTA` pelo DyFO são realizados exclusivamente via `mmap_mode='r'` ou streams de leitura.
- Coberto e protegido por teste automatizado `test_porta_reader_readonly_contract` (verifica hashes/mtimes dos arquivos antes e depois da execução).

## Key Navigation
- [[Pending Tasks]]
- [[Blocked Workstreams]]
"""

dyfo_md.parent.mkdir(parents=True, exist_ok=True)
dyfo_md.write_text(dyfo_content, encoding="utf-8")
print("Vault DyFO.md updated!")

# 2. Update Pending Tasks.md if exists
pending_md = vault_dir / "Pending Tasks.md"
if pending_md.exists():
    p_text = pending_md.read_text(encoding="utf-8")
    p_text = p_text.replace("- [ ] DyFO D1/D2 remediation", "- [x] DyFO D1/D2/D5/D6 causality remediation (100% completed)")
    p_text = p_text.replace("- [ ] DyFO portfolio integration adapter", "- [x] DyFO portfolio integration adapter (`DyFOAdapter`, `PortaDataReader`, `StructuralGraphSnapshot`) (100% completed)")
    p_text = p_text.replace("- [ ] DyFO BRACIS slides", "- [x] DyFO BRACIS 2026 presentation package (11-slide HTML deck, 6 high-res figures, speaker notes) (100% completed)")
    pending_md.write_text(p_text, encoding="utf-8")
    print("Vault Pending Tasks.md updated!")
