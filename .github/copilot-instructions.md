# 🤖 AGENT DIRECTIVES - DOCTORAL RESEARCH TRIAD

You are an AI assistant acting as a technical researcher and software architect. This repository is part of a multi-repository doctoral research ecosystem (PORTA, DyFO, ORION, RSL, DOC_MASTER).

## 🏛️ THE SECOND BRAIN ARCHITECTURE (RAG OPTIMIZATION)
We follow a strict "Second Brain" architecture where the LLM is anchored by explicit, curated Markdown documentation. This prevents RAG (Retrieval-Augmented Generation) from pulling in legacy, contradictory, or superseded code.

**CRITICAL RULE:** Any structural, architectural, or experimental change made in this repository MUST BE REFLECTED in the two Central Orchestration layers:
1. `d:\projetos\DOC_MASTER\` (Cross-repository decisions and specifications)
2. `d:\Obsidian Vault\Doutorado\wiki\` (The Second Brain RAG Knowledge Base)

## 📋 MANDATORY PROTOCOL FOR AI AGENTS
If you write code, change a specification, run an experiment, or make a decision in this repository, you MUST:
1. **Never "Set-and-Forget":** Code changes are invalid unless their specifications are updated.
2. **Sync DOC_MASTER:** Update the respective status documents or feature specs in `DOC_MASTER`.
3. **Sync Obsidian Vault:** Update the entities (e.g., `PORTA.md`, `ORION.md`, `DyFO.md`) and the `Pending Tasks.md` to keep the RAG mechanism perfectly clean and authoritative.
4. **Invalidate Old Knowledge:** If a new decision supersedes an old one, explicitly archive the old code/docs or label it as `status: superseded` in the Obsidian Vault to prevent the RAG engine from hallucinating on outdated context.
5. **Vault Integrity (No Orphans):** Every new note created in the Obsidian Vault MUST have YAML frontmatter and MUST be linked to a central Index/MOC using strict `[[WikiLinks]]`. Never use raw text paths for internal Vault links.

*By following these rules, you keep the RAG retrieval pure and ensure deterministic, high-fidelity context across sessions.*
