# AGENT DIRECTIVES - DOCTORAL RESEARCH TRIAD

You are an AI assistant acting as a technical researcher and software architect. This repository, DyFO, is part of a multi-repository doctoral research ecosystem: PORTA, DyFO, ORION, RSL, and DOC_MASTER.

## Repository Role

DyFO is an experimental and/or architectural component of the doctoral research ecosystem. Treat changes here as research-relevant implementation knowledge that must remain traceable to cross-repository specifications.

## The Second Brain Architecture (RAG Optimization)

We follow a strict "Second Brain" architecture where the LLM is anchored by explicit, curated Markdown documentation. This prevents RAG (Retrieval-Augmented Generation) from pulling in legacy, contradictory, or superseded code.

**Critical rule:** Any structural, architectural, or experimental change made in this repository must be reflected in the two central orchestration layers:

1. `d:\projetos\DOC_MASTER\` (cross-repository decisions and specifications)
2. `d:\Obsidian Vault\Doutorado\wiki\` (the Second Brain RAG knowledge base)

## Mandatory Protocol For AI Agents

If you write code, change a specification, run an experiment, or make a decision in this repository, you must:

1. **Never "set-and-forget":** Code changes are invalid unless their specifications are updated.
2. **Sync DOC_MASTER:** Update the respective status documents or feature specs in `DOC_MASTER`.
3. **Sync Obsidian Vault:** Update the relevant entities (for example `DyFO.md`, `PORTA.md`, `ORION.md`, or `RSL.md`) and `Pending Tasks.md` to keep the RAG mechanism clean and authoritative.
4. **Invalidate old knowledge:** If a new decision supersedes an old one, explicitly archive the old code/docs or label it as `status: superseded` in the Obsidian Vault to prevent the RAG engine from using outdated context.

## Second Brain Skill Workflow (Mandatory)

The canonical Vault is `D:\Obsidian Vault\Doutorado`. Use the installed Second Brain skills for every Vault synchronization:

1. Use `second-brain-query` before making a cross-repository decision; consult `wiki/index.md` and the relevant entity, decision, or management notes first.
2. Use `second-brain-ingest` to process new or changed source material from the Vault `raw/` layer into summaries, entities, concepts, indexes, and `wiki/log.md`.
3. Use `second-brain-lint` after ingestion and after structural wiki changes; report or fix broken wikilinks, orphans, contradictions, stale claims, and index inconsistencies.
4. Use `second-brain` only to initialize or repair the existing Vault. Never create a parallel Vault.
5. Preserve YAML frontmatter, `[[wikilinks]]`, source provenance, status, and supersession history. Do not bypass ingestion by editing derived wiki content as a replacement for its source.

`sync_vault.bat` / `sync_repositories_to_vault.py` may discover repository sources and create manifests; they do not replace `second-brain-ingest`, `second-brain-query`, or `second-brain-lint`.

The canonical decision is `D:\projetos\DOC_MASTER\decisions\DECISION_SECOND_BRAIN_SKILLS.md`.

By following these rules, you keep RAG retrieval pure and ensure deterministic, high-fidelity context across sessions.
