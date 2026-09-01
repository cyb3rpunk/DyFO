"""GraphRAG Prompt Engine & Neuro-Symbolic LLM Reasoner for DyFO.

Translates temporal causal subgraphs into Chain-of-Thought prompts and interacts with
Large Language Models (OpenAI, Anthropic, Gemini, Ollama, or deterministic local mock)
to generate causal risk explanations and symbolic portfolio constraints.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from dyfo.neurosymbolic.subgraph_extractor import CausalSubgraph

logger = logging.getLogger("DyFO.NeuroSymbolic")


@dataclass
class RiskExplanation:
    """Structured causal explanation and proposed symbolic constraints from LLM."""
    date: str
    macro_rationale: str
    spillover_risks: List[str] = field(default_factory=list)
    recommended_sector_caps: Dict[str, float] = field(default_factory=dict)
    exclude_tickers: List[str] = field(default_factory=list)
    hedging_action: str = "NONE"
    min_cash_buffer: float = 0.0
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GraphRAGPromptEngine:
    """Builds structured Chain-of-Thought prompts from DyFO causal subgraphs."""

    SYSTEM_PROMPT = (
        "You are an elite Senior Quantitative Risk Strategist and Neuro-Symbolic Portfolio AI.\n"
        "Your task is to analyze dynamic temporal graph co-movements, cross-sector spillovers, "
        "and macro shocks predicted by the DyFO Temporal Graph Neural Network.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Provide a rigorous, causal Chain-of-Thought explanation of why asset co-movements are changing.\n"
        "2. Identify hidden contagion pathways (e.g. supply chain, cross-sector spillovers).\n"
        "3. Output actionable, mathematically bounded symbolic constraints for portfolio optimization.\n"
        "4. You MUST end your response with a valid JSON block enclosed in ```json ... ``` with keys:\n"
        "   - 'macro_rationale': string\n"
        "   - 'spillover_risks': list of strings\n"
        "   - 'recommended_sector_caps': dict mapping GICS sector names to maximum weight bounds (e.g. 0.25)\n"
        "   - 'exclude_tickers': list of tickers to strictly set weight = 0.0 (if any extreme tail risk)\n"
        "   - 'hedging_action': one of ['NONE', 'MILD_HEDGE', 'STRONG_HEDGE', 'DEFENSIVE_ROTATE']\n"
        "   - 'min_cash_buffer': float between 0.0 and 0.50\n"
    )

    def build_prompt(
        self,
        subgraph: CausalSubgraph,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Construct prompt combining system directives, subgraph triples, and portfolio state."""
        natural_graph = subgraph.to_natural_text(max_triples=20)

        context_lines = []
        if portfolio_context:
            context_lines.append("\n--- Current Portfolio State ---")
            for k, v in portfolio_context.items():
                context_lines.append(f"- {k}: {v}")

        portfolio_block = "\n".join(context_lines)

        user_prompt = (
            f"{natural_graph}\n"
            f"{portfolio_block}\n\n"
            "Analyze the causal graph state above for tomorrow's market open.\n"
            "Explain the systemic dynamics and output the JSON block with risk constraints."
        )
        return user_prompt


class LLMReasoner:
    """Neuro-symbolic LLM reasoner with support for live APIs and deterministic mock fallback."""

    def __init__(
        self,
        backend: str = "mock",
        model_name: Optional[str] = None,
        temperature: float = 0.1,
    ):
        self.backend = backend.lower()
        self.model_name = model_name or ("gpt-4o-mini" if backend == "openai" else "claude-3-5-sonnet")
        self.temperature = temperature
        self.prompt_engine = GraphRAGPromptEngine()

    def reason(
        self,
        subgraph: CausalSubgraph,
        portfolio_context: Optional[Dict[str, Any]] = None,
    ) -> RiskExplanation:
        """Execute causal reasoning on the subgraph, returning structured RiskExplanation."""
        prompt = self.prompt_engine.build_prompt(subgraph, portfolio_context)

        if self.backend == "mock" or "MOCK_LLM" in os.environ:
            return self._mock_reasoning(subgraph)

        # Attempt live API call if backend configured and API keys present
        try:
            if self.backend == "openai" and "OPENAI_API_KEY" in os.environ:
                return self._call_openai(prompt, subgraph.date)
            elif self.backend == "anthropic" and "ANTHROPIC_API_KEY" in os.environ:
                return self._call_anthropic(prompt, subgraph.date)
            elif self.backend == "gemini" and "GEMINI_API_KEY" in os.environ:
                return self._call_gemini(prompt, subgraph.date)
        except Exception as e:
            logger.warning("Live LLM API call failed (%s). Falling back to deterministic mock reasoner.", e)

        return self._mock_reasoning(subgraph)

    def _mock_reasoning(self, subgraph: CausalSubgraph) -> RiskExplanation:
        """Deterministic, domain-aware mock reasoner for offline CI/CD and self-contained execution."""
        dt = subgraph.date
        conc = subgraph.eigen_concentration
        regime = subgraph.macro_regime

        # Causal rules based on eigenvalue concentration & top triples
        sector_caps: Dict[str, float] = {
            "Information Technology": 0.30,
            "Financials": 0.25,
            "Health Care": 0.25,
            "Consumer Discretionary": 0.20,
            "Energy": 0.20,
        }
        spillovers: List[str] = []
        exclude_tickers: List[str] = []
        hedging = "NONE"
        min_cash = 0.0

        if conc > 0.45 or regime == "HIGH_STRESS_CONTAGION":
            hedging = "STRONG_HEDGE"
            min_cash = 0.15
            sector_caps["Information Technology"] = 0.18
            sector_caps["Financials"] = 0.18
            sector_caps["Utilities"] = 0.30
            spillovers.append("Severe spectral eigenvalue concentration (>45%) indicates market-wide contagion risk.")
            rationale = (
                f"DyFO predicts severe spectral concentration ({conc:.1%}) on {dt}. "
                "Cross-sector correlations are surging due to systemic macro spillover. "
                "Defensive rotation into Utilities and minimum 15% cash buffer is strictly enforced."
            )
        elif conc < 0.22:
            hedging = "NONE"
            min_cash = 0.0
            spillovers.append("High idiosyncratic dispersion across sectors; favorable for fundamental stock picking.")
            rationale = (
                f"Market on {dt} exhibits high dispersion ({conc:.1%} top eigenvalue). "
                "Co-movement is localized within supply-chain clusters without systemic contagion. "
                "Unconstrained GMVP allocation is permitted."
            )
        else:
            hedging = "MILD_HEDGE"
            min_cash = 0.05
            spillovers.append("Moderate correlation innovations observed in high-beta technology and semiconductor nodes.")
            rationale = (
                f"Moderate market regime on {dt} with top-1 eigenvalue concentration at {conc:.1%}. "
                "Standard sector risk caps applied to prevent concentration in AI hardware supply chain."
            )

        # Inspect specific triples for asset-level anomalies
        for t in subgraph.triples:
            if t.delta_rho is not None and t.delta_rho > 0.35:
                spillovers.append(f"Acute co-movement shock between {t.source} and {t.target} (Δρ={t.delta_rho:+.2f}).")

        json_payload = {
            "macro_rationale": rationale,
            "spillover_risks": spillovers,
            "recommended_sector_caps": sector_caps,
            "exclude_tickers": exclude_tickers,
            "hedging_action": hedging,
            "min_cash_buffer": min_cash,
        }

        raw = f"```json\n{json.dumps(json_payload, indent=2)}\n```"

        return RiskExplanation(
            date=dt,
            macro_rationale=rationale,
            spillover_risks=spillovers,
            recommended_sector_caps=sector_caps,
            exclude_tickers=exclude_tickers,
            hedging_action=hedging,
            min_cash_buffer=min_cash,
            raw_response=raw,
        )

    def _call_openai(self, prompt: str, date_str: str) -> RiskExplanation:
        """Call OpenAI Chat Completions API."""
        import urllib.request
        api_key = os.environ["OPENAI_API_KEY"]
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": GraphRAGPromptEngine.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
        }
        req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
            return self._parse_llm_response(content, date_str)

    def _call_anthropic(self, prompt: str, date_str: str) -> RiskExplanation:
        """Call Anthropic Messages API."""
        import urllib.request
        api_key = os.environ["ANTHROPIC_API_KEY"]
        url = "https://api.anthropic.com/v1/messages"
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "max_tokens": 1024,
            "system": GraphRAGPromptEngine.SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["content"][0]["text"]
            return self._parse_llm_response(content, date_str)

    def _call_gemini(self, prompt: str, date_str: str) -> RiskExplanation:
        """Call Google Gemini API."""
        import urllib.request
        api_key = os.environ["GEMINI_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": GraphRAGPromptEngine.SYSTEM_PROMPT + "\n\n" + prompt}]}],
            "generationConfig": {"temperature": self.temperature},
        }
        req = urllib.request.Request(url, json.dumps(payload).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            return self._parse_llm_response(content, date_str)

    def _parse_llm_response(self, text: str, date_str: str) -> RiskExplanation:
        """Extract and validate JSON block from LLM natural text."""
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # Fallback regex for raw JSON
            match_raw = re.search(r"(\{.*\})", text, re.DOTALL)
            json_str = match_raw.group(1) if match_raw else "{}"

        try:
            parsed = json.loads(json_str)
        except Exception:
            parsed = {}

        return RiskExplanation(
            date=date_str,
            macro_rationale=parsed.get("macro_rationale", text[:300]),
            spillover_risks=parsed.get("spillover_risks", []),
            recommended_sector_caps=parsed.get("recommended_sector_caps", {}),
            exclude_tickers=parsed.get("exclude_tickers", []),
            hedging_action=parsed.get("hedging_action", "NONE"),
            min_cash_buffer=float(parsed.get("min_cash_buffer", 0.0)),
            raw_response=text,
        )
