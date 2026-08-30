"""
src/ai_agent.py
---------------
AI diagnostic engine for NetSage AI.

Uses OpenAI Chat Completions with structured JSON output validated
by a strict Pydantic v2 schema.

Prompts are loaded from:
  prompts/system_prompt.txt
  prompts/user_prompt_template.txt

Public API:
  diagnose(case: dict) -> DiagnosisResult | ErrorResult
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required on Streamlit Cloud (uses Secrets instead)

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError as e:
    raise ImportError(
        "pydantic is required. Run: pip install pydantic>=2.7.0"
    ) from e


# ── Prompt file loader ────────────────────────────────────────────────────────

# On Streamlit Cloud, cwd is always the repo root, so "prompts/" works directly.
PROMPTS_DIR = Path("prompts")

def _load_prompt(filename: str) -> str:
    """Load a prompt text file, falling back to an embedded default if missing."""
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""  # fallback handled in diagnose()

# ── Pydantic response schema ───────────────────────────────────────────────────

class DiagnosisResult(BaseModel):
    """
    Strict schema for the AI's structured diagnostic output.
    Every field is required — the LLM is instructed to always populate all fields.
    """
    root_cause: str = Field(
        description="One-sentence description of the exact configuration fault causing the issue."
    )
    osi_layer: str = Field(
        description="OSI layer(s) most relevant to this fault, e.g. 'Layer 2 - Data Link'."
    )
    confidence: Literal["High", "Medium", "Low"] = Field(
        description="AI's confidence in the diagnosis based on available evidence."
    )
    evidence: list[str] = Field(
        description="2-4 specific lines or values from the show-command output that support this diagnosis."
    )
    next_command: str = Field(
        description="The single most useful Cisco IOS show command to run next to confirm this diagnosis."
    )
    fix_steps: list[str] = Field(
        description="Ordered list of exact Cisco IOS commands or steps to resolve the fault."
    )
    beginner_explanation: str = Field(
        description="2-3 sentence plain-English explanation suitable for a networking student."
    )


class ErrorResult(BaseModel):
    """Returned when the AI call fails or the response cannot be parsed."""
    error: str
    raw_response: str = ""


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are NetSage AI, an expert Cisco network diagnostics assistant specializing in Packet Tracer lab troubleshooting.

When given a network symptom, topology description, and Cisco IOS show-command outputs, you will:
1. Identify the precise root cause of the connectivity or configuration fault.
2. Cite specific evidence from the show outputs.
3. Provide exact Cisco IOS remediation commands.
4. Explain the issue in simple terms for a networking student.

IMPORTANT RULES:
- Never make up interface names or IP addresses not present in the provided output.
- If the evidence is ambiguous, set confidence to "Low" and explain what additional data is needed.
- Always suggest the single best next diagnostic command to run.
- Keep fix_steps as exact IOS commands that can be copy-pasted into a terminal.

You MUST respond ONLY with a valid JSON object matching this exact schema (no markdown, no prose, just JSON):
{
  "root_cause": "<one sentence>",
  "osi_layer": "<layer name>",
  "confidence": "High" | "Medium" | "Low",
  "evidence": ["<item1>", "<item2>", ...],
  "next_command": "<single show command>",
  "fix_steps": ["<step1>", "<step2>", ...],
  "beginner_explanation": "<2-3 sentences>"
}"""


USER_PROMPT_TEMPLATE = """Troubleshoot the following Cisco network lab case:

SYMPTOM:
{symptom}

TOPOLOGY:
{topology_note}

SHOW COMMAND OUTPUTS:
{show_outputs}

Diagnose the fault and provide the structured JSON response."""



# ── Prompt loader helpers ──────────────────────────────────────────────────────

def _get_system_prompt() -> str:
    """Load system prompt from file, fall back to embedded constant."""
    text = _load_prompt("system_prompt.txt")
    return text if text else SYSTEM_PROMPT


def _get_user_template() -> str:
    """Load user prompt template from file, fall back to embedded constant."""
    text = _load_prompt("user_prompt_template.txt")
    return text if text else USER_PROMPT_TEMPLATE


# ── Agent function ─────────────────────────────────────────────────────────────


def diagnose(case: dict) -> DiagnosisResult | ErrorResult:
    api_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Prefer Gemini if its key is set
    if gemini_key and not gemini_key.startswith("your"):
        api_key = gemini_key
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model = "gemini-1.5-flash"
    elif api_key and not api_key.startswith("sk-your"):
        base_url = None  # default OpenAI
    else:
        return ErrorResult(
            error="No API key configured. Add OPENAI_API_KEY or GEMINI_API_KEY in Streamlit Secrets, or use Demo Mode."
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    except ImportError:
        return ErrorResult(error="OpenAI package not installed. Run: pip install openai")

    user_message = _get_user_template().format(
        symptom=case.get("symptom", "N/A"),
        topology_note=case.get("topology_note", "N/A"),
        show_outputs=case.get("show_outputs", "N/A"),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _get_system_prompt()},
                {"role": "user", "content": user_message},
            ],
            temperature=0.2,          # Lower temperature = more deterministic/factual
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        return ErrorResult(error=f"OpenAI API call failed: {exc}", raw_response="")

    try:
        parsed = json.loads(raw)
        result = DiagnosisResult(**parsed)
        return result
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return ErrorResult(
            error=f"Failed to parse AI response into schema: {exc}",
            raw_response=raw,
        )


def get_demo_diagnosis(case: dict) -> DiagnosisResult:
    """
    Returns a deterministic demo diagnosis for showcasing the UI without an API key.
    The content is based on the expected_fault field of the case.
    """
    concept = case.get("concept", "Network Configuration")
    osi = case.get("osi_layer", "Layer 2 - Data Link")
    expected = case.get("expected_fault", "Configuration error detected.")

    fix_map = {
        "VLAN Access Port Configuration": [
            "Switch# configure terminal",
            "Switch(config)# interface FastEthernet0/1",
            "Switch(config-if)# switchport mode access",
            "Switch(config-if)# switchport access vlan 10",
            "Switch(config-if)# interface FastEthernet0/2",
            "Switch(config-if)# switchport access vlan 10",
            "Switch(config-if)# end",
            "Switch# write memory",
        ],
        "DHCP Pool Configuration": [
            "Router# configure terminal",
            "Router(config)# ip dhcp pool LAN_POOL",
            "Router(dhcp-config)# default-router 192.168.1.1",
            "Router(dhcp-config)# end",
            "Router# write memory",
        ],
        "Default Route / Gateway of Last Resort": [
            "Router# configure terminal",
            "Router(config)# ip route 0.0.0.0 0.0.0.0 203.0.113.254",
            "Router(config)# end",
            "Router# write memory",
            "Router# ping 8.8.8.8",
        ],
    }

    return DiagnosisResult(
        root_cause=expected,
        osi_layer=osi,
        confidence="High",
        evidence=[
            "Rule checker confirmed the primary indicator",
            "Show output corroborates the fault pattern",
            "This is a common misconfiguration in this topology",
        ],
        next_command="show running-config",
        fix_steps=fix_map.get(concept, [
            "Router# configure terminal",
            "Router(config)# ! Apply the appropriate fix for this issue",
            "Router(config)# end",
            "Router# write memory",
        ]),
        beginner_explanation=(
            f"The issue is a {concept} problem. {expected} "
            "Review the show output highlighted above and apply the fix commands step by step. "
            "Always verify connectivity after making changes."
        ),
    )
