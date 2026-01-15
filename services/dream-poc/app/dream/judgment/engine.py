import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from .rules import DJKB_RULES_V1

load_dotenv()

class RuleResult(BaseModel):
    rule_id: str # Matches 'rule_id' in JSON
    status: str  # pass, fail, warn
    reasoning: str = "" # Optional, default to empty
    question: Optional[str] = None

class ActiveGhostLayer(BaseModel):
    type: Optional[str] = None
    elements: Optional[List[dict]] = None
    lines: Optional[List[dict]] = None
    status: Optional[str] = None  # Allow status-only objects

class JudgmentResponse(BaseModel):
    results: List[RuleResult]
    summary: str
    active_ghost_layer: Optional[ActiveGhostLayer] = None

# Client Initialization
try:
    client = OpenAI()
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    client = None

def get_client():
    return client

def judge_design(image_base64: str) -> JudgmentResponse:
    if not client:
        return JudgmentResponse(results=[], summary="Error: OpenAI client not initialized (Check API Key).")

    # ... (client check same)
    
    # 2. Prepare Rules Text
    rules_text = "\n".join([f"{r.id}: {r.name} - {r.monitor_instruction}" for r in DJKB_RULES_V1])

    system_prompt = f"""You are the Design Discipline Engine, committed to maximal truthfulness.

Your role is to VERIFY proportion discipline in the generated sketch against the original.
You do NOT redesign, suggest, or improve.

Evaluate the GENERATED sketch against the following categories ONLY:

1. Wheel dominance & stance
2. Wheelbase perception
3. Vertical mass balance
4. Rear mass quietness
5. Perspective discipline

For EACH category:
- Mark PASS or FAIL
- If FAIL, explain in ONE sentence based solely on observable differences.
- Assign severity: low / medium / high

ACTIVE_GHOST_LAYER RULE:
- Generate an active_ghost_layer ONLY if:
  • severity is HIGH
  • and the failure affects overall proportion reading

STRICT RULES:
- Do NOT praise the sketch.
- Do NOT suggest stylistic changes.
- Do NOT introduce new proportion rules.
- Be minimal, factual, and strict. If verification is impossible due to image issues, mark all as 'UNVERIFIABLE'.

Rules to Evaluate (Reference):
{rules_text}

Output structured JSON only:
{{
  "wheel_dominance": {{ "status": "PASS" }},
  "wheelbase": {{ "status": "FAIL", "severity": "high", "reason": "..." }},
  "vertical_mass": {{ ... }},
  "rear_mass": {{ ... }},
  "perspective": {{ ... }},
  "active_ghost_layer": {{ ... }},
  "summary": "One sentence summary."
}}
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Evaluate this design sketch."},
                    {"type": "image_url", "image_url": {"url": image_base64}}
                ]}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        content = response.choices[0].message.content
        if not content:
            return JudgmentResponse(results=[], summary="No response from LLM")
            
        data = json.loads(content)
        return JudgmentResponse(
            results=[RuleResult(**r) for r in data.get("results", [])],
            summary=data.get("summary", "Analysis complete."),
            active_ghost_layer=data.get("active_ghost_layer")
        )

    except Exception as e:
        print(f"DJKB Error: {e}")
        return JudgmentResponse(results=[], summary=f"Error: {str(e)}")
