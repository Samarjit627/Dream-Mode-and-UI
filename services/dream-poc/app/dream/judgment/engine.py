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
    type: str
    elements: List[dict]
    lines: Optional[List[dict]] = None

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

    system_prompt = f"""You are the 'Dream Discipline Engine', a senior industrial design judgment system.
Your goal is to evaluate a sketch against specific design rules with extreme restraint and precision.

QA PRINCIPLES:
1. Be CONCISE. Summary max 3 sentences.
2. If geometry is ambiguous, ask a meaning-based question. Do NOT guess.
3. If symmetry is broken, ask if it is intentional.

4. GHOST LAYER GENERATION (MANDATORY):
   - If ANY rule fails (Mass, Structure, etc.), you MUST generate an "active_ghost_layer".
   - Do NOT be hesitant. If there is a problem, visualize it.
   - Priority: Structural > Visual Mass > Symmetry > Hierarchy.
   - Coordinates:
     - x, y: PERCENTAGE (0-100) of the image width/height. (e.g. x=50, y=50 is center).
     - r: PERCENTAGE (0-100) radius relative to image width.
   - Types:
     - "lines": For structural corrections. [points=[{{x,y}}...], color, width].
     - "mass": For proportion issues. Soft blobs [{{x, y, r, opacity=0.4}}].
     - "structure": For weak connections. [{{x, y, r, opacity=0.5}}].
     - "symmetry": For axis.
   - Output format examples:
     "active_ghost_layer": {{
       "type": "lines",
       "lines": [
         {{ "points": [{{"x": 10, "y": 10}}, {{"x": 100, "y": 100}}], "color": "#00ff00", "width": 3 }}
       ],
       "elements": []
     }}
     OR
     "active_ghost_layer": {{
       "type": "mass",
       "elements": [
         {{ "shape": "blob", "x": 50, "y": 50, "r": 30, "opacity": 0.4 }}
       ]
     }}

YOUR TASK:
1. Evaluate the design against the following RULES.
2. For each rule, determine status (pass/fail/warn).
3. Generate active_ghost_layer for the most critical failure (prefer vector lines for structure).

Rules to Evaluate:
{rules_text}

OUTPUT FORMAT (JSON ONLY):
{{
  "summary": "Concise critique (max 3 sentences).",
  "active_ghost_layer": {{ ... }},
  "results": [
    {{ "rule_id": "rule_name", "status": "pass/fail/warn", "reasoning": "Why it passed/failed", "question": "Optional clarification" }}
  ]
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
