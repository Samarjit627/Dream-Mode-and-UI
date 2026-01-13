from pydantic import BaseModel
from typing import List

class DesignRule(BaseModel):
    id: str
    name: str
    question: str # Intent Shift Question (Exact text)
    monitor_instruction: str # Judgment Pattern (What to look for)

DJKB_RULES_V1 = [
    DesignRule(
        id="visual_mass",
        name="Visual Mass vs Functional Claim",
        question="Should this feel planted and stable, or light and agile?",
        monitor_instruction="Check for top-heavy visual mass. If the object looks unstable or top-heavy, flag it as a risk to perceived stability."
    ),
    DesignRule(
        id="thin_transitions",
        name="Thin Transition Between Volumes",
        question="Is this connection meant to feel delicate or strong?",
        monitor_instruction="Identify narrow necks or connections between two larger volumes. If found, flag as a potential stress concentration or weakness."
    ),
    DesignRule(
        id="over_symmetry",
        name="Over-Expressed Symmetry",
        question="Is emotional neutrality acceptable here, or should it feel more human?",
        monitor_instruction="Check for perfect symmetry in a non-static object. If it looks too rigid or engineered-first, flag it for lack of human warmth."
    ),
    DesignRule(
        id="feature_density",
        name="Excessive Feature Density",
        question="What should visually dominate — form or function?",
        monitor_instruction="Check for many secondary features competing (buttons, vents, lines). If there is no clear visual hierarchy, flag for listener confusion."
    ),
    DesignRule(
        id="flat_surfaces",
        name="Flat Surfaces with No Relief",
        question="Is this meant to feel minimal or economical?",
        monitor_instruction="Identify large uninterrupted planar areas. If they look unfinished or cheap (lack of relief), flag it."
    ),
    DesignRule(
        id="inconsistent_radii",
        name="Inconsistent Radius Language",
        question="Should this object feel precise or forgiving?",
        monitor_instruction="Check for mixed sharp and soft transitions. If radius language is confused/inconsistent, flag for lack of coherence."
    ),
    DesignRule(
        id="proportion_drift",
        name="Proportion Drift Across Views",
        question="Which view best represents the object’s identity?",
        monitor_instruction="If multiple views are present, check if the object looks good in one but awkward in another. Flag resolved 3D intent."
    )
]

