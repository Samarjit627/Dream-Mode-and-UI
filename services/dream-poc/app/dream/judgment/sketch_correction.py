"""
Sketch Correction Module

Uses OpenAI DALL-E-3 to generate a corrected version of a user's sketch
with visual highlights of the changes and a detailed summary.
"""
import os
import base64
import json
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI()
except Exception as e:
    print(f"Failed to initialize OpenAI client for sketch correction: {e}")
    client = None


class CorrectionResponse(BaseModel):
    """Response model for sketch correction"""
    corrected_image_url: str
    correction_summary: str
    issues_found: list[str]
    changes_made: list[str]
    error: Optional[str] = None


def generate_corrected_sketch(
    original_image_base64: str,
    object_type: str,
    image_width: int,
    image_height: int,
    creativity_factor: float = 0.3,
    initial_critique: str = ""
) -> CorrectionResponse:
    """
    Generates a corrected version of the user's sketch using DALL-E-3.
    
    Args:
        original_image_base64: Base64 encoded image (data URL or raw)
        object_type: Type of object identified (e.g., "car", "chair")
        image_width: Width of original image
        image_height: Height of original image
        creativity_factor: 0.0 (minimal changes) to 1.0 (full creative license)
        initial_critique: The design critique from the analysis phase
    
    Returns:
        CorrectionResponse with corrected image URL and detailed summary
    """
    if not client:
        return CorrectionResponse(
            corrected_image_url="",
            correction_summary="",
            issues_found=[],
            changes_made=[],
            error="OpenAI client not initialized. Check your API key."
        )
    
    # Step 1: First, use GPT-4o to analyze and describe what corrections are needed
    try:
        # Clean up base64 if it has data URL prefix
        if "base64," in original_image_base64:
            image_data = original_image_base64.split("base64,")[1]
        else:
            image_data = original_image_base64
            
        # Get detailed correction plan from vision model
        analysis_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert industrial design critic and sketch analyst.
Your task is to analyze a design sketch in EXTREME DETAIL and provide specific corrections.

CRITICAL: You MUST preserve the exact identity of the vehicle/object. Do NOT change:
- The vehicle TYPE (hatchback stays hatchback, sedan stays sedan, SUV stays SUV)
- The viewing ANGLE (side view, 3/4 view, front view, etc.)
- The overall SILHOUETTE and proportions
- The design LANGUAGE (sporty, elegant, rugged, etc.)

OUTPUT FORMAT (JSON):
{
    "exact_vehicle_type": "specific vehicle type (e.g., '5-door hatchback', 'compact SUV', '4-door sedan')",
    "body_style": "detailed body description (e.g., 'sleek hatchback with sloping rear roofline')",
    "viewing_angle": "exact viewing angle (e.g., 'side profile view from left', '3/4 front view')",
    "silhouette_description": "detailed description of the overall shape and silhouette",
    "design_language": "the design character (sporty, elegant, rugged, futuristic, classic)",
    "key_features": ["list of distinctive design elements that MUST be preserved"],
    "issues_found": [
        "Issue 1: Description of proportion problem",
        "Issue 2: Description of structural weakness"
    ],
    "corrections_needed": [
        "Correction 1: Specific fix with location",
        "Correction 2: Another specific fix"
    ]
}"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this {object_type} sketch (dimensions: {image_width}x{image_height}px).

Previous critique: {initial_critique}

Creativity level: {creativity_factor}/1.0 (0=minimal changes, 1=full creative license)

Identify ALL issues with proportions, structure, symmetry, and visual hierarchy.
Then describe EXACTLY what the corrected version should look like.
Be specific about locations (top, bottom, left, right, center) and proportions."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=1000
        )
        
        analysis = json.loads(analysis_response.choices[0].message.content)
        
        # Extract detailed vehicle information
        exact_vehicle_type = analysis.get("exact_vehicle_type", object_type)
        body_style = analysis.get("body_style", "")
        viewing_angle = analysis.get("viewing_angle", "side profile view")
        silhouette_description = analysis.get("silhouette_description", "")
        design_language = analysis.get("design_language", "modern")
        key_features = analysis.get("key_features", [])
        issues_found = analysis.get("issues_found", [])
        corrections_needed = analysis.get("corrections_needed", [])
        
    except Exception as e:
        print(f"Analysis phase failed: {e}")
        exact_vehicle_type = object_type
        body_style = ""
        viewing_angle = "side profile view"
        silhouette_description = ""
        design_language = "modern"
        key_features = []
        issues_found = ["Analysis failed - using generic correction"]
        corrections_needed = ["General proportion and structure improvement"]
    
    # Step 2: Generate the corrected sketch using DALL-E-3
    try:
        # Build key features string
        features_str = "\n".join([f"- {f}" for f in key_features]) if key_features else "- No specific features noted"
        
        # Craft the DALL-E prompt with EXTREME specificity
        dalle_prompt = f"""CRITICAL: Create a corrected design sketch of EXACTLY a {exact_vehicle_type}.

VEHICLE IDENTITY (DO NOT CHANGE THESE):
- Vehicle Type: {exact_vehicle_type}
- Body Style: {body_style}
- Viewing Angle: {viewing_angle}
- Design Language: {design_language}

SILHOUETTE TO MATCH:
{silhouette_description}

KEY FEATURES TO PRESERVE:
{features_str}

STYLE:
- Black and white LINE SKETCH on white background
- Clean, confident pencil/pen strokes
- Technical automotive design sketch style
- Highlight corrections with GREEN LINES (#00FF00)

CORRECTIONS TO APPLY (show in GREEN):
{chr(10).join([f"- {c}" for c in corrections_needed])}

IMPORTANT:
- This MUST be a {exact_vehicle_type}, NOT any other vehicle type
- The viewing angle MUST be: {viewing_angle}
- Preserve the overall silhouette and proportions
- Only use green to highlight the corrected areas"""

        image_response = client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            n=1,
            size="1024x1024",
            quality="hd",
            style="natural"  # More realistic/technical style
        )
        
        corrected_image_url = image_response.data[0].url
        
        # Step 3: Generate a human-readable summary
        summary = _generate_summary(exact_vehicle_type, issues_found, corrections_needed)
        
        return CorrectionResponse(
            corrected_image_url=corrected_image_url,
            correction_summary=summary,
            issues_found=issues_found,
            changes_made=corrections_needed,
            error=None
        )
        
    except Exception as e:
        print(f"DALL-E generation failed: {e}")
        return CorrectionResponse(
            corrected_image_url="",
            correction_summary="",
            issues_found=issues_found,
            changes_made=corrections_needed,
            error=f"Image generation failed: {str(e)}"
        )


def _describe_aspect_ratio(width: int, height: int) -> str:
    """Describes the aspect ratio in natural language"""
    ratio = width / height if height > 0 else 1
    if ratio > 1.5:
        return "wide landscape orientation (horizontal)"
    elif ratio < 0.67:
        return "tall portrait orientation (vertical)"
    else:
        return "roughly square orientation"


def _generate_summary(object_type: str, issues: list[str], corrections: list[str]) -> str:
    """Generates a human-readable summary of the corrections"""
    summary_parts = [f"**Corrected Sketch: {object_type.title()}**\n"]
    
    if issues:
        summary_parts.append("**Issues Identified:**")
        for i, issue in enumerate(issues[:5], 1):  # Limit to 5
            summary_parts.append(f"{i}. {issue}")
        summary_parts.append("")
    
    if corrections:
        summary_parts.append("**Corrections Applied (shown in GREEN):**")
        for i, correction in enumerate(corrections[:5], 1):  # Limit to 5
            summary_parts.append(f"{i}. {correction}")
        summary_parts.append("")
    
    summary_parts.append("*Toggle the overlay to compare with your original sketch.*")
    
    return "\n".join(summary_parts)
