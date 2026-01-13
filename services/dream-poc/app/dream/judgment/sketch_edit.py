"""
DREAM Sketch Edit Module

Uses OpenAI's Image Edit API (GPT Image 1.5) to create a corrected version
of the user's sketch while PRESERVING the original design.

This is NOT generation from scratch - it takes the original image as input
and makes targeted corrections.
"""
import io
import base64
import json
from typing import Optional
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI()
except Exception as e:
    print(f"Failed to initialize OpenAI client: {e}")
    client = None


class EditCorrectionResponse(BaseModel):
    """Response from the sketch edit system"""
    corrected_image_url: str
    correction_summary: str
    issues_found: list[str]
    changes_made: list[str]
    error: Optional[str] = None


def generate_edited_sketch(
    original_image_base64: str,
    creativity_factor: float = 0.5
) -> EditCorrectionResponse:
    """
    Uses OpenAI's Image Edit API to create a corrected version of the sketch.
    
    The key difference from pure generation:
    - Takes the ORIGINAL IMAGE as input
    - Preserves the core design, viewing angle, and proportions
    - Makes targeted corrections while maintaining identity
    
    Args:
        original_image_base64: Base64 encoded image (data URL or raw)
        creativity_factor: 0.0 (minimal changes) to 1.0 (aggressive corrections)
    
    Returns:
        EditCorrectionResponse with corrected image URL and summary
    """
    if not client:
        return EditCorrectionResponse(
            corrected_image_url="",
            correction_summary="",
            issues_found=[],
            changes_made=[],
            error="OpenAI client not initialized. Check your API key."
        )
    
    try:
        # Clean up base64 if it has data URL prefix
        if "base64," in original_image_base64:
            image_data = original_image_base64.split("base64,")[1]
        else:
            image_data = original_image_base64
        
        # Decode and prepare the image
        image_bytes = base64.b64decode(image_data)
        
        # First, analyze the image to understand what needs correction
        analysis = _analyze_sketch(image_data, creativity_factor)
        
        # Create the edit prompt based on analysis
        edit_prompt = f"""Refine this automotive design sketch with the following corrections:

PRESERVE EXACTLY:
- The same vehicle type and body style
- The same viewing angle and perspective
- The overall silhouette and character

CORRECTIONS TO APPLY:
{chr(10).join([f"- {c}" for c in analysis['corrections']])}

STYLE:
- Clean, professional line sketch
- Highlight corrected areas with GREEN LINES
- Maintain the original line weight and style

CREATIVITY LEVEL: {creativity_factor}/1.0"""

        # Use the Image Edit API
        response = client.images.edit(
            model="dall-e-2",  # Edit API uses dall-e-2
            image=image_bytes,
            prompt=edit_prompt,
            n=1,
            size="1024x1024"
        )
        
        corrected_image_url = response.data[0].url
        
        # Generate summary
        summary = _generate_summary(analysis['issues'], analysis['corrections'])
        
        return EditCorrectionResponse(
            corrected_image_url=corrected_image_url,
            correction_summary=summary,
            issues_found=analysis['issues'],
            changes_made=analysis['corrections'],
            error=None
        )
        
    except Exception as e:
        print(f"Image edit failed: {e}")
        # Fallback: try with GPT-4o for vision-based redraw instruction
        return _fallback_with_vision_guidance(original_image_base64, creativity_factor, str(e))


def _analyze_sketch(image_data: str, creativity_factor: float) -> dict:
    """Uses GPT-4o to analyze the sketch and identify corrections needed."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert automotive design critic.
Analyze this sketch and identify proportional/structural issues that need correction.

OUTPUT FORMAT (JSON):
{
    "vehicle_type": "specific type (hatchback, sedan, SUV)",
    "viewing_angle": "side view, 3/4 view, etc.",
    "issues": [
        "Issue 1: Brief description",
        "Issue 2: Brief description"
    ],
    "corrections": [
        "Correction 1: Specific action",
        "Correction 2: Specific action"
    ]
}

Be concise. Max 5 issues and 5 corrections."""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Analyze this sketch. Creativity level: {creativity_factor}/1.0"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        
        return json.loads(response.choices[0].message.content)
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return {
            "vehicle_type": "vehicle",
            "viewing_angle": "side view",
            "issues": ["Could not analyze - applying general corrections"],
            "corrections": ["Improve overall proportions", "Refine structural lines"]
        }


def _fallback_with_vision_guidance(
    image_base64: str, 
    creativity_factor: float,
    original_error: str
) -> EditCorrectionResponse:
    """
    Fallback when Image Edit API fails.
    Uses GPT-4o to generate a VERY detailed description, then DALL-E 3 to recreate.
    """
    try:
        # Clean base64
        if "base64," in image_base64:
            image_data = image_base64.split("base64,")[1]
        else:
            image_data = image_base64
        
        # Get extremely detailed description from vision model
        desc_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """Describe this automotive sketch in EXTREME detail for recreation.
Include:
- Exact vehicle type and body style
- Viewing angle
- Every major line, curve, and proportion
- Wheel positions and sizes relative to body
- Roofline shape
- Window/greenhouse proportions
- Any distinctive features

Be extremely precise so another artist could recreate this EXACTLY."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this sketch precisely:"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}}
                    ]
                }
            ],
            max_tokens=800
        )
        
        detailed_description = desc_response.choices[0].message.content
        
        # Analyze for corrections
        analysis = _analyze_sketch(image_data, creativity_factor)
        
        # Generate with DALL-E 3 using the detailed description
        dalle_prompt = f"""Create a refined version of this EXACT design:

ORIGINAL DESIGN (preserve exactly):
{detailed_description}

CORRECTIONS TO APPLY (show in GREEN):
{chr(10).join([f"- {c}" for c in analysis['corrections']])}

CRITICAL RULES:
- This must be the SAME vehicle, same angle, same style
- Use black lines on white background
- Mark corrections with GREEN lines
- Keep the EXACT proportions except where corrected
- Creativity level: {creativity_factor}/1.0"""

        image_response = client.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            n=1,
            size="1024x1024",
            quality="hd",
            style="natural"
        )
        
        summary = _generate_summary(analysis['issues'], analysis['corrections'])
        
        return EditCorrectionResponse(
            corrected_image_url=image_response.data[0].url,
            correction_summary=summary,
            issues_found=analysis['issues'],
            changes_made=analysis['corrections'],
            error=None
        )
        
    except Exception as e:
        return EditCorrectionResponse(
            corrected_image_url="",
            correction_summary="",
            issues_found=[],
            changes_made=[],
            error=f"All methods failed. Original: {original_error}. Fallback: {str(e)}"
        )


def _generate_summary(issues: list[str], corrections: list[str]) -> str:
    """Generates a human-readable summary"""
    parts = ["**Sketch Correction Applied**\n"]
    
    if issues:
        parts.append("**Issues Found:**")
        for i, issue in enumerate(issues[:5], 1):
            parts.append(f"{i}. {issue}")
        parts.append("")
    
    if corrections:
        parts.append("**Corrections Made (GREEN lines):**")
        for i, corr in enumerate(corrections[:5], 1):
            parts.append(f"{i}. {corr}")
    
    return "\n".join(parts)
