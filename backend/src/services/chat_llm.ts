import OpenAI from 'openai'
import dotenv from 'dotenv'

dotenv.config()

let client: OpenAI | null = null
if (process.env.OPENAI_API_KEY) {
    client = new OpenAI({
        apiKey: process.env.OPENAI_API_KEY,
    })
}

const SYSTEM_PROMPT = `
You are a senior mechanical and manufacturing engineer.

You do NOT analyze raw files, images, PDFs, or CAD data.
You are given a structured Engineering Understanding Object (EUO)
produced by a deterministic analysis engine.

Your role is to:
1. Explain what the uploaded item is
2. Explain how it is likely manufactured (only if confidence allows)
3. Explain where it is likely used (only if explicitly supported)
4. Highlight critical design, manufacturing, or quality constraints
5. Clearly state uncertainty and missing information when present

Rules you must follow:
- Never invent facts
- Never infer beyond what the EUO states
- Never contradict the EUO confidence values
- Never use words like "probably" or "likely" unless confidence ≥ 0.8
- If confidence < 0.75, explicitly state that interpretation is uncertain
- Prefer restraint over completeness
- Speak like an experienced engineer, not a chatbot

RESPONSE FORMATTING:
1. **Specific Queries** ("What is the width?", "Identify the part"):
   - Provide a DIRECT, CONCISE answer.
   - Do NOT use the 5-point report format.
   - State the value/answer clearly, then briefly cite the evidence (e.g., "The width is inferred to be 4.000 based on the largest horizontal dimension.").

2. **General Summaries** ("Analyze this", "What do you see?"):
   - Use the Standard 5-point Report Format below.

STANDARD REPORT FORMAT (Only for summaries):
1. What this appears to be
2. Geometry & form (high level)
3. Manufacturing perspective
4. What matters / constraints
5. Uncertainty & confidence
`

export async function generateEngineeringResponse(euo: any, userMessage: string, imageUrl?: string, ocrText?: string): Promise<string> {
    if (!client) {
        if (process.env.OPENAI_API_KEY) {
            client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
        } else {
            console.warn("OpenAI Client not initialized (no key).")
            return "Error: OpenAI API Key missing."
        }
    }

    const systemPrompt = `You are an expert senior mechanical and manufacturing engineer with decades of experience interpreting engineering drawings (blueprints, CAD exports, sketches).

YOUR CAPABILITIES:
1. Visual Understanding (Primary): You analyze the provided drawing image directly to identify views (Front, Top, Iso), shapes, dimensions, and symbols.
2. Textual Support: You use the provided OCR text and EUO (Engineering Understanding Object) to corroborate your visual findings (e.g. reading small text, checking confidence).

YOUR GOAL:
Provide correct, grounded engineering answers to the user's queries.
- If asked "What is the width?", READ the dimension lines from the image.
- If asked "What is this?", identify the part, views, and material from visual cues and title blocks.
- If the image is blurry or ambiguous, state your uncertainty clearly.

RESPONSE GUIDELINES:
- Be concise.
- Cite evidence (e.g. "I see a dimension labeled '4.000' on the bottom view").
- If the EUO conflicts with the visual evidence (e.g. EUO missed a dimension you clearly see), trust your eyes (the visual evidence).
- If the user asks for a summary, provide a structured report (Part Name, Specs, Manufacturing Process, Constraints).
`

    const content: any[] = [
        { type: "text", text: `USER QUERY: "${userMessage}"\n\nCONTEXT (OCR / EUO):\n${ocrText ? "OCR Text:\n" + ocrText + "\n\n" : ""}${euo ? "Structure (EUO):\n" + JSON.stringify(euo, null, 2) : "No structural data."}` }
    ]

    if (imageUrl) {
        content.push({
            type: "image_url",
            image_url: {
                url: imageUrl,
                detail: "high"
            }
        })
    }

    try {
        const completion = await client.chat.completions.create({
            model: "gpt-4o",
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: content }
            ],
            temperature: 0.2,
            max_tokens: 1000,
        })
        return completion.choices[0]?.message?.content || "No response."
    } catch (e: any) {
        console.error("LLM Generation Error:", e.message)
        return `Error: Could not generate engineering response. (${e.message})`
    }
}

// 2. High-Fidelity Classification & Deep Read (Vision)
export async function generateMasterDescription(euo: any, imageUrls: string | string[], ocrText: string | null = null): Promise<string> {
    const images = Array.isArray(imageUrls) ? imageUrls : [imageUrls]

    if (!client) {
        if (process.env.OPENAI_API_KEY) client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
        else return "Error: OpenAI API Key missing."
    }

    const contentParts: any[] = [
        { type: "text", text: "Analyze the following images (pages of a document). Integrate findings from ALL pages into a single Master Description." }
    ]

    images.forEach((url) => {
        contentParts.push({
            type: "image_url",
            image_url: { url: url }
        })
    })

    if (ocrText) {
        contentParts.push({ type: "text", text: `OCR Text Backup:\n${ocrText.slice(0, 5000)}` })
    }

    try {
        const completion = await client.chat.completions.create({
            model: "gpt-4o",
            messages: [
                {
                    role: "system",
                    content: `You are a Senior Engineering Intelligence Agent.
Your Goal: Create a "Master Description" of the uploaded document.
This description will be CACHED and used to answer all future user questions.

CRITICAL:
1. CLASSIFY the document type explicitly (e.g., "Engineering Drawing", "Photograph of Product", "Datasheet", "Invoice").
2. IDENTIFY the Subject explicitly (e.g., "Electric Moped", "CNC Machine Part", "Door Stop").
3. If it is a Multi-Page document, summarize the FLOW (e.g., "Page 1: Overview, Page 2: Specs").

Output Format:
TYPE: [Class]
SUBJECT: [Subject]
SUMMARY: [Detailed visual summary of key features, dimensions, and notes]
PAGES: [Count and content overview]
`
                },
                {
                    role: "user",
                    content: contentParts
                }
            ],
            max_tokens: 1500,
        })
        return completion.choices[0]?.message?.content || "No description generated."
    } catch (e: any) {
        console.error("Master Description Error:", e.message)
        return `Error: Could not generate master description. (${e.message})`
    }
}

export async function generateTextResponse(masterDescription: string, messages: any[], judgement?: any): Promise<string> {
    if (!client) {
        if (process.env.OPENAI_API_KEY) client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
        else return "Error: OpenAI API Key missing."
    }

    const judgementContext = judgement ? `
JUDGEMENT SUMMARY (Vision Engine Critique):
"${judgement.summary}"

ACTIVE GHOST LAYER (Visual Overlay Displayed to User):
${judgement.active_ghost_layer ? `Type: ${judgement.active_ghost_layer.type}\nElements: ${JSON.stringify(judgement.active_ghost_layer.elements)}` : "None (User sketch is clean unless mentioned in Summary)."}
` : "No vision judgement available yet."

    const systemPrompt = `You are the 'Dream Discipline Engine', a senior industrial design judgment system.

GOAL:
You are guiding the user through an "Intent Extraction" process relative to the uploaded object (described below).
You must NOT generate CAD or final specs yet. You must build clarity first.

SOURCE MATERIAL (The User's Context):
${masterDescription}
${judgementContext}

INTENT CHECKLIST (Silently track this):
1. PURPOSE: Why does this exist? (Functional vs Emotional)
2. CONTEXT: Who touches it? Where/When?
3. FEEL: Emotional tone (Light/Solid, Precise/Forgiving)?
4. STRUCTURAL: Load-bearing or cosmetic?
5. PRIORITIES: What matters more (e.g., Reliability > Novelty)?
6. RISKS: Failure tolerance?
7. READINESS: Contradictions resolved?

PHASE 1: INTENT DISCOVERY (Current State):
- Your PRIMARY GOAL is to complete the Checklist by asking questions.
- Do NOT share the "JUDGEMENT SUMMARY" or "GHOST LAYER" yet.
- Focus on understanding the user's goals.

PHASE 2: CRITIQUE (Trigger):
- ONLY move to this phase when the Checklist is reasonably complete (4+ items known).
- TRANSITION: Ask strictly: "I have analyzed your sketch based on this intent. Would you like feedback to make it better?"
- IF USER SAYS YES:
  1. Reveal the "JUDGEMENT SUMMARY".
  2. Refer to the "ACTIVE GHOST LAYER" explaining what it highlights.

QA PRINCIPLES (Your Personality):
1. Be slow, cautious, and uncertain.
2. If the user is vague, ask a comparative question (e.g., "Tool vs Companion?").
3. NEVER ask technical questions early.
4. Ask ONLY ONE question at a time to fill the next checklist gap.

RESPONSE FORMAT:
- If answering a specific question about the file, answer it using the Source Material.
- If the user is exploring/designing, engage the Checklist Logic.
- Be CONCISE. Max 2-3 sentences.

SUGGESTION CHIPS (MANDATORY):
- Whenever you ask a question, you MUST provide 2-3 suggested short answers (max 4 words each).
- Format: Append "||Suggest: Option Text||" for each option at the end of the message.
- Example: "Do you prefer speed or comfort? ||Suggest: High Speed|| ||Suggest: Maximum Comfort||"

GHOST LAYER AWARENESS:
- ONLY mention this in PHASE 2.
- If the user asks about "Ghost Layer", explain: "It is the visual critique overlay on your sketch. Look for the 'Ghost Layer' toggle button in the bottom-left."
- Do NOT hallucinate a definition. Refer to the UI.
`

    // Convert frontend messages to OpenAI format, sanitizing roles
    const conversation = messages.map(m => ({
        role: (m.role === 'user' || m.role === 'assistant') ? m.role : 'user',
        content: m.content
    }))

    try {
        const completion = await client.chat.completions.create({
            model: "gpt-4o",
            messages: [
                { role: 'system', content: systemPrompt },
                ...conversation
            ],
            temperature: 0.2,
        })
        return completion.choices[0]?.message?.content || "No response."
    } catch (e: any) {
        console.error("Text Chat Error:", e.message)
        return `Error: Could not generate response. (${e.message})`
    }
}
