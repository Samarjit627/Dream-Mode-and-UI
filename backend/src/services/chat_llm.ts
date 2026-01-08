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

export async function generateTextResponse(masterDescription: string, userMessage: string): Promise<string> {
    if (!client) {
        if (process.env.OPENAI_API_KEY) client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })
        else return "Error: OpenAI API Key missing."
    }

    const systemPrompt = `You are an expert senior mechanical engineer.
You are answering user questions based on a PRE-GENERATED MASTER DESCRIPTION of a file.

SOURCE MATERIAL:
${masterDescription}

YOUR GOAL:
Answer the user's question accurately using ONLY the information in the Master Description.
- If the answer is in the description, state it confidently.
- If the answer is NOT in the description (e.g. a tiny detail not captured), respond: "I cannot see that detail in my current notes. Please ask me to 'look closer' or 'rescan' to check the image again."

TONE:
Professional, concise, engineering-focused.
`

    try {
        const completion = await client.chat.completions.create({
            model: "gpt-4o",
            messages: [
                { role: 'system', content: systemPrompt },
                { role: 'user', content: userMessage }
            ],
            temperature: 0.2,
        })
        return completion.choices[0]?.message?.content || "No response."
    } catch (e: any) {
        console.error("Text Chat Error:", e.message)
        return `Error: Could not generate response. (${e.message})`
    }
}
