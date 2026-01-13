import { Router, Request, Response, NextFunction } from 'express'
import { maybeProxyPost } from '../services/proxy.js'
import { generateEngineeringResponse, generateMasterDescription, generateTextResponse } from '../services/chat_llm.js'

const r = Router()

type ChatMessage = { role: 'user' | 'assistant' | 'system'; content: string }

type ChatRequestBody = {
  message?: string
  messages?: ChatMessage[]
  context?: any
}

type GroundedIntent =
  | 'summary'
  | 'ocr'
  | 'contacts'
  | 'dimensions'
  | 'objects'
  | 'engineering'
  | 'unknown'

function buildEngineeringUnderstandingReport(body: ChatRequestBody): string {
  const ctx = body?.context || {}
  const understanding = (ctx?.understanding && typeof ctx.understanding === 'object') ? ctx.understanding : null
  const euoFromUnified = (understanding && (understanding as any).kind === 'engineering_drawing') ? (understanding as any).payload : null
  const euo = (euoFromUnified && typeof euoFromUnified === 'object')
    ? euoFromUnified
    : ((ctx?.engineeringUnderstanding && typeof ctx.engineeringUnderstanding === 'object') ? ctx.engineeringUnderstanding : null)
  const judgement = (ctx?.judgement && typeof ctx.judgement === 'object') ? ctx.judgement : null
  if (!euo) return ''

  const summaryReady = Boolean((euo as any)?.summary_ready)
  if (!summaryReady) {
    return "There isn’t enough reliable information to summarize this confidently."
  }

  const doc = ((euo as any).document_classification && typeof (euo as any).document_classification === 'object') ? (euo as any).document_classification : {}
  const part = ((euo as any).part_identity && typeof (euo as any).part_identity === 'object') ? (euo as any).part_identity : {}
  const geom = ((euo as any).geometry_understanding && typeof (euo as any).geometry_understanding === 'object') ? (euo as any).geometry_understanding : {}
  const ann = ((euo as any).annotation_understanding && typeof (euo as any).annotation_understanding === 'object') ? (euo as any).annotation_understanding : {}
  const intent = ((euo as any).manufacturing_intent && typeof (euo as any).manufacturing_intent === 'object') ? (euo as any).manufacturing_intent : {}
  const constraints = ((euo as any).constraints && typeof (euo as any).constraints === 'object') ? (euo as any).constraints : {}
  const risks = Array.isArray((euo as any).quality_risks) ? (euo as any).quality_risks : []
  const std = ((euo as any).standards_and_compliance && typeof (euo as any).standards_and_compliance === 'object') ? (euo as any).standards_and_compliance : {}
  const unc = ((euo as any).uncertainty && typeof (euo as any).uncertainty === 'object') ? (euo as any).uncertainty : {}
  const conf = ((euo as any).confidence && typeof (euo as any).confidence === 'object') ? (euo as any).confidence : {}

  const evidence = Array.isArray((euo as any)._evidence) ? (euo as any)._evidence : []

  const lines: string[] = []

  lines.push('1) What this is')
  lines.push('')
  lines.push('Document type: engineering drawing')
  const docType = String((doc as any)?.document_type || '').trim()
  const industryHint = String((doc as any)?.industry_hint || '').trim()
  const stdHint = String((doc as any)?.drawing_standard_hint || '').trim()
  if (docType && docType !== 'UNKNOWN') lines.push(`EUO document_type: ${docType}`)
  if (industryHint && industryHint !== 'UNKNOWN') lines.push(`Industry hint: ${industryHint}`)
  if (stdHint && stdHint !== 'UNKNOWN') lines.push(`Drawing standard hint: ${stdHint}`)

  const partName = String((part as any)?.name || '').trim()
  const internalId = String((part as any)?.internal_id || '').trim()
  const partClass = String((part as any)?.classification || '').trim()
  const funcRole = String((part as any)?.functional_role || '').trim()
  if (partName) lines.push(`Part: ${partName}`)
  if (internalId) lines.push(`Internal ID: ${internalId}`)
  if (partClass && partClass !== 'UNKNOWN') lines.push(`Classification: ${partClass}`)
  if (funcRole && funcRole !== 'UNKNOWN') lines.push(`Functional role: ${funcRole}`)
  if (judgement?.summary) {
    lines.push('')
    lines.push('Evidence (system judgement):')
    lines.push(String(judgement.summary).trim())
  }

  lines.push('')
  lines.push('2) How it is made')
  lines.push('')
  const proc = String((intent as any)?.likely_process || '').trim()
  const matClass = String((intent as any)?.material_class || '').trim()
  const scale = String((intent as any)?.production_scale || '').trim()
  const post = Array.isArray((intent as any)?.post_processes) ? (intent as any).post_processes.map(String).filter(Boolean) : []
  if (matClass && matClass !== 'UNKNOWN') lines.push(`Material class: ${matClass}`)
  if (proc && proc !== 'UNKNOWN') lines.push(`Likely process: ${proc}`)
  if (scale && scale !== 'UNKNOWN') lines.push(`Production scale: ${scale}`)
  if (post.length) lines.push(`Post-processes: ${post.slice(0, 8).join(', ')}`)
  if ((!proc || proc === 'UNKNOWN') && (!matClass || matClass === 'UNKNOWN')) {
    lines.push('Not enough reliable signals in the artifact to state material/process confidently.')
  }

  lines.push('')
  lines.push('3) Where it is likely used')
  lines.push('')
  if (industryHint && industryHint !== 'UNKNOWN') lines.push(`Industry: ${industryHint}`)
  if (!industryHint || industryHint === 'UNKNOWN') lines.push('Industry: unknown (not confidently grounded).')

  lines.push('')
  lines.push('4) What matters in manufacturing / design')
  lines.push('')
  const dimsPresent = Boolean((ann as any)?.dimensions_present)
  const tolClass = String((ann as any)?.tolerance_class || '').trim()
  const surfReqs = Array.isArray((ann as any)?.surface_requirements) ? (ann as any).surface_requirements.map(String).filter(Boolean) : []
  const procNotes = Array.isArray((ann as any)?.process_notes) ? (ann as any).process_notes.map(String).filter(Boolean) : []
  lines.push(`Annotations: dimensions_present=${dimsPresent ? 'yes' : 'no'}${tolClass ? `, tolerance_class=${tolClass}` : ''}`)
  if (surfReqs.length) lines.push(`Surface requirements: ${surfReqs.slice(0, 10).join(', ')}`)
  if (procNotes.length) lines.push(`Process notes: ${procNotes.slice(0, 10).join(', ')}`)

  const fcs = Array.isArray((constraints as any)?.functional_constraints) ? (constraints as any).functional_constraints.map(String).filter(Boolean) : []
  const ccs = Array.isArray((constraints as any)?.cosmetic_constraints) ? (constraints as any).cosmetic_constraints.map(String).filter(Boolean) : []
  const rcs = Array.isArray((constraints as any)?.regulatory_constraints) ? (constraints as any).regulatory_constraints.map(String).filter(Boolean) : []
  if (fcs.length || ccs.length || rcs.length) {
    lines.push('Constraints:')
    if (fcs.length) lines.push(`- Functional: ${fcs.slice(0, 10).join(', ')}`)
    if (ccs.length) lines.push(`- Cosmetic: ${ccs.slice(0, 10).join(', ')}`)
    if (rcs.length) lines.push(`- Regulatory: ${rcs.slice(0, 10).join(', ')}`)
  } else {
    lines.push('Constraints: none confidently extracted yet.')
  }

  if (risks.length) {
    lines.push('')
    lines.push('Quality / manufacturing risks:')
    for (const rr of risks.slice(0, 6)) {
      const rt = String((rr as any)?.risk_type || '').trim()
      const d = String((rr as any)?.description || '').trim()
      const sev = String((rr as any)?.severity || '').trim()
      if (!rt && !d) continue
      lines.push(`- ${rt}${sev ? ` (${sev})` : ''}${d ? `: ${d}` : ''}`)
    }
  }

  const refStd = Array.isArray((std as any)?.referenced_standards) ? (std as any).referenced_standards.map(String).filter(Boolean) : []
  const comp = String((std as any)?.compliance_domain || '').trim()
  if (refStd.length || (comp && comp !== 'UNKNOWN')) {
    lines.push('')
    lines.push('Standards / compliance:')
    if (refStd.length) lines.push(`- Referenced: ${refStd.slice(0, 12).join(', ')}`)
    if (comp && comp !== 'UNKNOWN') lines.push(`- Domain: ${comp}`)
  }

  if (evidence.length) {
    const keep = evidence
      .filter((e: any) => e && typeof e === 'object' && (e.claim || e.value))
      .slice(0, 6)
    if (keep.length) {
      lines.push('')
      lines.push('Grounding snippets (audit):')
      for (const e of keep) {
        const claim = String((e as any).claim || '').trim()
        const val = String((e as any).value || '').trim()
        const ec = (typeof (e as any).confidence === 'number') ? Number((e as any).confidence).toFixed(2) : ''
        lines.push(`- ${claim}${val ? ` = ${val}` : ''}${ec ? ` (conf=${ec})` : ''}`)
        const sn = Array.isArray((e as any).snippets) ? (e as any).snippets : []
        for (const s of sn.slice(0, 3)) {
          const ss = String(s || '').trim()
          if (!ss) continue
          lines.push(`  - ${ss}`)
        }
      }
    }
  }

  lines.push('')
  lines.push('5) What is uncertain (if any)')
  lines.push('')
  const overall = (typeof (conf as any)?.overall === 'number') ? Number((conf as any).overall) : null
  const br = ((conf as any)?.breakdown && typeof (conf as any).breakdown === 'object') ? (conf as any).breakdown : {}
  if (overall !== null && Number.isFinite(overall)) lines.push(`Overall confidence: ${overall.toFixed(2)}`)
  const geomC = (typeof (br as any)?.geometry === 'number') ? Number((br as any).geometry) : null
  const annC = (typeof (br as any)?.annotations === 'number') ? Number((br as any).annotations) : null
  const intC = (typeof (br as any)?.intent === 'number') ? Number((br as any).intent) : null
  const sub: string[] = []
  if (geomC !== null && Number.isFinite(geomC)) sub.push(`geometry=${geomC.toFixed(2)}`)
  if (annC !== null && Number.isFinite(annC)) sub.push(`annotations=${annC.toFixed(2)}`)
  if (intC !== null && Number.isFinite(intC)) sub.push(`intent=${intC.toFixed(2)}`)
  if (sub.length) lines.push(`Breakdown: ${sub.join(', ')}`)

  const missing = Array.isArray((unc as any)?.missing_information) ? (unc as any).missing_information.map(String).filter(Boolean) : []
  const amb = Array.isArray((unc as any)?.ambiguous_features) ? (unc as any).ambiguous_features.map(String).filter(Boolean) : []
  if (missing.length || amb.length) {
    if (missing.length) {
      lines.push('Missing information:')
      for (const m of missing.slice(0, 10)) lines.push(`- ${m}`)
    }
    if (amb.length) {
      lines.push('Ambiguous features:')
      for (const a of amb.slice(0, 10)) lines.push(`- ${a}`)
    }
  } else if (overall !== null && Number.isFinite(overall) && overall >= 0.8) {
    lines.push('No major uncertainties flagged by the system at the current confidence threshold.')
  } else {
    lines.push('Uncertainty is present but not yet itemized.')
  }

  return lines.join('\n')
}

function buildUnifiedUnderstandingReport(body: ChatRequestBody): string {
  const ctx = body?.context || {}
  const u = (ctx?.understanding && typeof ctx.understanding === 'object') ? ctx.understanding : null
  if (!u) return ''

  const msg = String(body?.message || body?.messages?.slice(-1)?.[0]?.content || '').trim()
  const intent = classifyIntent(msg)

  const kind = String((u as any).kind || '').trim()
  const payload = (u as any).payload

  if (kind === 'engineering_drawing') {
    // Only use EUO renderer for engineering summary/engineering questions.
    // For OCR/page questions, let the generic fallback handlers answer from OCR context.
    if (intent === 'summary' || intent === 'engineering' || intent === 'unknown') {
      return buildEngineeringUnderstandingReport({ ...body, context: { ...(ctx as any), engineeringUnderstanding: payload } })
    }
    return ''
  }

  if (kind === 'document') {
    // For OCR/page questions, let the generic fallback handlers answer from OCR context.
    if (intent === 'ocr') {
      return ''
    }
    const p = (payload && typeof payload === 'object') ? payload : {}
    const ocrText = String(p?.ocrFullText || p?.ocrSummary || '').trim()
    const contacts = (p?.contacts && typeof p.contacts === 'object') ? p.contacts : null
    const lines: string[] = []
    lines.push('1) What this is')
    lines.push('')
    lines.push('Document type: document (PDF / scanned)')
    if (p?.judgement?.summary) {
      lines.push('')
      lines.push('Evidence (system judgement):')
      lines.push(String(p.judgement.summary).trim())
    }
    lines.push('')
    lines.push('2) Key contents (OCR)')
    lines.push('')
    if (ocrText) {
      const preview = ocrText.replace(/\s+/g, ' ').slice(0, 500)
      lines.push(preview + (ocrText.length > 500 ? '…' : ''))
    } else {
      lines.push('No OCR text available yet.')
    }
    lines.push('')
    lines.push('3) Extracted entities')
    lines.push('')
    const emails = Array.isArray(contacts?.emails) ? contacts.emails.map(String).filter(Boolean) : []
    const phones = Array.isArray(contacts?.phones) ? contacts.phones.map(String).filter(Boolean) : []
    const urls = Array.isArray(contacts?.urls) ? contacts.urls.map(String).filter(Boolean) : []
    if (emails.length) lines.push(`Email(s): ${emails.slice(0, 5).join(', ')}`)
    if (phones.length) lines.push(`Phone(s): ${phones.slice(0, 5).join(', ')}`)
    if (urls.length) lines.push(`URL(s): ${urls.slice(0, 5).join(', ')}`)
    if (!emails.length && !phones.length && !urls.length) lines.push('No contact-like entities extracted.')
    lines.push('')
    lines.push('4) What matters / next questions')
    lines.push('')
    lines.push('- Ask: which fields do you need (invoice number/date/amount, spec clauses, etc.)?')
    lines.push('- Ask: should I extract tables or only text sections?')
    lines.push('')
    lines.push('5) What is uncertain (if any)')
    lines.push('')
    lines.push('OCR quality and document structure parsing vary by scan quality; tables/forms may need dedicated extraction.')
    return lines.join('\n')
  }

  if (kind === 'image') {
    // For OCR/page questions, let the generic fallback handlers answer from OCR context.
    if (intent === 'ocr') {
      return ''
    }
    const p = (payload && typeof payload === 'object') ? payload : {}
    const scene = p?.scene
    const objs = Array.isArray(p?.objects) ? p.objects : []
    const ocr = String(p?.ocrSummary || '').trim()
    const lines: string[] = []
    lines.push('1) What this is')
    lines.push('')
    lines.push('Document type: image')
    const topScene = String(scene?.top_label || '').trim()
    if (topScene) lines.push(`Scene guess: ${topScene}`)
    lines.push('')
    lines.push('2) What is visible')
    lines.push('')
    if (objs.length) {
      lines.push('Detected objects:')
      for (const o of objs.slice(0, 10)) {
        const name = String(o?.class || o?.class_clip_guess || o?.label || '').trim()
        const sc = (typeof o?.confidence === 'number') ? ` (conf=${Number(o.confidence).toFixed(2)})` : (typeof o?.clip_score === 'number' ? ` (conf=${Number(o.clip_score).toFixed(2)})` : '')
        if (!name) continue
        lines.push(`- ${name}${sc}`)
      }
    } else {
      lines.push('No object detections available.')
    }
    lines.push('')
    lines.push('3) Text on the image (OCR)')
    lines.push('')
    if (ocr) lines.push(ocr.replace(/\s+/g, ' ').slice(0, 400) + (ocr.length > 400 ? '…' : ''))
    else lines.push('No OCR text detected.')
    lines.push('')
    lines.push('4) What matters / next questions')
    lines.push('')
    lines.push('- Tell me what you want: identify objects, read text, or extract key fields.')
    lines.push('')
    lines.push('Image understanding depends on detection confidence and OCR quality; ambiguous images may need user guidance.')

    // Explicitly show Engineering Intelligence status if available
    if (p?.engineering_understanding_object) {
      const euo = p.engineering_understanding_object
      const cls = String(euo?.document_classification?.document_type || 'UNKNOWN')
      const conf = Number(euo?.confidence?.overall || 0)
      lines.push('')
      lines.push('6) Engineering Intelligence (EUO)')
      lines.push('')
      lines.push(`Applied analysis: ${cls} (Confidence: ${Math.round(conf * 100)}%)`)
      if (conf < 0.75) lines.push('Note: Confidence too low to switch to Engineering Drawing mode.')
    }
    return lines.join('\n')
  }

  if (kind === 'cad') {
    const p = (payload && typeof payload === 'object') ? payload : {}
    const fname = String(p?.file?.name || '').trim()
    const ext = String(p?.file?.ext || '').trim()
    const lines: string[] = []
    lines.push('1) What this is')
    lines.push('')
    lines.push('Document type: CAD model')
    if (fname) lines.push(`File: ${fname}${ext ? ` (.${ext})` : ''}`)
    lines.push('')
    lines.push('2) What I can do right now')
    lines.push('')
    lines.push('- Render the model (viewer)')
    lines.push('- Answer UI questions (how to export, how to interpret views)')
    lines.push('')
    lines.push('3) What is missing for manufacturing intelligence')
    lines.push('')
    lines.push('We need a CAD analysis step (mesh/solid statistics + feature hints). STEP preview is currently stubbed, so we only have a visual model, not measured features.')
    lines.push('')
    lines.push('4) Next actions')
    lines.push('')
    lines.push('- If you want: I can add a lightweight mesh analysis (triangles, bounding box) for GLB/OBJ in the backend.')
    lines.push('- For STEP: we need a server-side converter that outputs measured geometry.')
    lines.push('')
    lines.push('5) What is uncertain (if any)')
    lines.push('')
    lines.push('Without CAD feature extraction, manufacturing/process inference is not reliable.')
    return lines.join('\n')
  }

  return ''
}

function _uniqStr(xs: string[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const x of xs) {
    const s = String(x || '').trim()
    if (!s) continue
    const k = s.toLowerCase()
    if (seen.has(k)) continue
    seen.add(k)
    out.push(s)
  }
  return out
}

function _pickOcrLines(ocrText: string, re: RegExp, limit = 12): string[] {
  const src = String(ocrText || '').replace(/\r/g, '\n')
  if (!src.trim()) return []
  const lines = src
    .split('\n')
    .map((s) => s.trim())
    .filter(Boolean)
  const picked: string[] = []
  for (const ln of lines) {
    if (!re.test(ln)) continue
    // Avoid dumping huge OCR blobs.
    if (ln.length > 160) continue
    picked.push(ln)
    if (picked.length >= limit) break
  }
  return _uniqStr(picked)
}

function _looksLikeGarbageField(s: string): boolean {
  const v = String(s || '').trim()
  if (!v) return true
  if (v.length < 2) return true
  if (v.length > 80) return true
  if (/\[object\s+Object\]/i.test(v)) return true
  // Too many digits/punctuation relative to letters usually means OCR noise.
  const letters = (v.match(/[A-Za-z]/g) || []).length
  const digits = (v.match(/[0-9]/g) || []).length
  const punct = (v.match(/[^A-Za-z0-9\s]/g) || []).length
  if (letters <= 0) return true
  if (digits > letters * 2) return true
  if (punct > letters * 2) return true
  // Common OCR garbage from notes.
  if (/\b(shall|conform|standard|speed|flammability)\b/i.test(v) && letters < 12) return true
  // Looks like a sentence/notes rather than a part name.
  if (/\b(shall|must|conform|purpose only|do not start)\b/i.test(v)) return true
  // Too many standalone numbers/steps often comes from note numbering.
  if ((v.match(/\b\d+\b/g) || []).length >= 4 && letters < 12) return true
  // Part names usually contain at least one long-ish token.
  if (!/[A-Za-z]{4,}/.test(v)) return true
  return false
}

function _cleanDimText(x: any): string {
  const s = String(x?.text ?? x?.value ?? x ?? '').trim()
  if (!s) return ''
  if (/\[object\s+Object\]/i.test(s)) return ''
  return s.replace(/\s+/g, ' ').slice(0, 40)
}

function _isPlausibleDimension(s: string): boolean {
  const v = String(s || '').trim()
  if (!v) return false
  if (v.length > 40) return false
  // Typical dimension tokens.
  if (/\bGR\d{3,4}\b/i.test(v)) return true
  if (/\bR\s*\d+(?:\.\d+)?\b/i.test(v)) return true
  if (/[⌀Øø]/.test(v) && /\d/.test(v)) return true
  if (/\d+(?:\.\d+)?\s*(?:mm|cm|in|\")\b/i.test(v)) return true
  if (/\d+(?:\.\d+)?\s*[±\+\-]\s*\d+(?:\.\d+)?/.test(v)) return true
  // Plain numeric dims like 26.1
  if (/^\d+(?:\.\d+)?$/.test(v)) {
    // Drop trivial single-digit integers (usually bullet numbers, not dimensions).
    if (/^\d$/.test(v)) return false
    // Prefer decimals or multi-digit values.
    return v.includes('.') || v.length >= 2
  }
  return false
}

function _extractPartNameFromOcr(ocrText: string): string {
  const src = String(ocrText || '').replace(/\r/g, '\n')
  if (!src.trim()) return ''
  const lines = src.split('\n').map((s) => s.trim()).filter(Boolean)

  // Strong patterns common in title blocks.
  const patterns: RegExp[] = [
    /\b(?:part\s*name|name|品名)\b\s*[:：]?\s*([A-Z0-9][A-Z0-9\s,\-\.]{4,})/i,
    /\b(cushion[^\n]{4,60})\b/i,
  ]

  for (const ln of lines) {
    for (const re of patterns) {
      const m = ln.match(re)
      if (!m) continue
      const cand = String(m[1] || '').trim()
      if (!cand) continue
      const cleaned = cand.replace(/\s+/g, ' ').slice(0, 60)
      if (_looksLikeGarbageField(cleaned)) continue
      return cleaned
    }
  }
  return ''
}

function buildEngineeringDrawingReport(body: ChatRequestBody): string {
  const ctx = body?.context || {}
  const ed = (ctx?.engineeringDrawing && typeof ctx.engineeringDrawing === 'object') ? ctx.engineeringDrawing : null
  const dom = (ctx?.engineeringDom && typeof ctx.engineeringDom === 'object') ? ctx.engineeringDom : null
  const judgement = (ctx?.judgement && typeof ctx.judgement === 'object') ? ctx.judgement : null
  const md = (ed && (ed as any).metadata && typeof (ed as any).metadata === 'object') ? (ed as any).metadata : {}
  const ii = (ed && (ed as any).inferred_intent && typeof (ed as any).inferred_intent === 'object') ? (ed as any).inferred_intent : {}
  const conf = (ed && (ed as any).confidence && typeof (ed as any).confidence === 'object') ? (ed as any).confidence : {}
  const ocrFull = String(ctx?.ocrFullText || '').trim()
  const ocrSummary = String(ctx?.ocrSummary || '').trim()
  const ocrText = ocrFull || ocrSummary

  // Enhanced Engineering Understanding Object (EUO) support
  const euo = (ctx?.engineeringUnderstanding && typeof ctx.engineeringUnderstanding === 'object') ? ctx.engineeringUnderstanding : null
  const euoPart = (euo?.part_identity && typeof euo.part_identity === 'object') ? euo.part_identity : null
  const euoIntent = (euo?.manufacturing_intent && typeof euo.manufacturing_intent === 'object') ? euo.manufacturing_intent : null
  const euoGeom = (euo?.geometry_understanding && typeof euo.geometry_understanding === 'object') ? euo.geometry_understanding : null
  const euoAnn = (euo?.annotation_understanding && typeof euo.annotation_understanding === 'object') ? euo.annotation_understanding : null
  const euoConstraints = (euo?.constraints && typeof euo.constraints === 'object') ? euo.constraints : null

  const mdEvidence = (md && typeof md.evidence === 'object') ? (md as any).evidence : null
  const titleRaw = String(md?.part_name || md?.title || md?.fields?.title || '').trim()
  const titleEvConf = Number(mdEvidence?.part_name?.confidence ?? 0)
  const titleIsTrusted = titleEvConf >= 0.8
  const ocrTitle = (!titleIsTrusted || _looksLikeGarbageField(titleRaw)) ? _extractPartNameFromOcr(ocrText) : ''
  let title = (titleIsTrusted && !_looksLikeGarbageField(titleRaw)) ? titleRaw : ocrTitle

  // Prefer EUO Part Name if available and valid
  const euoName = String(euoPart?.name || '').trim()
  if (euoName && euoName !== 'UNSPECIFIED PART' && euoName !== 'UNKNOWN') {
    title = euoName
  }

  const drawingNo = String(md?.part_number || md?.drawing_no || md?.fields?.drawing_no || '').trim()
  const ref = String(md?.reference_drawing || md?.fields?.reference_drawing || md?.fields?.ref_dwg || '').trim()
  const materialRaw = String(md?.material || md?.fields?.material || '').trim()
  const materialEvConf = Number(mdEvidence?.material?.confidence ?? 0)
  const materialIsTrusted = materialEvConf >= 0.8
  const materialLooksOk = /\b(rubber|spc\d{2,4}|pp|pe|abs|pc|pa\d{1,2}|tpu|epdm|silicone|nbr)\b/i.test(materialRaw)
  let material = (materialIsTrusted || materialLooksOk) && !_looksLikeGarbageField(materialRaw) ? materialRaw : ''

  // Prefer EUO Material if available
  const euoMat = String(euoIntent?.material_class || '').trim()
  if (euoMat && euoMat !== 'UNKNOWN') {
    material = euoMat
  }

  const scale = String(md?.scale || md?.fields?.scale || '').trim()
  const rev = String(md?.revision || md?.fields?.revision || md?.fields?.rev || '').trim()
  const units = String(md?.units || '').trim()

  const views = (ed && Array.isArray((ed as any).views)) ? (ed as any).views : []
  const regions = (ed && Array.isArray((ed as any).regions)) ? (ed as any).regions : []
  const domViews = (dom && Array.isArray((dom as any).views)) ? (dom as any).views : []

  const dimsFromCtx = Array.isArray(ctx?.dimensions) ? ctx.dimensions.map((d: any) => _cleanDimText(d)).filter(Boolean) : []
  const dimCands = (ed && (ed as any).annotations && Array.isArray((ed as any).annotations.dimensions)) ? (ed as any).annotations.dimensions : []
  const dimTexts = _uniqStr(
    [
      ...dimsFromCtx,
      ...dimCands.map((d: any) => _cleanDimText(d)).filter(Boolean),
    ].filter(_isPlausibleDimension)
  ).slice(0, 16)

  const noteLines = _pickOcrLines(
    ocrText,
    /\b(note|notes|tolerance|tolerances|finish|grain|vulcaniz|burr|deburr|flammability|do not start tooling)\b/i,
    10,
  )
  const stdLines = _pickOcrLines(
    ocrText,
    /\b(iso|jis|bsdm|bsdl|bsd\d{3,}|fmvss|sae|astm|din)\b/i,
    10,
  )

  const materialHintLines = _pickOcrLines(
    ocrText,
    /\b(material|rubber|plastic|spc\d{2,4}|epdm|silicone|nbr|tpu)\b/i,
    6,
  )

  const viewTypes = _uniqStr(
    [...views, ...domViews]
      .map((v: any) => String(v?.type || v?.view_type || v?.label || '').trim())
      .filter(Boolean)
  )
  const regionTypes = _uniqStr(regions.map((r: any) => String(r?.type || '').trim()).filter(Boolean))
  const globalConf = (typeof conf?.global === 'number') ? conf.global : null
  const borderConf = (typeof conf?.sheet === 'number') ? conf.sheet : (typeof conf?.border === 'number' ? conf.border : null)

  // Use EUO confidence if available
  const euoConf = (typeof euo?.confidence?.overall === 'number') ? euo.confidence.overall : null

  const lines: string[] = []
  lines.push('1. What is this document?')
  lines.push('')
  if (euo?.document_classification?.document_type) {
    lines.push(`Document Classification: ${euo.document_classification.document_type}`)
  } else {
    lines.push('This appears to be a manufacturing technical drawing (engineering drawing).')
  }

  if (judgement?.summary && !euo) {
    lines.push('')
    lines.push('Evidence (system judgement):')
    lines.push(String(judgement.summary).trim())
  }

  if (euoConf !== null) {
    lines.push('')
    lines.push(`Confidence (EUO): ${Number(euoConf).toFixed(2)}`)
  } else if (borderConf !== null || globalConf !== null) {
    const bits: string[] = []
    if (globalConf !== null) bits.push(`global_conf=${Number(globalConf).toFixed(2)}`)
    if (borderConf !== null) bits.push(`border_conf=${Number(borderConf).toFixed(2)}`)
    if (bits.length) {
      lines.push('')
      lines.push(`Confidence: ${bits.join(', ')}`)
    }
  }

  lines.push('')
  lines.push('2. What is the part?')
  lines.push('')
  if (title) {
    lines.push(`Part name/title: ${title}`)
  } else {
    lines.push("I couldn't confidently extract a part name/title yet.")
  }
  if (drawingNo) lines.push(`Part/Drawing number: ${drawingNo}`)
  if (ref) lines.push(`Reference drawing: ${ref}`)
  if (scale) lines.push(`Scale: ${scale}`)
  if (rev) lines.push(`Revision: ${rev}`)
  if (units) lines.push(`Units: ${units}`)

  if (euoPart?.classification) lines.push(`Classification: ${euoPart.classification}`)

  if (ii?.part_family || ii?.use_case) {
    lines.push('')
    lines.push('Interpretation (rule-based):')
    if (ii.part_family) lines.push(`- Part family: ${String(ii.part_family)}`)
    if (ii.use_case) lines.push(`- Use case: ${String(ii.use_case)}`)
  }

  lines.push('')
  lines.push('3. What material / process?')
  lines.push('')
  if (material) {
    lines.push(`Material: ${material}`)
  } else {
    lines.push('Material: not confidently parsed.')
  }

  if (euoIntent?.likely_process && euoIntent.likely_process !== 'UNKNOWN') {
    lines.push(`Likely Process: ${euoIntent.likely_process}`)
  }

  if (euoIntent?.production_scale && euoIntent.production_scale !== 'UNKNOWN') {
    lines.push(`Production Scale: ${euoIntent.production_scale}`)
  }

  if (ii?.material_class && !euoIntent) {
    lines.push(`Material class (inferred): ${String(ii.material_class)}`)
  }

  lines.push('')
  lines.push('4. Geometry & Structure')
  lines.push('')

  if (euoGeom) {
    if (euoGeom.primary_form && euoGeom.primary_form !== 'UNKNOWN') lines.push(`Primary Form: ${euoGeom.primary_form}`)
    if (euoGeom.dimensional_complexity) lines.push(`Complexity: ${euoGeom.dimensional_complexity}`)
    if (euoGeom.key_features && euoGeom.key_features.length) lines.push(`Key Features: ${euoGeom.key_features.join(', ')}`)
  }

  const viewCount = Array.isArray(views) ? views.length : 0
  const domViewCount = Array.isArray(domViews) ? domViews.length : 0
  if (viewCount > 0 || domViewCount > 0) {
    const counts = [`views=${viewCount}`, `dom_views=${domViewCount}`].join(', ')
    lines.push(`Detected Views: ${counts}`)
  }

  // Legacy view hints only if no EUO geometry info
  if (!euoGeom) {
    const regionHint = regionTypes.length ? `Regions: ${regionTypes.join(', ')}` : ''
    const viewHint = viewTypes.length ? `View types (detected labels): ${viewTypes.join(', ')}` : ''
    if (regionHint || viewHint) lines.push(`Structure Hints: ${regionHint}; ${viewHint}`)
  }


  lines.push('')
  lines.push('5. What dimensions are written?')
  lines.push('')
  if (dimTexts.length) {
    lines.push('Examples detected:')
    for (const d of dimTexts) lines.push(`- ${d}`)
  } else {
    lines.push("I didn't extract reliable dimension text yet. (This usually improves with higher-resolution exports.)")
  }

  lines.push('')
  lines.push('6. What annotations / manufacturing notes / standards are written?')
  lines.push('')
  if (noteLines.length) {
    lines.push('Notes (OCR-matched lines):')
    for (const n of noteLines) lines.push(`- ${n}`)
  } else {
    lines.push('Notes: not reliably isolated from OCR yet.')
  }
  if (stdLines.length) {
    lines.push('')
    lines.push('Standards / spec references (OCR-matched lines):')
    for (const s of stdLines) lines.push(`- ${s}`)
  }

  return lines.join('\n')
}

function classifyIntent(msgRaw: string): GroundedIntent {
  const msg = String(msgRaw || '').trim()
  const lower = msg.toLowerCase()
  if (!msg) return 'unknown'

  // UI quick-actions (buttons) sent as plain text.
  if (lower.includes('list detected dimensions') || lower.includes('dimensions/tolerances')) return 'dimensions'
  if (lower.includes('extract key notes') || lower.includes('notes/annotations') || lower.includes('notes') || lower.includes('annotations')) return 'ocr'
  if (lower.includes('part resembles') || lower.includes('what the part resembles')) return 'summary'

  const asksSummary = /\b(summary|summarize|summarise|details|information|info)\b/i.test(msg)
  const asksWhatIsThis = /\b(what is this|what\s+is\s+it|what\s+do\s+you\s+see|identify\s+this|tell\s+me\s+about\s+this)\b/i.test(lower)
  if (asksSummary || asksWhatIsThis) return 'summary'

  const asksOcr =
    lower.includes("what's written") ||
    lower.includes('whats written') ||
    lower.includes('what written') ||  // typo variant
    lower.includes('what is written') ||
    lower.includes('read the text') ||
    lower.includes('read text') ||
    lower.includes('what does it say') ||
    lower.includes('extract text') ||
    lower.includes('ocr') ||
    /\b(written|text)\s*(on|in)?\s*page\s*\d+/i.test(lower) ||  // "written on page 3", "text on page 5"
    /\bpage\s*\d+\s*(text|written|content)/i.test(lower)  // "page 3 text", "page 5 content"
  if (asksOcr) return 'ocr'

  const asksContact =
    /\b(email|e-mail)\b/i.test(msg) ||
    /\b(phone|mobile|tel|telephone|call)\b/i.test(msg) ||
    /\b(website|web\s*site|url|link|domain)\b/i.test(msg) ||
    /\b(address|location|where)\b/i.test(msg)
  if (asksContact) return 'contacts'

  const asksDims = /\b(dimension|dimensions|size|sizes|tolerance|tolerances|diameter|radius|mm|cm|inch|\b⌀\b|\bø\b)\b/i.test(msg)
  if (asksDims) return 'dimensions'

  const asksEng = /\b(scale|material|finish|revision|rev\b|sheet|drawing\s*(?:no|number)|dwg\b|title\b|drawn\s*by|checked\s*by|approved\s*by|date)\b/i.test(msg)
  if (asksEng) return 'engineering'

  const asksObjects = /\b(objects|object|items|things|what\s+is\s+in\s+this|detect|detections|identify)\b/i.test(lower)
  if (asksObjects) return 'objects'

  return 'unknown'
}

function _attemptNaiveQA(query: string, ocrText: string): string | null {
  if (!query || !ocrText || query.length < 4) return null
  const qObj = query.toLowerCase().replace(/[^a-z0-9 ]/g, '')
  const stopWords = ['what', 'where', 'when', 'show', 'tell', 'find', 'is', 'the', 'a', 'an', 'in', 'on', 'of', 'for', 'to', 'this']
  const keywords = qObj.split(' ').filter(w => w.length > 3 && !stopWords.includes(w))
  if (keywords.length === 0) return null

  const lines = ocrText.replace(/\r/g, '\n').split('\n').filter(l => l.trim().length > 5)
  let bestLine = ''
  let maxScore = 0

  for (const line of lines) {
    const lLow = line.toLowerCase()
    let score = 0
    let matchCount = 0
    for (const k of keywords) {
      if (lLow.includes(k)) {
        score += 1
        matchCount++
      }
    }
    // Boost check: if line contains "Answer:" or "Title:" or similar structure relative to query? 
    // Keep it simple: Term frequency
    if (score > maxScore) {
      maxScore = score
      bestLine = line
    }
  }

  // Threshold: at least one keyword match
  if (maxScore > 0) return bestLine.trim()
  return null
}

async function fallbackReply(body: ChatRequestBody): Promise<string> {
  const msg = String(body?.message || body?.messages?.slice(-1)?.[0]?.content || '').trim()
  if (!msg) return 'Send a message and I will reply here.'

  const lower = msg.toLowerCase()
  const contacts = (body?.context?.contacts && typeof body.context.contacts === 'object') ? body.context.contacts : null
  const judgementKind = String(body?.context?.judgement?.kind || '').trim().toLowerCase()
  const judgementSummary = String(body?.context?.judgement?.summary || '').trim()
  const ocrFull = String(body?.context?.ocrFullText || '').trim()
  const ocrSummary = String(body?.context?.ocrSummary || '').trim()
  const ocrText = ocrFull || ocrSummary
  const ocrPages = Array.isArray(body?.context?.ocrPages) ? body.context.ocrPages : null
  const perceptionV1 = (body?.context?.perception_v1 && typeof body.context.perception_v1 === 'object')
    ? body.context.perception_v1
    : null
  const engineering = (body?.context?.engineering && typeof body.context.engineering === 'object') ? body.context.engineering : null
  const engineeringDom = (body?.context?.engineeringDom && typeof body.context.engineeringDom === 'object') ? body.context.engineeringDom : null
  const engineeringDrawing = (body?.context?.engineeringDrawing && typeof body.context.engineeringDrawing === 'object')
    ? body.context.engineeringDrawing
    : null
  const engineeringUnderstanding = (body?.context?.engineeringUnderstanding && typeof body.context.engineeringUnderstanding === 'object')
    ? body.context.engineeringUnderstanding
    : null
  const understanding = (body?.context?.understanding && typeof body.context.understanding === 'object')
    ? body.context.understanding
    : null

  // Senior Engineering Chat Integration

  // OPTIMIZATION: Visual Session Caching
  // A0.2: Enable Conversation-Led Mode even without MasterDescription
  const history = (Array.isArray(body.messages) && body.messages.length > 0)
    ? body.messages
    : [{ role: 'user', content: msg }]

  const sourceContext = body?.context?.masterDescription || "No file uploaded. User is in Conversation-Led Intent Discovery Mode."

  // Always try LLM first for broader queries, unless it's a specific trivial lookup?
  // Actually, for Dream Mode, we ALWAYS want the LLM.
  // We can assume if we are here, we want the LLM to handle it.
  const judgement = (body?.context?.judgement && typeof body.context.judgement === 'object') ? body.context.judgement : null
  const resp = await generateTextResponse(sourceContext, history, judgement)
  return resp


  if (understanding) {
    const imageUrl = (body?.context?.imageUrl && typeof body.context.imageUrl === 'string') ? body.context.imageUrl : undefined
    const engineered = await generateEngineeringResponse({
      ...understanding,
      // Inject supplementary context for the engineer
      text_content: ocrFull || ocrSummary || undefined,
      scene_data: body?.context?.scene || undefined,
      detected_objects: body?.context?.objects || undefined
    }, msg, imageUrl, ocrText)
    if (engineered) return engineered
  }

  // Auto-QA: Try to answer question from full text content
  let fullSearchText = ocrText
  if (ocrPages && Array.isArray(ocrPages)) {
    fullSearchText = ocrPages.map((p: any) => String(p.text || '')).join('\n')
  }

  // Relaxed QA trigger: allow questions without ? if they start with W-words
  const isQuestion = msg.includes('?') || /^(who|what|where|when|how|why|which)\b/i.test(msg)
  if (isQuestion && msg.length > 6) {
    // Exclude specific intents that are handled below
    const isSpecific = /\b(scale|material|finish|rev|sheet|drawing|title|date|drawn|view|dim|tol|object|ocr|written)\b/i.test(msg)
    if (!isSpecific) {
      const qa = _attemptNaiveQA(msg, fullSearchText)
      if (qa) return `Best answer found in document:\n> "${qa}"`
    }
  }

  const asksScale = /\bscale\b/i.test(msg)
  const asksMaterial = /\bmaterial\b/i.test(msg)
  const asksFinish = /\bfinish\b|surface\s*finish/i.test(msg)
  const asksRevision = /\brev\b|revision/i.test(msg)
  const asksSheet = /\bsheet\b/i.test(msg)
  const asksDrawingNo = /\b(drawing\s*(?:no|number)|dwg\s*(?:no|number)|dwg\b)\b/i.test(msg)
  const asksTitle = /\btitle\b/i.test(msg)
  const asksDate = /\bdate\b/i.test(msg)
  const asksAuthor = /\b(drawn\s*by|checked\s*by|approved\s*by)\b/i.test(msg)
  const asksViews = /\b(view|views|viewport|orthographic|front\s*view|top\s*view|side\s*view|section\s*view|detail\s*view)\b/i.test(msg)

  const engPage0 = (engineering && Array.isArray((engineering as any).pages) && (engineering as any).pages.length)
    ? (engineering as any).pages[0]
    : null
  const engFields = (engPage0 && engPage0.title_block && typeof engPage0.title_block === 'object' && engPage0.title_block.fields && typeof engPage0.title_block.fields === 'object')
    ? engPage0.title_block.fields
    : null
  const engScaleHint = (engPage0 && engPage0.border_notes && typeof engPage0.border_notes === 'object')
    ? String(engPage0.border_notes.scale_hint || '').trim()
    : ''

  const domViews = (engineeringDom && Array.isArray((engineeringDom as any).views)) ? (engineeringDom as any).views : null
  const domDimCands = (engineeringDom && (engineeringDom as any).annotations && typeof (engineeringDom as any).annotations === 'object' && Array.isArray((engineeringDom as any).annotations.dimensions))
    ? (engineeringDom as any).annotations.dimensions
    : null

  const dims = Array.isArray(body?.context?.dimensions) ? body.context.dimensions : null
  const objs = Array.isArray(body?.context?.objects) ? body.context.objects : null

  const emails = Array.isArray(contacts?.emails) ? contacts.emails.map(String).filter(Boolean) : []
  const phones = Array.isArray(contacts?.phones) ? contacts.phones.map(String).filter(Boolean) : []
  const urls = Array.isArray(contacts?.urls) ? contacts.urls.map(String).filter(Boolean) : []
  const addressLike = Array.isArray(contacts?.address_like) ? contacts.address_like.map(String).filter(Boolean) : []

  const asksEmail = /\b(email|e-mail)\b/i.test(msg)
  const asksPhone = /\b(phone|mobile|tel|telephone|call)\b/i.test(msg)
  const asksWebsite = /\b(website|web\s*site|url|link|domain)\b/i.test(msg)
  const asksAddress = /\b(address|location|where)\b/i.test(msg)
  const asksSummary = /\b(summary|summarize|summarise|details|information|info)\b/i.test(msg)
  const asksWhatIsThis = /\b(what is this|what\s+is\s+it|what\s+do\s+you\s+see|identify\s+this|tell\s+me\s+about\s+this)\b/i.test(lower)
  const asksResembles = /\b(part\s+resembles|what\s+the\s+part\s+resembles|resembles)\b/i.test(lower)
  const asksNotes = /\b(notes|annotations|key\s+notes|extract\s+key\s+notes)\b/i.test(lower)

  const intent = classifyIntent(msg)

  // Unified understanding is the sole truth layer when present.
  // Unified understanding is the sole truth layer when present.
  if (understanding && (asksSummary || asksWhatIsThis || intent === 'summary')) {
    const out = buildUnifiedUnderstandingReport(body)
    if (out) return out
  }

  // Backward-compat: EUO is the truth layer for engineering drawings when present.
  if (judgementKind === 'engineering_drawing' && engineeringUnderstanding) {
    // Do not let EUO block OCR/page questions; only use it for summary/engineering queries.
    if (intent === 'summary' || intent === 'engineering' || intent === 'unknown') {
      return buildEngineeringUnderstandingReport(body)
    }
  }

  // Default "intelligent" report for engineering drawings.
  if (
    judgementKind === 'engineering_drawing' &&
    (intent === 'summary' || intent === 'engineering' || intent === 'unknown' || asksResembles)
  ) {
    return buildEngineeringDrawingReport(body)
  }

  if ((asksSummary || asksWhatIsThis) && judgementSummary) {
    return judgementSummary
  }

  if ((asksSummary || asksWhatIsThis) && judgementKind === 'engineering_drawing') {
    const md = engineeringDrawing && (engineeringDrawing as any).metadata && typeof (engineeringDrawing as any).metadata === 'object'
      ? (engineeringDrawing as any).metadata
      : null
    const title = String(md?.title || md?.part_name || md?.fields?.title || '').trim()
    const material = String(md?.material || md?.fields?.material || '').trim()
    const scale = String(md?.scale || md?.fields?.scale || '').trim()
    const rev = String(md?.revision || md?.fields?.revision || md?.fields?.rev || '').trim()
    const drawingNo = String(md?.drawing_no || md?.fields?.drawing_no || '').trim()

    const lines: string[] = []
    if (title) lines.push(`Part/title: ${title}`)
    if (drawingNo) lines.push(`Drawing No.: ${drawingNo}`)
    if (material) lines.push(`Material: ${material}`)
    if (scale) lines.push(`Scale: ${scale}`)
    if (rev) lines.push(`Revision: ${rev}`)

    if (lines.length) {
      return `This is an engineering drawing.\n\n${lines.join('\n')}`
    }
  }

  if ((asksResembles || asksWhatIsThis) && judgementKind === 'engineering_drawing') {
    const md = engineeringDrawing && (engineeringDrawing as any).metadata && typeof (engineeringDrawing as any).metadata === 'object'
      ? (engineeringDrawing as any).metadata
      : null
    const title = String(md?.title || md?.part_name || md?.fields?.title || '').trim()
    if (title) {
      return `Best grounded guess (from title block): this part is "${title}".`
    }
    return "I can't confidently infer what the part resembles yet (beyond the title block). Ask: \"what is the title?\" or \"read the text\", or share which view/feature you mean.";
  }

  if (asksNotes && judgementKind === 'engineering_drawing') {
    // Pull likely NOTE lines from OCR text (common patterns: NOTE, NOTES, GENERAL).
    const src = String(ocrFull || ocrSummary || '').replace(/\r/g, '\n')
    const lines = src
      .split('\n')
      .map((s) => s.trim())
      .filter(Boolean)
    const picked = lines
      .filter((s) => /\b(note|notes|general|tolerance)\b/i.test(s))
      .slice(0, 20)
    if (picked.length) {
      return `Notes/annotations I can read (OCR-based):\n${picked.map((s) => `- ${s}`).join('\n')}`
    }
    return "I couldn't reliably isolate note/annotation lines from OCR. Try asking: \"what's written\" or \"what's written on page 1\" and I'll show the extracted text.";
  }

  if (asksEmail && (emails.length || ocrText)) {
    if (!emails.length) return "I couldn't confidently find an email address in the text I extracted."
    return `Email(s) found:\n${emails.map((e: string) => `- ${e}`).join('\n')}`
  }
  if (asksPhone && (phones.length || ocrText)) {
    if (!phones.length) return "I couldn't confidently find a phone number in the text I extracted."
    return `Phone number(s) found:\n${phones.map((p: string) => `- ${p}`).join('\n')}`
  }
  if (asksWebsite && (urls.length || ocrText)) {
    if (!urls.length) return "I couldn't confidently find a website/URL in the text I extracted."
    return `Website(s) found:\n${urls.map((u: string) => `- ${u}`).join('\n')}`
  }
  if (asksAddress && (addressLike.length || ocrText)) {
    if (!addressLike.length) return "I couldn't confidently find an address/location line in the text I extracted."
    return `Address (best guess):\n- ${addressLike[0]}`
  }

  if (asksSummary && (judgementKind === 'business_card' || (emails.length || phones.length || urls.length))) {
    const lines: string[] = ['This looks like a visiting card. Here is what I found:']
    if (emails.length) lines.push(`\nEmail:\n${emails.map((e: string) => `- ${e}`).join('\n')}`)
    if (phones.length) lines.push(`\nPhone:\n${phones.map((p: string) => `- ${p}`).join('\n')}`)
    if (urls.length) lines.push(`\nWebsite:\n${urls.map((u: string) => `- ${u}`).join('\n')}`)
    if (addressLike.length) lines.push(`\nAddress (best guess):\n- ${addressLike[0]}`)
    return lines.join('\n')
  }

  const asksDims = /\b(dimension|dimensions|size|sizes|tolerance|tolerances|diameter|radius|mm|cm|inch|\b⌀\b|\bø\b)\b/i.test(msg)
  if (asksDims && dims && dims.length) {
    const lines = dims.slice(0, 25).map((d: any) => {
      const raw = String(d?.raw || '').trim()
      if (raw) return `- ${raw}`
      const v = (typeof d?.value === 'number') ? d.value : null
      const u = (typeof d?.unit === 'string' && d.unit) ? d.unit : ''
      const k = (typeof d?.kind === 'string' && d.kind) ? d.kind : 'dimension'
      if (v === null) return `- ${k}`
      return `- ${k}: ${v}${u ? ' ' + u : ''}`
    })
    return `Dimensions/tolerances I detected:\n${lines.join('\n')}`
  }

  if (asksDims && judgementKind === 'engineering_drawing' && domDimCands && domDimCands.length) {
    const lines = domDimCands.slice(0, 40).map((d: any) => {
      const t = String(d?.text || '').trim()
      return t ? `- ${t}` : '- (dimension text)'
    })
    return `Dimension text candidates (from the drawing viewport):\n${lines.join('\n')}`
  }

  if (asksViews && judgementKind === 'engineering_drawing' && domViews && domViews.length) {
    const lines = domViews.slice(0, 12).map((v: any, i: number) => {
      const bb = Array.isArray(v?.bounding_box) ? v.bounding_box : null
      const segs = (v?.stats && typeof v.stats === 'object') ? v.stats.segment_count : null
      const segTxt = (typeof segs === 'number') ? `, segments=${segs}` : ''
      const bbTxt = bb ? `bbox=[${bb.map((n: any) => Number(n).toFixed(0)).join(', ')}]` : 'bbox=?'
      return `- View ${String(v?.id || i + 1)}: ${bbTxt}${segTxt}`
    })
    return `Detected view regions (line clusters) in the drawing:\n${lines.join('\n')}`
  }

  const asksObjects = /\b(objects|object|items|things|what\s+is\s+in\s+this|detect|detections|identify)\b/i.test(lower)
  if (asksObjects) {
    if (objs && objs.length) {
      const lines = objs.slice(0, 12).map((o: any) => {
        const nm = String(o?.class || o?.class_clip_guess || '').trim() || 'object'
        const conf = (typeof o?.confidence === 'number') ? o.confidence : (typeof o?.clip_score === 'number' ? o.clip_score : null)
        const pct = (typeof conf === 'number') ? ` (${Math.round(conf * 100)}%)` : ''
        return `- ${nm}${pct}`
      })
      return `Objects/visual hints:\n${lines.join('\n')}`
    }
    // Fallback to scene
    const scene = body?.context?.scene || body?.context?.understanding?.payload?.scene
    if (scene && (scene.top_label || scene.label)) {
      const lbl = String(scene.top_label || scene.label)
      const score = scene.top_score || scene.score || 0
      return `I didn't detect specific objects, but the scene looks like: ${lbl} (${Math.round(score * 100)}%)`
    }
    return "I couldn't identify specific objects in this image."
  }

  const asksOcr =
    lower.includes("what's written") ||
    lower.includes('whats written') ||
    lower.includes('what written') ||  // typo variant
    lower.includes('what is written') ||
    lower.includes('read the text') ||
    lower.includes('read text') ||
    lower.includes('what does it say') ||
    lower.includes('extract text') ||
    lower.includes('ocr') ||
    /\b(written|text)\s*(on|in)?\s*page\s*\d+/i.test(lower) ||
    /\bpage\s*\d+\s*(text|written|content)/i.test(lower)

  if (asksOcr) {
    const pageNum = parseInt(msg.match(/\bpage\s*(\d+)\b/i)?.[1] || '1', 10)
    function textFromPerceptionPage(pn: number): string {
      try {
        const pageText = (perceptionV1 as any)?.page_text
        if (pageText && typeof pageText === 'object') {
          const direct = String((pageText as any)[String(pn)] || '').trim()
          if (direct) return direct
        }

        const blocks = Array.isArray((perceptionV1 as any)?.text_blocks) ? (perceptionV1 as any).text_blocks : []
        const keep = blocks.filter((b: any) => Number(b?.page) === pn && typeof b?.text === 'string' && String(b.text).trim())
        if (!keep.length) return ''
        const sorted = keep
          .map((b: any) => {
            const bn = Array.isArray(b?.bbox_norm) && b.bbox_norm.length === 4 ? b.bbox_norm : null
            const bp = Array.isArray(b?.bbox_px) && b.bbox_px.length === 4 ? b.bbox_px : null
            const y = bn ? Number(bn[1] || 0) : (bp ? Number(bp[1] || 0) : 0)
            const x = bn ? Number(bn[0] || 0) : (bp ? Number(bp[0] || 0) : 0)
            return { x, y, t: String(b.text).trim() }
          })
          .sort((a: any, b: any) => (a.y - b.y) || (a.x - b.x))
        const parts = sorted.map((s: any) => s.t)
        const merged = parts.join(' ').replace(/\s+/g, ' ').trim()
        return merged
      } catch {
        return ''
      }
    }
    if (pageNum && perceptionV1) {
      const txt = textFromPerceptionPage(pageNum)
      if (txt) {
        const clipped = txt.length > 1800 ? txt.slice(0, 1800) + '…' : txt
        return `Here's what I can read from page ${pageNum}:\n\n${clipped}`
      }

      try {
        const pageText = (perceptionV1 as any)?.page_text
        const keys = pageText && typeof pageText === 'object' ? Object.keys(pageText) : []
        if (keys.length) {
          const maxPage = Math.max(...keys.map((k: any) => Number(k || 0)).filter((n: any) => Number.isFinite(n) && n > 0))
          if (Number.isFinite(maxPage) && maxPage > 0) {
            return `I don't have readable text for page ${pageNum} from perception yet. Available pages: 1–${maxPage}.\n\nIf this PDF has more pages, we may currently only process the first ~10 pages.`
          }
        }
      } catch {
        // ignore
      }
    }

    if (pageNum && ocrPages && ocrPages.length) {
      const found = ocrPages.find((p: any) => Number(p?.page) === pageNum)
      const txt = String(found?.text || '').trim()
      if (!txt) {
        const maxPage = Math.max(...ocrPages.map((p: any) => Number(p?.page || 0)))
        return `I don't have readable OCR text for page ${pageNum}. Available pages: 1–${maxPage}.`
      }
      const clipped = txt.length > 1600 ? txt.slice(0, 1600) + '…' : txt
      return `Here's what I can read from page ${pageNum}:\n\n${clipped}`
    }

    // If not a page-specific request, fall back to engineering structured extraction for drawings.
    if (judgementKind === 'engineering_drawing' && engPage0) {
      const lines: string[] = ["Here's what I can extract from the engineering drawing (structured):"]
      if (engFields && typeof engFields === 'object') {
        const orderedKeys = ['title', 'drawing_no', 'scale', 'sheet', 'revision', 'material', 'finish', 'date', 'drawn_by', 'checked_by', 'approved_by']
        const fieldLines = orderedKeys
          .map((k) => ({ k, v: String((engFields as any)[k] || '').trim() }))
          .filter((kv) => kv.v)
          .map((kv) => `- ${kv.k.replace(/_/g, ' ').toUpperCase()}: ${kv.v}`)
        if (fieldLines.length) lines.push(`\nTitle block:\n${fieldLines.join('\n')}`)
      }
      if (engScaleHint && !(engFields && String((engFields as any).scale || '').trim())) {
        lines.push(`\nBorder notes:\n- SCALE: ${engScaleHint}`)
      }
      const mainTxt = (engPage0.main_area && typeof engPage0.main_area === 'object') ? String(engPage0.main_area.text || '').trim() : ''
      if (mainTxt) {
        const clippedMain = mainTxt.length > 800 ? mainTxt.slice(0, 800) + '…' : mainTxt
        lines.push(`\nMain drawing area (text near geometry/dimensions):\n${clippedMain}`)
      }
      return lines.join('\n')
    }

    if (!ocrText && !(ocrPages && ocrPages.length)) {
      return "I couldn't detect readable text in the uploaded file. Try a higher-resolution image, better lighting, or a tighter crop around the text."
    }

    const base = ocrFull || ocrSummary || ''
    const short = String(base).replace(/\s+/g, ' ').trim()
    const clipped = short.length > 1200 ? short.slice(0, 1200) + '…' : short
    const hint = ocrPages && ocrPages.length ? `\n\nTip: you can ask \"what's written on page 2\".` : ''
    return `Here's what I can read from the upload:\n\n${clipped}${hint}`
  }

  if ((asksScale || asksMaterial || asksFinish || asksRevision || asksSheet || asksDrawingNo || asksTitle || asksDate || asksAuthor) && (judgementKind === 'engineering_drawing' || engineering)) {
    const get = (k: string) => (engFields && typeof engFields === 'object') ? String((engFields as any)[k] || '').trim() : ''
    if (asksScale) {
      const v = (engineeringDrawing && String((engineeringDrawing as any)?.metadata?.scale || (engineeringDrawing as any)?.metadata?.fields?.scale || '').trim()) || get('scale') || engScaleHint
      return v ? `Scale: ${v}` : "I couldn't confidently find the scale in the title block/border notes."
    }
    if (asksMaterial) {
      const v = (engineeringDrawing && String((engineeringDrawing as any)?.metadata?.material || (engineeringDrawing as any)?.metadata?.fields?.material || '').trim()) || get('material')
      return v ? `Material: ${v}` : "I couldn't confidently find the material in the title block."
    }
    if (asksFinish) {
      const v = get('finish')
      return v ? `Finish: ${v}` : "I couldn't confidently find the finish/surface finish in the title block."
    }
    if (asksRevision) {
      const v = (engineeringDrawing && String((engineeringDrawing as any)?.metadata?.revision || (engineeringDrawing as any)?.metadata?.fields?.revision || (engineeringDrawing as any)?.metadata?.fields?.rev || '').trim()) || get('revision')
      return v ? `Revision: ${v}` : "I couldn't confidently find the revision in the title block."
    }
    if (asksSheet) {
      const v = get('sheet')
      return v ? `Sheet: ${v}` : "I couldn't confidently find the sheet number in the title block."
    }
    if (asksDrawingNo) {
      const v = (engineeringDrawing && String((engineeringDrawing as any)?.metadata?.drawing_no || (engineeringDrawing as any)?.metadata?.fields?.drawing_no || '').trim()) || get('drawing_no')
      return v ? `Drawing No.: ${v}` : "I couldn't confidently find the drawing/document number in the title block."
    }
    if (asksTitle) {
      const v = (engineeringDrawing && String((engineeringDrawing as any)?.metadata?.title || (engineeringDrawing as any)?.metadata?.part_name || (engineeringDrawing as any)?.metadata?.fields?.title || '').trim()) || get('title')
      return v ? `Title: ${v}` : "I couldn't confidently find the title in the title block."
    }
    if (asksDate) {
      const v = get('date')
      return v ? `Date: ${v}` : "I couldn't confidently find the date in the title block."
    }
    if (asksAuthor) {
      const drawn = get('drawn_by')
      const checked = get('checked_by')
      const approved = get('approved_by')
      const lines: string[] = []
      if (drawn) lines.push(`- Drawn by: ${drawn}`)
      if (checked) lines.push(`- Checked by: ${checked}`)
      if (approved) lines.push(`- Approved by: ${approved}`)
      if (lines.length) return `People:\n${lines.join('\n')}`
      return "I couldn't confidently find drawn/checked/approved names in the title block."
    }
  }

  if (lower.includes('help')) {
    return "Tell me what you uploaded (image/STEP) and what you want to do next, and I'll guide you."
  }
  if (lower.includes('what') && lower.includes('see')) {
    const detected = body?.context?.detectedLabel
    if (detected) return `I can see you uploaded something like: ${detected}. What would you like to improve or analyze?`
    return "I can help analyze what you uploaded. What are you trying to achieve?"
  }

  // Final fallback for unknown intent in local mode
  return `I'm sorry, I couldn't find a specific answer to your question in the document. (Running in Rule-Based Mode)\n\nTry asking for: 'summary', 'text', 'identify objects', or specific engineering fields.`
}

r.post('/chat', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const body = (req.body || {}) as ChatRequestBody

    // Option 1: deterministic router for grounded intents.
    // For anything we can answer from the analysis context, bypass the worker proxy.
    try {
      const msg = String(body?.message || body?.messages?.slice(-1)?.[0]?.content || '').trim()
      const intent = classifyIntent(msg)

      const judgementKind = String(body?.context?.judgement?.kind || '').trim().toLowerCase()

      const hasJudgementSummary = !!String(body?.context?.judgement?.summary || '').trim()
      const hasOcrCtx =
        !!String(body?.context?.ocrFullText || '').trim() ||
        !!String(body?.context?.ocrSummary || '').trim() ||
        (Array.isArray(body?.context?.ocrPages) && body.context.ocrPages.length > 0) ||
        !!(body?.context?.perception_v1 && typeof body.context.perception_v1 === 'object' &&
          (body.context.perception_v1 as any)?.page_text && typeof (body.context.perception_v1 as any).page_text === 'object' &&
          Object.keys((body.context.perception_v1 as any).page_text || {}).length)
      const hasContacts = !!(body?.context?.contacts && typeof body.context.contacts === 'object')
      const hasDims = Array.isArray(body?.context?.dimensions) && body.context.dimensions.length > 0
      const hasObjects = Array.isArray(body?.context?.objects) && body.context.objects.length > 0
      const hasEngineering = !!(body?.context?.engineering && typeof body.context.engineering === 'object')
      const hasEngineeringDrawing = !!(body?.context?.engineeringDrawing && typeof body.context.engineeringDrawing === 'object')
      const hasEngineeringDom = !!(body?.context?.engineeringDom && typeof body.context.engineeringDom === 'object')

      const grounded =
        (intent === 'summary' && (hasJudgementSummary || hasEngineeringDrawing || hasEngineeringDom)) ||
        (intent === 'ocr' && hasOcrCtx) ||
        (intent === 'contacts' && (hasContacts || hasOcrCtx)) ||
        (intent === 'dimensions' && (hasDims || hasOcrCtx)) ||
        (intent === 'objects' && (hasObjects || hasJudgementSummary)) ||
        (intent === 'engineering' && (hasEngineering || hasJudgementSummary || hasOcrCtx))

      // For engineering drawings, prefer deterministic/local answers over proxy.
      const preferLocal = judgementKind === 'engineering_drawing' && (hasEngineeringDrawing || hasEngineeringDom || hasOcrCtx || hasJudgementSummary)

      if (grounded || preferLocal) {
        return res.json({
          id: `local_grounded_${Date.now()}`,
          reply: await fallbackReply(body),
          model: 'local-grounded',
        })
      }
    } catch {
      // If routing pre-check fails for any reason, continue to proxy/fallback.
    }

    // If worker is enabled, forward request. If worker is down, fall back.
    try {
      const out = await maybeProxyPost('/chat', body)
      if (out) return res.json(out)
    } catch {
      // fall through to local fallback
    }

    return res.json({
      id: `local_${Date.now()}`,
      reply: await fallbackReply(body),
      model: 'local-fallback',
    })
  } catch (e) {
    next(e)
  }
})

r.post('/chat/describe', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const body = (req.body || {}) as ChatRequestBody
    const euo = body?.context?.understanding?.payload || body?.context?.engineeringUnderstanding || null
    const ocrText = body?.context?.ocrFullText || body?.context?.ocrSummary || undefined

    // Support single or multiple images
    const singleUrl = (body?.context?.imageUrl && typeof body.context.imageUrl === 'string') ? body.context.imageUrl : undefined
    const multiUrls = (body?.context?.imageUrls && Array.isArray(body.context.imageUrls)) ? body.context.imageUrls : undefined
    const finalImages = multiUrls || (singleUrl ? [singleUrl] : undefined)

    if (!finalImages || finalImages.length === 0) {
      return res.json({ id: `desc_err_${Date.now()}`, description: null, error: "No image URL provided." })
    }

    console.log("[DEEP READ] Generating description for:", finalImages.length, "images")
    const description = await generateMasterDescription(euo, finalImages, ocrText)
    console.log("[DEEP READ] Result:", description.slice(0, 100) + '...')
    return res.json({
      id: `desc_${Date.now()}`,
      description,
      model: 'gpt-4o-vision'
    })
  } catch (e) {
    next(e)
  }
})

export default r
