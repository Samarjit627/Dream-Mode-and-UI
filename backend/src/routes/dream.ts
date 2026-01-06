import { Router } from 'express'
import { createProxyMiddleware, responseInterceptor } from 'http-proxy-middleware'

const DREAM_TARGET = process.env.DREAM_POC_BASE || 'http://localhost:8001'

const r = Router()

function safeJsonParse(s: string): any {
  try {
    return JSON.parse(s)
  } catch {
    return null
  }
}

function buildLogSummary(payload: any): any {
  const j = payload && typeof payload === 'object' ? payload : {}
  const judgement = (j.judgement && typeof j.judgement === 'object') ? j.judgement : {}
  const ed = (j.engineering_drawing && typeof j.engineering_drawing === 'object') ? j.engineering_drawing : null
  const md = (ed && (ed as any).metadata && typeof (ed as any).metadata === 'object') ? (ed as any).metadata : null
  const ii = (ed && (ed as any).inferred_intent && typeof (ed as any).inferred_intent === 'object') ? (ed as any).inferred_intent : null
  const conf = (ed && (ed as any).confidence && typeof (ed as any).confidence === 'object') ? (ed as any).confidence : null

  const pv1 = (j.perception_v1 && typeof j.perception_v1 === 'object') ? j.perception_v1 : null
  const pv1Validation = (pv1 && (pv1 as any).validation && typeof (pv1 as any).validation === 'object') ? (pv1 as any).validation : null
  const pv1Hist = (pv1Validation && (pv1Validation as any).ocr_confidence_histogram && typeof (pv1Validation as any).ocr_confidence_histogram === 'object')
    ? (pv1Validation as any).ocr_confidence_histogram
    : null

  const ocrSummary = String(j.ocrSummary || j.text?.summary || '').trim()
  const ocrFull = String(j.ocrFullText || j.text?.full || '').trim()

  return {
    request_id: j.request_id || j.requestId || j.id || null,
    status: j.status || null,
    judgement: {
      kind: judgement.kind || null,
      summary: judgement.summary ? String(judgement.summary).slice(0, 600) : null,
    },
    engineering_drawing: ed ? {
      metadata: md || null,
      inferred_intent: ii || null,
      confidence: conf || null,
    } : null,
    perception_v1: pv1 ? {
      text_blocks: Array.isArray((pv1 as any).text_blocks) ? (pv1 as any).text_blocks.length : 0,
      lines: Array.isArray((pv1 as any).lines) ? (pv1 as any).lines.length : 0,
      closed_shapes: Array.isArray((pv1 as any).closed_shapes) ? (pv1 as any).closed_shapes.length : 0,
      arrow_candidates: Array.isArray((pv1 as any).arrow_candidates) ? (pv1 as any).arrow_candidates.length : 0,
      ocr_conf_hist: pv1Hist || null,
      closed_loop_line_ratio: (pv1Validation && typeof (pv1Validation as any).closed_loop_line_ratio === 'number')
        ? (pv1Validation as any).closed_loop_line_ratio
        : null,
    } : null,
    ocr: {
      summary_preview: ocrSummary ? ocrSummary.slice(0, 400) : null,
      full_len: ocrFull ? ocrFull.length : 0,
    }
  }
}

// Proxy /api/v1/dream/* -> FastAPI dream-poc service
r.use(
  '/v1/dream',
  createProxyMiddleware({
    target: DREAM_TARGET,
    changeOrigin: true,
    selfHandleResponse: true,
    on: {
      proxyRes: responseInterceptor(async (responseBuffer: any, proxyRes: any, req: any, res: any) => {
        try {
          const url = String(req?.originalUrl || req?.url || '')
          const ct = String(proxyRes?.headers?.['content-type'] || '')
          const wantsLog = /\/v1\/dream\/(analyze|result)\b/i.test(url)
          const isJson = ct.includes('application/json')
          if (wantsLog && isJson) {
            const raw = responseBuffer ? responseBuffer.toString('utf8') : ''
            const j = safeJsonParse(raw)
            const summary = buildLogSummary(j)
            // eslint-disable-next-line no-console
            console.log('[dream-proxy]', url)
            // eslint-disable-next-line no-console
            console.log(JSON.stringify(summary, null, 2))
          }
        } catch {
          // ignore logging errors
        }
        return responseBuffer
      }),
    },
    pathRewrite: (path: string) => {
      // Depending on Express + http-proxy-middleware versions, `path` may be:
      // - '/v1/dream/analyze' (mount path included)
      // - '/analyze' (mount path stripped)
      // Normalize to Dream PoC's FastAPI routes under '/api/v1/dream/*'.
      // Dream PoC health lives at '/health' (not under '/api/v1/dream').
      if (path === '/health' || path === '/v1/dream/health' || path === '/api/v1/dream/health') return '/health'
      if (path.startsWith('/api/v1/dream')) return path
      if (path.startsWith('/v1/dream')) return path.replace(/^\/v1\/dream/, '/api/v1/dream')
      return `/api/v1/dream${path}`
    },
  })
)

export default r
