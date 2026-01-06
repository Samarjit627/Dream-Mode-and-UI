import * as mobilenet from '@tensorflow-models/mobilenet'
import '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd'
import Tesseract from 'tesseract.js'

let modelPromise: Promise<mobilenet.MobileNet> | null = null
let cocoPromise: Promise<any> | null = null

export async function classifyImage(url: string): Promise<{label: string; prob: number} | null> {
  try {
    if (!modelPromise) modelPromise = mobilenet.load()
    const model = await modelPromise
    const img = await loadImage(url)
    const preds: Array<{ className: string; probability: number }> = await model.classify(img as any)
    if (!preds || preds.length === 0) return null
    const top = preds[0]
    return { label: top.className, prob: top.probability }
  } catch {
    return null
  }
}

export type Detection = { label: string; score: number; bbox: [number, number, number, number] } // normalized x,y,w,h

export async function detectObjects(url: string): Promise<Detection[]> {
  try {
    if (!cocoPromise) cocoPromise = cocoSsd.load({ base: 'lite_mobilenet_v2' })
    const model = await cocoPromise
    const img = await loadImage(url)
    const dets: any[] = await model.detect(img as any)
    const w = img.naturalWidth || img.width
    const h = img.naturalHeight || img.height
    return dets.map((d: any) => {
      const [x, y, bw, bh] = d.bbox as [number, number, number, number]
      return { label: (d as any).class || (d as any).className || 'object', score: d.score || 0, bbox: [x / w, y / h, bw / w, bh / h] }
    })
  } catch {
    return []
  }
}

export type OcrWord = { text: string; bbox: [number, number, number, number] } // normalized x,y,w,h

export async function ocrImage(url: string): Promise<{ text: string; words: OcrWord[] } | null> {
  try {
    const res = await Tesseract.recognize(url, 'eng', { logger: () => {} })
    const w = res?.data?.image?.width || 1
    const h = res?.data?.image?.height || 1
    const words: OcrWord[] = (res?.data?.words || []).map((wrd: any) => {
      const bb = wrd?.bbox || wrd?.bbox0 || wrd?.bbox1 || wrd?.bbox2 || wrd?.bbox3 || wrd?.bbox
      const x = (bb?.x0 ?? wrd?.x0 ?? 0) / w
      const y = (bb?.y0 ?? wrd?.y0 ?? 0) / h
      const bw = ((bb?.x1 ?? wrd?.x1 ?? 0) - (bb?.x0 ?? wrd?.x0 ?? 0)) / w
      const bh = ((bb?.y1 ?? wrd?.y1 ?? 0) - (bb?.y0 ?? wrd?.y0 ?? 0)) / h
      return { text: String(wrd?.text || wrd?.symbol || ''), bbox: [x, y, bw, bh] }
    })
    const text = res?.data?.text || ''
    return { text, words }
  } catch {
    return null
  }
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const i = new Image()
    i.crossOrigin = 'anonymous'
    i.onload = () => resolve(i)
    i.onerror = reject
    i.src = src
  })
}
