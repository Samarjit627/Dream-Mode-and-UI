import JSZip from 'jszip'
import { PDFDocument, StandardFonts, rgb } from 'pdf-lib'

export async function exportDreamZip(session: {
  intent: any
  chosenPack: any
  concepts: any[]
  overlays: any[]
  trials: any[]
}) {
  const zip = new JSZip()

  // session.json
  zip.file('session.json', JSON.stringify(session, null, 2))

  // previews placeholders
  const previews = zip.folder('previews')!
  previews.file('preview-1.png', new Uint8Array())
  previews.file('preview-2.png', new Uint8Array())
  previews.file('preview-3.png', new Uint8Array())

  // simple PDF
  const pdf = await PDFDocument.create()
  const page = pdf.addPage([595.28, 841.89]) // A4
  const font = await pdf.embedFont(StandardFonts.Helvetica)
  const { width, height } = page.getSize()
  const title = 'Axis5 Dream Report'
  page.drawText(title, { x: 50, y: height - 80, size: 24, font, color: rgb(0.1,0.1,0.1) })
  page.drawText('Summary (mock):', { x: 50, y: height - 110, size: 12, font })
  page.drawText(`Intent: ${JSON.stringify(session.intent)}`, { x: 50, y: height - 130, size: 10, font })
  const pdfBytes = await pdf.save()
  zip.file('Axis5-DreamReport.pdf', pdfBytes)

  const blob = await zip.generateAsync({ type: 'blob' })
  const ts = new Date().toISOString().replace(/[:.]/g,'-')
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = `Axis5-Dream-${ts}.zip`
  a.click()
  URL.revokeObjectURL(a.href)
}
