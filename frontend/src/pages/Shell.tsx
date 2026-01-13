import React, { useEffect, useMemo, useState } from 'react'
import { NavLink, Outlet, useNavigate, useLocation } from 'react-router-dom'
import ThemeToggle from '../components/ThemeToggle'
import ToastHub from '../components/ToastHub'
import RightChat from '../components/RightChat'
import Viewer from '../components/Viewer'
import IntentDrawer from '../components/IntentDrawer'
import BestNextStep from '../components/BestNextStep'
import { uploadStep } from '../lib/api'
import { exportDreamZip } from '../lib/export'
import type { Detection, OcrWord } from '../lib/vision'
import * as pdfjsLib from 'pdfjs-dist';

// Dream Mode Imports
import DreamCanvas from '../features/dream/DreamCanvas'
import GhostLayer, { GhostStroke, Annotation } from '../features/dream/GhostLayer'
import { useStrokeIngestion } from '../features/dream/hooks/useStrokeIngestion'

// Start Worker
pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;

export type Intent = {
  goal?: string
  use?: string
  character?: string
  priority?: string
  grammar?: string
  surface?: string
  constraints?: string
}

export default function Shell() {
  const navigate = useNavigate()
  const location = useLocation();

  // Force Dream Mode when clicking "Dream" tab (navigating to /dream root or build)
  useEffect(() => {
    if (location.pathname.startsWith('/dream')) {
      setArtifact('dream');
    }
  }, [location.pathname]);
  // Ensure location is used
  const [drawerOpen, setDrawerOpen] = useState(false)
  const [intent, setIntent] = useState<Intent>({})
  // Default to dream mode if at root /dream
  const [artifact, setArtifact] = useState<'none' | 'sketch' | 'cad' | 'dream'>(
    (location.pathname.startsWith('/dream')) ? 'dream' : 'none'
  )
  const [glbUrl, setGlbUrl] = useState<string | null>(null)
  const [sketchUrl, setSketchUrl] = useState<string | null>(null)
  const [pdfUrl, setPdfUrl] = useState<string | null>(null)
  const [viewerMessage, setViewerMessage] = useState<string | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [objUrl, setObjUrl] = useState<string | null>(null)
  const [objFile, setObjFile] = useState<File | null>(null)
  const [mtlFile, setMtlFile] = useState<File | null>(null)
  const [detected, setDetected] = useState<{ label: string; prob: number } | null>(null)
  const [sceneCandidates, setSceneCandidates] = useState<Array<{ label: string; score: number }> | null>(null)
  const [sceneUncertainty, setSceneUncertainty] = useState<{ margin?: number; entropy_norm?: number } | null>(null)
  const [ocrSummary, setOcrSummary] = useState<string | null>(null)
  const [ocrFullText, setOcrFullText] = useState<string | null>(null)
  const [ocrPages, setOcrPages] = useState<Array<{ page: number; text: string }> | null>(null)
  const [perceptionV1, setPerceptionV1] = useState<any | null>(null)
  const [judgement, setJudgement] = useState<{ kind?: string; confidence?: number; evidence?: any; summary?: string } | null>(null)
  const [engineering, setEngineering] = useState<any | null>(null)
  const [engineeringDom, setEngineeringDom] = useState<any | null>(null)
  const [engineeringDrawing, setEngineeringDrawing] = useState<any | null>(null)
  const [engineeringUnderstanding, setEngineeringUnderstanding] = useState<any | null>(null)
  const [understanding, setUnderstanding] = useState<{ kind: string; payload?: any } | null>(null)
  const [contacts, setContacts] = useState<{ emails?: string[]; phones?: string[]; urls?: string[]; address_like?: string[]; icon_hints?: any } | null>(null)
  const [dimensions, setDimensions] = useState<Array<{ kind?: string; value?: number; unit?: string | null; prefix?: string | null; raw?: string }> | null>(null)
  const [chatObjects, setChatObjects] = useState<Array<{ class?: string; confidence?: number; class_clip_guess?: string; clip_score?: number }> | null>(null)
  const [boxes, setBoxes] = useState<Detection[] | null>(null)
  const [words, setWords] = useState<OcrWord[] | null>(null)
  const [masterDescription, setMasterDescription] = useState<string | null>(null)
  const [resolvedImageUrls, setResolvedImageUrls] = useState<string[] | null>(null)
  const [resolvedImageUrl, setResolvedImageUrl] = useState<string | null>(null) // Legacy/Fallback
  const [showGhost, setShowGhost] = useState(true);

  // --- Dream Mode State ---
  const { strokes, startStroke, addPoint, endStroke } = useStrokeIngestion();
  const [ghostStrokes, setGhostStrokes] = useState<GhostStroke[]>([]);
  const [symmetryAxis, setSymmetryAxis] = useState<{ x: number; type: 'vertical' } | null>(null);
  const [isDreamThinking, setIsDreamThinking] = useState(false);
  const [lockedGraph, setLockedGraph] = useState<any>(null);

  // --- Template-Based Sketch Warping State ---
  const [warpedImageBase64, setWarpedImageBase64] = useState<string | null>(null);
  const [comparisonImageBase64, setComparisonImageBase64] = useState<string | null>(null);
  const [warpCorrections, setWarpCorrections] = useState<string[]>([]);
  const [showWarpedImage, setShowWarpedImage] = useState(false);
  const [isWarping, setIsWarping] = useState(false);
  const [selectedSegment, setSelectedSegment] = useState<string>('hatchback');
  const [selectedView, setSelectedView] = useState<string>('front_3q');

  const onPointerDown = (e: React.PointerEvent) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    startStroke(e.nativeEvent.offsetX, e.nativeEvent.offsetY, e.pressure);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (e.buttons !== 1) return;
    addPoint(e.nativeEvent.offsetX, e.nativeEvent.offsetY, e.pressure);
  };
  const onPointerUp = (e: React.PointerEvent) => endStroke();

  // Dream Logic Loop
  useEffect(() => {
    if (strokes.length === 0) return;
    const timer = setTimeout(async () => {
      setIsDreamThinking(true);
      try {
        const res = await fetch('/api/v1/dream/ingest', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ strokes, canvas_width: window.innerWidth, canvas_height: window.innerHeight })
        });
        const data = await res.json();
        if (data.clean_strokes) {
          setGhostStrokes(data.clean_strokes.map((s: any) => ({
            id: s.id, points: s.points, color: '#00ff00', width: 3
          })));
        }
        if (data.structure?.symmetries?.length > 0 && data.structure.symmetries[0].type === 'vertical') {
          setSymmetryAxis({ x: data.structure.symmetries[0].x, type: 'vertical' });
        }
      } catch (e) { console.error(e) }
      finally { setIsDreamThinking(false); }
    }, 800);
    return () => clearTimeout(timer);
  }, [strokes]);

  // Lock Intent Handler
  const onLockIntent = async () => {
    setIsDreamThinking(true);
    try {
      const res = await fetch('/api/v1/dream/lock', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strokes, canvas_width: window.innerWidth, canvas_height: window.innerHeight })
      });
      const graph = await res.json();
      setLockedGraph(graph);
      setJudgement({
        kind: 'sketch_critique',
        confidence: 0.9,
        summary: `Intent Locked. ${graph.nodes?.length} primitives. ${graph.structure?.symmetries?.length ? 'Symmetric.' : ''} Ready for build.`,
        evidence: graph
      });
      setUnderstanding({ kind: 'dream_sketch', payload: graph });
      window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Intent Locked' }));
      // Navigate or update UI to show Build button?
      navigate('/dream/build', { state: { lockedGraph: graph } });
    } catch (e) { console.error(e); }
    finally { setIsDreamThinking(false); }
  };

  // --- Mentor Mode State ---
  const [correctionStrength, setCorrectionStrength] = useState<number>(0.665);
  const [isTuning, setIsTuning] = useState(false);

  const tuneSketch = async (newStrength: number) => {
    if (!sketchUrl && !resolvedImageUrls?.[0]) return;
    setIsTuning(true);
    setCorrectionStrength(newStrength); // Update UI immediately

    try {
      // Get the image as base64 (Original)
      const imageUrl = sketchUrl || resolvedImageUrls?.[0];
      let imageBase64 = '';
      if (imageUrl) {
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        imageBase64 = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.readAsDataURL(blob);
        });
      }

      const res = await fetch('/api/v1/dream/mentor/tune', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          base_image: imageBase64,
          segment: selectedSegment,
          view: selectedView,
          strength: newStrength
        })
      });

      const result = await res.json();
      if (!result.success) {
        window.dispatchEvent(new CustomEvent('axis5:toast', { detail: `Tuning failed: ${result.message}` }));
        return;
      }

      // Update with tuned result
      setWarpedImageBase64(result.corrected_image_base64);
      window.dispatchEvent(new CustomEvent('axis5:toast', { detail: `Strength updated to ${newStrength}` }));

    } catch (e) {
      console.error(e);
      window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Tuning failed' }));
    } finally {
      setIsTuning(false);
    }
  };

  // --- Geometric Sketch Warping ---
  const warpSketch = async () => {
    if (!sketchUrl && !resolvedImageUrls?.[0]) {
      window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'No sketch to warp' }));
      return;
    }

    setIsWarping(true);
    setWarpedImageBase64(null);
    setWarpCorrections([]);

    try {
      // Get the image as base64
      const imageUrl = sketchUrl || resolvedImageUrls?.[0];
      let imageBase64 = '';

      if (imageUrl) {
        const response = await fetch(imageUrl);
        const blob = await response.blob();
        imageBase64 = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result as string);
          reader.readAsDataURL(blob);
        });
      }

      const res = await fetch('/api/v1/dream/correct', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: imageBase64,
          segment: selectedSegment,
          view: selectedView,
          strength: correctionStrength
        })
      });

      const result = await res.json();

      if (!result.success) {
        window.dispatchEvent(new CustomEvent('axis5:toast', { detail: `Warp failed: ${result.message}` }));
        return;
      }

      // Store the warped image and comparison
      setWarpedImageBase64(result.warped_image_base64);
      setComparisonImageBase64(result.comparison_image_base64 || null);
      setWarpCorrections(result.corrections || []);
      setShowWarpedImage(true);

      const correctionCount = result.corrections?.length || 0;
      window.dispatchEvent(new CustomEvent('axis5:toast', {
        detail: correctionCount > 0
          ? `Sketch warped! ${correctionCount} proportional corrections applied.`
          : 'Proportions already correct!'
      }));

    } catch (e) {
      console.error('Warp failed:', e);
      window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Failed to warp sketch' }));
    } finally {
      setIsWarping(false);
    }
  };

  // --- Feedback Loop ---
  const applyCorrection = () => {
    if (!warpedImageBase64) return;

    // Convert base64 to Blob URL for consistency
    fetch(warpedImageBase64)
      .then(res => res.blob())
      .then(blob => {
        const newUrl = URL.createObjectURL(blob);
        setSketchUrl(newUrl);
        // If we had a resolvedImageUrl (upload), override it too or just rely on sketchUrl
        setResolvedImageUrls(null);

        setWarpedImageBase64(null);
        setComparisonImageBase64(null);
        setShowWarpedImage(false);
        window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Correction Applied! 🚀' }));
      });
  };

  const discardCorrection = () => {
    setWarpedImageBase64(null);
    setComparisonImageBase64(null);
    setShowWarpedImage(false);
    window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Correction Discarded' }));
  };


  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isToggle = (e.key === '\\' && (e.metaKey || e.ctrlKey))
      if (isToggle) { e.preventDefault(); setDrawerOpen(v => !v); return }
      if (e.key === 'Escape') setDrawerOpen(false)
    }
    const onArtifact = (e: any) => {
      const kind = e?.detail?.kind as 'sketch' | 'cad' | 'unknown'
      const file: File | undefined = e?.detail?.file
      if (kind === 'sketch' && file) {
        const url = URL.createObjectURL(file)
        setSketchUrl(url)
        setGlbUrl(null)
        setArtifact('sketch')
        setMasterDescription(null)
        setResolvedImageUrls(null)
        setResolvedImageUrl(null)
      } else if ((kind === 'cad') && file) {
        const ext = (file.name.split('.').pop() || '').toLowerCase()
        if (ext === 'glb') {
          const url = URL.createObjectURL(file)
          setGlbUrl(url)
          setSketchUrl(null)
          setArtifact('cad')
        } else {
          // STEP/STP handled by upload endpoint
          onDropFile(file)
        }
      } else if (kind === 'sketch' || kind === 'cad') {
        setArtifact(kind)
      }
    }
    const onIntentOpen = () => setDrawerOpen(true)
    window.addEventListener('keydown', onKey)
    window.addEventListener('axis5:artifact', onArtifact as any)
    window.addEventListener('axis5:intent:open', onIntentOpen as any)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('axis5:artifact', onArtifact as any)
      window.removeEventListener('axis5:intent:open', onIntentOpen as any)
    }
  }, [])

  const onBestNext = () => {
    if (artifact === 'sketch') navigate('/dream/analyze', { state: { artifact } })
    else if (artifact === 'cad') navigate('/dream/mentor', { state: { artifact } })
    else setDrawerOpen(true)
  }

  // Visual Session Caching: Deep Read & Image Resolution
  useEffect(() => {
    if (!sketchUrl) return

    const resolveImage = async () => {
      // DEBUG TOAST
      // window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Debug: Starting Image Resolution...' }))
      setResolvedImageUrls(null)
      setResolvedImageUrl(null)
      let resolved: string[] = []

      // DEBUG: Check file type
      const isPdf = sketchUrl.includes('.pdf') || (pdfUrl && sketchUrl === pdfUrl)

      if (isPdf) {
        console.log("Resolving PDF Pages...", sketchUrl)
        try {
          const loadingTask = pdfjsLib.getDocument(sketchUrl);
          const pdf = await loadingTask.promise;
          const totalPages = Math.min(pdf.numPages, 5); // Cap at 5 pages for cost/perf

          for (let i = 1; i <= totalPages; i++) {
            const page = await pdf.getPage(i);
            const viewport = page.getViewport({ scale: 1.5 });
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            if (context) {
              // @ts-ignore
              await page.render({ canvasContext: context, viewport: viewport }).promise;
              resolved.push(canvas.toDataURL('image/jpeg'));
            }
          }
        } catch (e) {
          console.error("PDF Render Error", e)
          window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Debug: PDF Render Failed' }))
        }
      } else if (sketchUrl.startsWith('blob:')) {
        try {
          const blob = await fetch(sketchUrl).then(r => r.blob())
          // Double check mime type
          if (blob.type === 'application/pdf') {
            // Recursive call or duplicate logic?
            // To be safe, let's treat it as generic image unless explicit.
            // Actually, if blob is PDF, we should leverage the logic above.
            // For now, assume generic image if not explicitly PDF extension/flag,
            // but we can add blob-type checking if robust.
          }

          const url = await new Promise<string>((resolve) => {
            const reader = new FileReader()
            reader.onload = () => resolve(reader.result as string)
            reader.readAsDataURL(blob)
          })
          resolved.push(url)
        } catch (e) {
          console.error("Failed to resolve blob", e)
          window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Debug: Blob Resolution Failed' }))
        }
      } else {
        resolved.push(sketchUrl)
      }

      if (resolved.length > 0) {
        // window.dispatchEvent(new CustomEvent('axis5:toast', { detail: `Debug: Resolved ${resolved.length} images` }))
        setResolvedImageUrls(resolved)
        setResolvedImageUrl(resolved[0] || null)
      } else {
        window.dispatchEvent(new CustomEvent('axis5:toast', { detail: 'Debug: No images resolved!' }))
      }
    }
    resolveImage()
  }, [sketchUrl, pdfUrl])

  useEffect(() => {
    if (!resolvedImageUrls || resolvedImageUrls.length === 0 || !perceptionV1 || masterDescription) return
    const triggerDeepRead = async () => {
      try {
        console.log("Triggering Deep Read...", resolvedImageUrls.length, "pages")
        const res = await fetch('/api/chat/describe', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            context: {
              imageUrls: resolvedImageUrls, // Send Array
              perception_v1: perceptionV1,
              ocrFullText: ocrFullText,
              ocrSummary: ocrSummary,
              engineeringUnderstanding,
              understanding
            }
          })
        })
        if (res.ok) {
          const j = await res.json()
          if (j.description) {
            setMasterDescription(j.description)
            console.log("Deep Read Complete.")
            window.dispatchEvent(new CustomEvent('axis5:toast', { detail: `Intelligence: Deep Read Complete (${resolvedImageUrls.length} pgs)` }))
          }
        }
      } catch (e) {
        console.error("Deep Read Failed", e)
      }
    }
    // Increased delay for multi-page render
    const t = setTimeout(triggerDeepRead, 3000)
    return () => clearTimeout(t)
  }, [resolvedImageUrls, perceptionV1, masterDescription, ocrFullText])

  const onDropFile = async (file: File) => {
    const ext = (file.name.split('.').pop() || '').toLowerCase()
    console.log('[upload] received', { name: file.name, ext })
    if (['png', 'jpg', 'jpeg', 'webp'].includes(ext) || ext === 'pdf') {
      const url = URL.createObjectURL(file)
      if (ext === 'pdf') {
        setPdfUrl(url)
        setSketchUrl(null)
      } else {
        setSketchUrl(url)
        setPdfUrl(null)
      }
      setGlbUrl(null)
      setObjUrl(null)
      setObjFile(null)
      setMtlFile(null)
      setArtifact('sketch')
      setViewerMessage(ext === 'pdf' ? 'Showing PDF' : 'Showing sketch image')
      setAnalyzing(true)
      setDetected(null)
      setSceneCandidates(null)
      setSceneUncertainty(null)
      setOcrSummary(null)
      setOcrFullText(null)
      setOcrPages(null)
      setPerceptionV1(null)
      setJudgement(null)
      setEngineering(null)
      setEngineeringDom(null)
      setEngineeringUnderstanding(null)
      setUnderstanding(null)
      setContacts(null)
      setDimensions(null)
      setChatObjects(null)
      setBoxes(null)
      setWords(null)
      // Backend-assisted recognition via Dream PoC (YOLO/CLIP/OCR)
      try {
        const fd = new FormData()
        fd.append('file', file)
        const res = await fetch('/api/v1/dream/analyze?background=false', { method: 'POST', body: fd })
        if (!res.ok) {
          let detail = ''
          try {
            const txt = await res.text()
            detail = txt ? `: ${txt.slice(0, 240).replace(/\s+/g, ' ')}` : ''
          } catch {
            // ignore
          }
          throw new Error(`dream analyze ${res.status}${detail}`)
        }
        const out = await res.json()

        const sceneLabel = out?.scene?.top_label || out?.scene?.label || null
        const sceneScore = typeof out?.scene?.top_score === 'number' ? out.scene.top_score : (typeof out?.scene?.score === 'number' ? out.scene.score : null)
        const topObj = Array.isArray(out?.objects) && out.objects.length ? out.objects[0] : null

        const judgementKind = (out?.judgement && typeof out.judgement === 'object' && typeof out.judgement.kind === 'string')
          ? String(out.judgement.kind)
          : null

        // Prefer judgement for engineering drawings; otherwise prefer grounded detections (YOLO) over CLIP scene guess.
        const isEng = judgementKind === 'engineering_drawing' || (ext === 'pdf' && typeof sceneLabel === 'string' && sceneLabel.toLowerCase().includes('engineering drawing'))
        const label = isEng ? 'engineering drawing' : (topObj?.class || sceneLabel || topObj?.class_clip_guess || null)
        const score = (isEng && typeof sceneScore === 'number')
          ? sceneScore
          : (typeof topObj?.confidence === 'number' ? topObj.confidence : (typeof sceneScore === 'number' ? sceneScore : null))

        if (Array.isArray(out?.objects)) {
          setChatObjects(
            out.objects
              .filter((o: any) => o && (typeof o.class === 'string' || typeof o.class_clip_guess === 'string'))
              .map((o: any) => ({
                class: typeof o.class === 'string' ? o.class : undefined,
                confidence: typeof o.confidence === 'number' ? o.confidence : undefined,
                class_clip_guess: typeof o.class_clip_guess === 'string' ? o.class_clip_guess : undefined,
                clip_score: typeof o.clip_score === 'number' ? o.clip_score : undefined,
              }))
              .slice(0, 12)
          )
        }

        if (Array.isArray(out?.scene?.candidates) && out.scene.candidates.length) {
          setSceneCandidates(
            out.scene.candidates
              .filter((c: any) => c && typeof c.label === 'string')
              .map((c: any) => ({ label: String(c.label), score: Number(c.score || 0) }))
              .slice(0, 7)
          )
        }
        if (out?.scene?.uncertainty && typeof out.scene.uncertainty === 'object') {
          setSceneUncertainty({
            margin: typeof out.scene.uncertainty.margin === 'number' ? out.scene.uncertainty.margin : undefined,
            entropy_norm: typeof out.scene.uncertainty.entropy_norm === 'number' ? out.scene.uncertainty.entropy_norm : undefined,
          })
        }

        if (label) {
          setDetected({ label, prob: typeof score === 'number' ? Math.max(0, Math.min(1, score)) : 0.5 })
        }

        // Image-only overlays (skip for PDF)
        if (ext !== 'pdf') {
          const img = new Image()
          img.onload = () => {
            const w = img.naturalWidth || 1
            const h = img.naturalHeight || 1
            const dets = (Array.isArray(out?.objects) ? out.objects : [])
              .filter((o: any) => Array.isArray(o?.bbox) && o.bbox.length === 4)
              .map((o: any) => {
                const [x1, y1, x2, y2] = o.bbox
                const nx = Math.max(0, Math.min(1, x1 / w))
                const ny = Math.max(0, Math.min(1, y1 / h))
                const nw = Math.max(0, Math.min(1, (x2 - x1) / w))
                const nh = Math.max(0, Math.min(1, (y2 - y1) / h))
                return { label: String(o?.class || o?.label || 'object'), score: Number(o?.confidence || 0), bbox: [nx, ny, nw, nh] as [number, number, number, number] }
              })
            if (dets.length) setBoxes(dets)
          }
          img.src = url
        }

        if (typeof out?.text?.summary === 'string' && out.text.summary.trim()) {
          setOcrSummary(String(out.text.summary))
          const sample = String(out.text.summary).trim().slice(0, 60).replace(/\s+/g, ' ')
          if (sample) window.dispatchEvent(new CustomEvent('axis5:toast', { detail: `OCR: ${sample}${String(out.text.summary).length > 60 ? '…' : ''}` }))
        }
        if (typeof out?.text?.full_text === 'string' && out.text.full_text.trim()) {
          setOcrFullText(String(out.text.full_text))
        }

        if (out?.perception_v1 && typeof out.perception_v1 === 'object') {
          setPerceptionV1(out.perception_v1)
          try {
            const ptxt = (out.perception_v1 as any)?.page_text
            const p1 = ptxt && typeof ptxt === 'object' ? String(ptxt['1'] || '').trim() : ''
            if (p1 && (!out?.text?.summary || !String(out.text.summary || '').trim())) {
              const preview = p1.slice(0, 400)
              setOcrSummary(preview + (p1.length > 400 ? '…' : ''))
            }
          } catch {
            // ignore
          }
        }

        if (out?.judgement && typeof out.judgement === 'object') {
          setJudgement(out.judgement)
        }

        if (out?.text?.engineering && typeof out.text.engineering === 'object') {
          setEngineering(out.text.engineering)
        }

        if (out?.text?.engineering_drawing_dom && typeof out.text.engineering_drawing_dom === 'object') {
          setEngineeringDom(out.text.engineering_drawing_dom)
        }

        if (out?.engineering_understanding_object && typeof out.engineering_understanding_object === 'object') {
          setEngineeringUnderstanding(out.engineering_understanding_object)
        } else if (out?.engineering_drawing && typeof out.engineering_drawing === 'object') {
          setEngineeringDrawing(out.engineering_drawing)
          const euo = (out.engineering_drawing as any)?.engineering_understanding
          if (euo && typeof euo === 'object') setEngineeringUnderstanding(euo)
        }

        // Unified understanding object for chat (single truth layer).
        try {
          const kind = (out?.judgement && typeof out.judgement.kind === 'string') ? String(out.judgement.kind) : ''
          if (kind === 'engineering_drawing') {
            if (out?.engineering_understanding_object) {
              setUnderstanding({ kind: 'engineering_drawing', payload: out.engineering_understanding_object })
            } else {
              const euo = (out?.engineering_drawing as any)?.engineering_understanding
              if (euo && typeof euo === 'object') {
                setUnderstanding({ kind: 'engineering_drawing', payload: euo })
              } else {
                setUnderstanding({ kind: 'engineering_drawing', payload: out?.engineering_drawing || null })
              }
            }
          } else if (kind && kind.includes('document')) {
            setUnderstanding({
              kind: 'document',
              payload: {
                judgement: out?.judgement || null,
                ocrSummary: out?.text?.summary || null,
                ocrFullText: out?.text?.full_text || null,
                contacts: out?.text?.extracted_fields?.contacts || null,
                engineering_understanding_object: out?.engineering_understanding_object || (out?.engineering_drawing as any)?.engineering_understanding || null,
              },
            })
          } else {
            setUnderstanding({
              kind: 'image',
              payload: {
                judgement: out?.judgement || null,
                scene: out?.scene || null,
                objects: out?.objects || null,
                ocrSummary: out?.text?.summary || null,
                engineering_understanding_object: out?.engineering_understanding_object || (out?.engineering_drawing as any)?.engineering_understanding || null,
              },
            })
          }
        } catch {
          // ignore
        }

        const c = out?.text?.extracted_fields?.contacts
        if (c && typeof c === 'object') {
          setContacts({
            emails: Array.isArray(c.emails) ? c.emails.map((x: any) => String(x)) : undefined,
            phones: Array.isArray(c.phones) ? c.phones.map((x: any) => String(x)) : undefined,
            urls: Array.isArray(c.urls) ? c.urls.map((x: any) => String(x)) : undefined,
            address_like: Array.isArray(c.address_like) ? c.address_like.map((x: any) => String(x)) : undefined,
            icon_hints: c.icon_hints,
          })
        }

        const d = out?.text?.extracted_fields?.dimensions
        if (Array.isArray(d) && d.length) {
          setDimensions(
            d
              .filter((x: any) => x && (typeof x.raw === 'string' || typeof x.value === 'number'))
              .map((x: any) => ({
                kind: typeof x.kind === 'string' ? x.kind : undefined,
                value: typeof x.value === 'number' ? x.value : undefined,
                unit: typeof x.unit === 'string' ? x.unit : null,
                prefix: typeof x.prefix === 'string' ? x.prefix : null,
                raw: typeof x.raw === 'string' ? x.raw : undefined,
              }))
              .slice(0, 40)
          )
        }

        if (Array.isArray(out?.text?.pages) && out.text.pages.length) {
          const ps = out.text.pages
            .filter((p: any) => p && typeof p.page === 'number' && typeof p.text === 'string')
            .map((p: any) => ({ page: Number(p.page), text: String(p.text || '') }))
          if (ps.length) setOcrPages(ps)
        }

        if (ext !== 'pdf' && Array.isArray(out?.text?.words) && out.text.words.length) {
          const ws = out.text.words
            .filter((w: any) => w && Array.isArray(w.bbox) && w.bbox.length === 4)
            .map((w: any) => ({
              text: String(w.text || ''),
              bbox: [Number(w.bbox[0] || 0), Number(w.bbox[1] || 0), Number(w.bbox[2] || 0), Number(w.bbox[3] || 0)] as [number, number, number, number],
            }))
          if (ws.length) setWords(ws)
        }
      } catch (e: any) {
        window.dispatchEvent(new CustomEvent('axis5:toast', { detail: `Backend recognition failed: ${e?.message || 'network'}` }))
      } finally {
        setAnalyzing(false)
      }
      return
    }
    if (ext === 'glb') {
      const url = URL.createObjectURL(file)
      setGlbUrl(url)
      setSketchUrl(null)
      setPdfUrl(null)
      setObjUrl(null)
      setObjFile(null)
      setMtlFile(null)
      setArtifact('cad')
      setUnderstanding({ kind: 'cad', payload: { file: { name: file.name, ext }, preview: { glbUrl: url } } })
      setViewerMessage('Showing GLB model')
      return
    }
    if (ext === 'obj') {
      const url = URL.createObjectURL(file)
      setObjUrl(url)
      setObjFile(file)
      // keep existing mtlFile if previously dropped
      setGlbUrl(null)
      setSketchUrl(null)
      setPdfUrl(null)
      setArtifact('cad')
      setUnderstanding({ kind: 'cad', payload: { file: { name: file.name, ext }, preview: { objUrl: url, hasMtl: Boolean(mtlFile) } } })
      setViewerMessage('Rendering OBJ locally')
      return
    }
    if (ext === 'mtl') {
      setMtlFile(file)
      setViewerMessage('MTL loaded (materials will apply if matching OBJ)')
      return
    }
    if (['step', 'stp'].includes(ext)) {
      try {
        const out = await uploadStep(file)
        if (out?.previewUrl) {
          setGlbUrl(out.previewUrl)
          setSketchUrl(null)
          setPdfUrl(null)
          setObjUrl(null)
          setObjFile(null)
          setMtlFile(null)
          setArtifact('cad')
          setUnderstanding({ kind: 'cad', payload: { file: { name: file.name, ext }, preview: { glbUrl: out.previewUrl }, note: 'STEP preview is stubbed' } })
          setViewerMessage('STEP preview is stubbed; showing demo model')
          return
        }
        // Fallback if response missing previewUrl
        setGlbUrl('https://modelviewer.dev/shared-assets/models/Astronaut.glb')
        setSketchUrl(null)
        setPdfUrl(null)
        setObjUrl(null)
        setObjFile(null)
        setMtlFile(null)
        setArtifact('cad')
        setUnderstanding({ kind: 'cad', payload: { file: { name: file.name, ext }, preview: { glbUrl: 'https://modelviewer.dev/shared-assets/models/Astronaut.glb' }, note: 'STEP preview is stubbed' } })
        setViewerMessage('Upload service returned no preview; showing demo model')
      } catch (e: any) {
        // Network/server error: show fallback demo GLB + message
        setGlbUrl('https://modelviewer.dev/shared-assets/models/Astronaut.glb')
        setSketchUrl(null)
        setPdfUrl(null)
        setObjUrl(null)
        setObjFile(null)
        setMtlFile(null)
        setArtifact('cad')
        setUnderstanding({ kind: 'cad', payload: { file: { name: file.name, ext }, preview: { glbUrl: 'https://modelviewer.dev/shared-assets/models/Astronaut.glb' }, note: 'CAD upload failed; using demo model' } })
        setViewerMessage('Upload failed; showing demo model')
      }
      return
    }
  }

  const onExport = async () => {
    await exportDreamZip({
      intent,
      chosenPack: null,
      concepts: [],
      overlays: [],
      trials: [],
    })
  }



  return (
    <div className="min-h-screen flex flex-col">
      <ToastHub />
      <header className="border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="mx-auto max-w-7xl px-4 py-3 flex items-center justify-between">
          <nav className="flex gap-4 text-sm">
            <NavLink to="/dream" className={({ isActive }) => isActive ? 'font-semibold' : ''}>Dream</NavLink>
            <button disabled className="opacity-50 cursor-not-allowed" title="Coming after Dream v1.0">Build</button>
            <button disabled className="opacity-50 cursor-not-allowed" title="Coming after Dream v1.0">Scale</button>
          </nav>
          <div className="flex items-center gap-3">
            <button onClick={() => setDrawerOpen(true)} className="text-xs border px-2 py-1 rounded">Intent ⌘/Ctrl+\\</button>
            <button onClick={onExport} className="text-xs border px-2 py-1 rounded">Export</button>
            <ThemeToggle />
          </div>
        </div>
        <div className="border-t" style={{ borderColor: 'var(--border)' }}>
          <div className="mx-auto max-w-7xl px-4 py-2">
            <nav className="flex gap-3 text-xs">
              <NavLink to="/dream/analyze" className={({ isActive }) => isActive ? 'font-semibold underline' : ''}>Analyze</NavLink>
              <NavLink to="/dream/mentor" className={({ isActive }) => isActive ? 'font-semibold underline' : ''}>Mentor</NavLink>
              <NavLink to="/dream/ideate" className={({ isActive }) => isActive ? 'font-semibold underline' : ''}>Ideate</NavLink>
            </nav>
          </div>
        </div>
      </header>

      <BestNextStep artifact={artifact as any} onAction={onBestNext} />

      <div className="flex-1 w-full px-4 py-4 min-h-0">
        <div className="h-[calc(100vh-220px)] min-h-[520px] min-h-0 grid grid-cols-1 md:grid-cols-[70%_30%] gap-4">
          <div className="h-full rounded-3xl border overflow-hidden relative" style={{ background: 'var(--panel)', borderColor: 'var(--border)' }}>
            <Viewer
              glbUrl={glbUrl}
              sketchUrl={sketchUrl}
              pdfUrl={pdfUrl}
              objUrl={objUrl}
              objFile={objFile}
              mtlFile={mtlFile}
              onDropFile={onDropFile}
              message={viewerMessage}
              preferSketch={artifact === 'sketch'}
              boxes={boxes || undefined}
              words={words || undefined}
              busy={analyzing}
              busyLabel={pdfUrl ? "Analyzing PDF…" : "Analyzing image…"}
            />
            {(artifact === 'dream' || artifact === 'sketch') && (
              <div
                className={`absolute inset-0 z-10 bg-transparent pointer-events-none`}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  if (e.dataTransfer.files?.[0]) {
                    onDropFile(e.dataTransfer.files[0]);
                  }
                }}
              >
                <DreamCanvas
                  strokes={strokes}
                  onPointerDown={onPointerDown}
                  onPointerMove={onPointerMove}
                  onPointerUp={onPointerUp}
                />
                {/* Ghost Layer Disabled per User Request */}
                {/* {showGhost && (
                  <GhostLayer
                     ...
                  />
                )} */}

                {/* Warped Sketch Display with Comparison */}
                {showWarpedImage && (warpedImageBase64 || comparisonImageBase64) && (
                  <div className="absolute inset-0 z-40 flex items-center justify-center bg-black/80">
                    <div className="relative max-w-[95%] max-h-[90%] flex flex-col items-center">
                      {/* Show comparison if available, otherwise warped */}
                      <img
                        src={comparisonImageBase64 || warpedImageBase64 || ''}
                        alt="Warped Sketch Comparison"
                        className="max-w-full max-h-[70vh] rounded-lg shadow-2xl"
                      />
                      <div className="mt-3 bg-black/70 text-white p-3 rounded-lg text-sm max-w-[90%]">
                        <div className="font-bold text-green-400 mb-1">Proportions Corrected</div>
                        <div className="text-xs text-gray-300 mb-2">
                          {warpCorrections.length > 0
                            ? warpCorrections.join(' • ')
                            : 'Proportions were already correct!'}
                        </div>
                        {comparisonImageBase64 && (
                          <div className="text-xs text-yellow-300">
                            🔴 RED = Original position | 🟢 GREEN = Corrected position
                          </div>
                        )}
                      </div>
                      {/* Apply / Discard Actions */}
                      <div className="flex gap-4 mt-4 pointer-events-auto z-50">
                        <button
                          onClick={discardCorrection}
                          className="px-6 py-2 rounded-full border border-white/30 text-white hover:bg-white/10 transition-colors"
                        >
                          Discard
                        </button>
                        <button
                          onClick={applyCorrection}
                          className="px-6 py-2 rounded-full bg-green-500 text-black font-bold hover:bg-green-400 transition-colors shadow-xl"
                        >
                          Apply Correction
                        </button>
                      </div>
                    </div>
                  </div>
                )}


                {/* Bottom Button Bar */}
                <div className="absolute bottom-4 left-4 flex gap-2 pointer-events-none flex-wrap">
                  <button onClick={onLockIntent} className="pointer-events-auto bg-green-500 text-black font-bold px-4 py-2 rounded-full shadow-lg hover:bg-green-400">
                    LOCK INTENT
                  </button>

                  {/* Segment Selector */}
                  <select
                    value={selectedSegment}
                    onChange={(e) => setSelectedSegment(e.target.value)}
                    className="pointer-events-auto px-3 py-2 rounded-full bg-black/50 text-white border border-white/30 text-sm"
                  >
                    <option value="scooter">Scooter</option>
                    <option value="hatchback">Hatchback</option>
                    <option value="sedan">Sedan</option>
                    <option value="suv">SUV</option>
                  </select>

                  {/* View Selector */}
                  <select
                    value={selectedView}
                    onChange={(e) => setSelectedView(e.target.value)}
                    className="pointer-events-auto px-3 py-2 rounded-full bg-black/50 text-white border border-white/30 text-sm"
                  >
                    <option value="front_3q">Front 3/4</option>
                    <option value="rear_3q">Rear 3/4</option>
                    <option value="side">Side</option>
                  </select>

                  {/* MENTOR SLIDER (Toolbar) */}
                  <div className="pointer-events-auto flex items-center gap-2 bg-black/50 px-3 py-2 rounded-full border border-white/30">
                    <span className="text-xs font-bold text-purple-300">STR: {correctionStrength}</span>
                    <input
                      type="range" min="0.4" max="0.8" step="0.01" value={correctionStrength}
                      onChange={(e) => setCorrectionStrength(parseFloat(e.target.value))}
                      className="w-20 accent-purple-500 cursor-pointer"
                    />
                  </div>

                  {/* Generative Sketch Correction */}
                  <button
                    onClick={warpSketch}
                    disabled={isWarping}
                    className={`pointer-events-auto px-4 py-2 rounded-full shadow-lg font-bold border transition-colors ${isWarping
                      ? 'bg-yellow-500/50 text-black border-yellow-500 animate-pulse'
                      : 'bg-purple-600 text-white border-purple-400 hover:bg-purple-500'
                      }`}
                  >
                    {isWarping ? '🔄 Correcting...' : '✨ AI Correct'}
                  </button>

                  {/* View Corrected Result */}
                  {warpedImageBase64 && (
                    <button
                      onClick={() => setShowWarpedImage(!showWarpedImage)}
                      className={`pointer-events-auto px-4 py-2 rounded-full shadow-lg font-bold border transition-colors ${showWarpedImage
                        ? 'bg-green-500 text-black border-green-300'
                        : 'bg-black/50 text-white border-white/50'
                        }`}
                    >
                      {showWarpedImage ? '🟢 Corrected: ON' : '⚪ View Corrected'}
                    </button>
                  )}

                  {isDreamThinking && <span className="text-green-500 animate-pulse self-center font-mono">DREAMING...</span>}
                </div>
                {/* DEBUG PANEL REMOVED */}
              </div>
            )}

          </div>

          <div className="h-full rounded-3xl border overflow-hidden" style={{ background: 'var(--panel)', borderColor: 'var(--border)' }}>
            <RightChat
              onAttach={(kind: 'sketch' | 'cad') => setArtifact(kind)}
              context={{
                artifact,
                understanding,
                detectedLabel: detected?.label || null,
                detectedProb: detected?.prob ?? null,
                sceneCandidates,
                sceneUncertainty,
                ocrSummary,
                ocrFullText,
                ocrPages,
                perception_v1: perceptionV1,
                judgement,
                engineering,
                engineeringDom,
                engineeringDrawing,
                engineeringUnderstanding,
                contacts,
                dimensions,
                objects: chatObjects,
                imageUrl: (resolvedImageUrls && resolvedImageUrls[0]) || sketchUrl,
                warpedImageBase64,
                warpCorrections,
                isWarping,
                onRequestWarp: warpSketch,
              }}
            />
          </div>
        </div>
      </div>

      <IntentDrawer open={drawerOpen} onClose={() => setDrawerOpen(false)} intent={intent} setIntent={setIntent} />
    </div>
  )
}
