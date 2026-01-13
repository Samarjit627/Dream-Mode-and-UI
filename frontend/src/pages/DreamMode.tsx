import React, { useState, useEffect, useRef } from 'react'
import DreamCanvas from '../features/dream/DreamCanvas'
import GhostLayer, { GhostStroke, Annotation } from '../features/dream/GhostLayer'
import { useStrokeIngestion } from '../features/dream/hooks/useStrokeIngestion'
import { useNavigate } from 'react-router-dom';

export default function DreamMode() {
    const { strokes, startStroke, addPoint, endStroke } = useStrokeIngestion();
    const navigate = useNavigate();
    // Intelligence State
    const [ghostStrokes, setGhostStrokes] = useState<GhostStroke[]>([]);
    const [annotations, setAnnotations] = useState<Annotation[]>([]);
    const [symmetryAxis, setSymmetryAxis] = useState<{ x: number; type: 'vertical' } | null>(null);
    const [isThinking, setIsThinking] = useState(false);

    const onPointerDown = (e: React.PointerEvent) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        startStroke(e.nativeEvent.offsetX, e.nativeEvent.offsetY, e.pressure);
    };

    const onPointerMove = (e: React.PointerEvent) => {
        if (e.buttons !== 1) return;
        addPoint(e.nativeEvent.offsetX, e.nativeEvent.offsetY, e.pressure);
    };

    const onPointerUp = (e: React.PointerEvent) => {
        endStroke();
    };

    // The "Loop": When strokes change, ask the backend
    useEffect(() => {
        if (strokes.length === 0) return;

        const timer = setTimeout(async () => {
            setIsThinking(true);
            try {
                // 1. Ingest (Structure)
                const res = await fetch('/api/v1/dream/ingest', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        strokes,
                        canvas_width: window.innerWidth,
                        canvas_height: window.innerHeight
                    })
                });
                const data = await res.json();

                // Map primitives/clean strokes to Ghosts
                if (data.clean_strokes) {
                    setGhostStrokes(data.clean_strokes.map((s: any) => ({
                        id: s.id,
                        points: s.points,
                        color: '#00ff00', // Green ghosts
                        width: 3
                    })));
                }

                if (data.structure?.symmetries?.length > 0) {
                    const sym = data.structure.symmetries[0];
                    if (sym.type === 'vertical') {
                        setSymmetryAxis({ x: sym.x, type: 'vertical' });
                    }
                }

                // 2. Judge (Critique) - Triggered less frequently? 
                // For V1, we'll trigger only if explicit "Lock" or separate button?
                // Or maybe just every 5 strokes to save tokens?
                // Let's hold off on auto-judge for now to save tokens, 
                // but we can mock an annotation.

            } catch (e) {
                console.error("Dream Loop Error", e);
            } finally {
                setIsThinking(false);
            }
        }, 800); // 800ms debounce

        return () => clearTimeout(timer);
    }, [strokes]);

    // State for Locking
    const [lockedGraph, setLockedGraph] = useState<any>(null);

    const handleLock = async () => {
        setIsThinking(true);
        try {
            const res = await fetch('/api/v1/dream/lock', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    strokes,
                    canvas_width: window.innerWidth,
                    canvas_height: window.innerHeight
                })
            });
            const graph = await res.json();
            setLockedGraph(graph);
        } catch (e) {
            console.error("Lock Failed", e);
        } finally {
            setIsThinking(false);
        }
    };

    return (
        <div className="flex flex-col w-full h-full relative bg-neutral-950">
            {/* Layer 1: The Input Canvas */}
            <DreamCanvas
                strokes={strokes}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
            />

            {/* Layer 2: The Ghost Overlay */}
            <GhostLayer
                ghostStrokes={ghostStrokes}
                annotations={annotations}
                ghostElements={symmetryAxis ? [{
                    id: 'rt_sym',
                    shape: 'axis',
                    x: symmetryAxis.x,
                    y: 0,
                    type: symmetryAxis.type
                }] : []}
            />

            {/* Layer 3: HUD & Controls */}
            <div className="absolute bottom-8 left-0 right-0 flex justify-center pointer-events-none">
                {!lockedGraph ? (
                    <button
                        onClick={handleLock}
                        className="pointer-events-auto bg-white text-black px-6 py-3 rounded-full font-bold shadow-lg hover:bg-neutral-200 transition-colors flex items-center gap-2"
                    >
                        <span>🔒</span>
                        <span>LOCK INTENT</span>
                    </button>
                ) : (
                    <div className="bg-green-900/90 text-white p-6 rounded-xl border border-green-500 max-w-md pointer-events-auto">
                        <h2 className="text-xl font-bold mb-2">Intent Locked</h2>
                        <div className="space-y-1 text-sm opacity-80 mb-4">
                            <p>Primitives: {lockedGraph.nodes?.length || 0}</p>
                            <p>Structure: {lockedGraph.structure?.symmetries?.length ? 'Symmetric' : 'Asymmetric'}</p>
                            {lockedGraph.structure?.alignments?.length > 0 && <p>Alignments: {lockedGraph.structure.alignments.length}</p>}
                        </div>
                        <button
                            className="w-full bg-green-500 text-black py-2 rounded font-bold hover:bg-green-400"
                            onClick={() => navigate('/dream/build', { state: { lockedGraph } })}
                        >
                            PROCEED TO BUILD MODE →
                        </button>

                    </div>
                )}
            </div>

            {/* Status Indicator */}
            {isThinking && (
                <div className="absolute top-4 right-4 text-xs text-green-500 font-mono animate-pulse">
                    DREAMING...
                </div>
            )}
        </div>
    )
}
