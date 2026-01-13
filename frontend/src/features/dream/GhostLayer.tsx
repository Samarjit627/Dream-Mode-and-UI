import React, { useRef, useEffect } from 'react';

export type GhostStroke = {
    id: string;
    points: { x: number; y: number }[];
    color: string;
    width: number;
    type?: 'line' | 'arc' | 'circle'; // Primitive type
};

export type GhostElement = {
    id: string;
    shape: 'blob' | 'axis' | 'zone';
    x: number;
    y: number;
    // Blob props
    r?: number;
    opacity?: number;
    // Axis props
    type?: 'vertical' | 'horizontal';
    length?: number;
};

export type Annotation = {
    id: string;
    x: number;
    y: number;
    text: string;
    type: 'critique' | 'info';
};

export type GhostLayerProps = {
    ghostStrokes: GhostStroke[];
    annotations: Annotation[];
    ghostElements?: GhostElement[];
};

export default function GhostLayer({ ghostStrokes, annotations, ghostElements = [] }: GhostLayerProps) {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // 0. Render Semantic Ghost Elements (Mass, Symmetry, etc.)
        ghostElements.forEach(el => {
            if (el.shape === 'blob') {
                // Type 1: Mass & Balance (Soft Blob)
                // Draw a radial gradient
                if (!el.r) return;
                const g = ctx.createRadialGradient(el.x, el.y, 0, el.x, el.y, el.r);
                const alpha = el.opacity || 0.4;
                g.addColorStop(0, `rgba(0, 255, 255, ${alpha})`); // Cyan Core
                g.addColorStop(1, 'rgba(0, 255, 255, 0)'); // Fade out

                ctx.fillStyle = g;
                ctx.beginPath();
                ctx.arc(el.x, el.y, el.r, 0, Math.PI * 2);
                ctx.fill();
            } else if (el.shape === 'axis') {
                // Type 3: Symmetry Axis
                ctx.beginPath();
                ctx.setLineDash([15, 15]);
                ctx.strokeStyle = '#00ffff';
                ctx.lineWidth = 1.5;
                ctx.globalAlpha = 0.6;
                // Draw full height if vertical
                if (el.type === 'vertical') {
                    ctx.moveTo(el.x, 0);
                    ctx.lineTo(el.x, canvas.height);
                }
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.globalAlpha = 1.0;
            }
        });

        // 1. Draw Ghost Strokes (The "Clean" Intent)
        ghostStrokes.forEach(stroke => {
            if (stroke.points.length < 2) return;
            ctx.beginPath();
            ctx.strokeStyle = stroke.color || '#00ff00';
            ctx.lineWidth = stroke.width || 3;
            ctx.globalAlpha = 0.6; // Ghostly transparency

            const first = stroke.points[0];
            ctx.moveTo(first.x, first.y);
            for (let i = 1; i < stroke.points.length; i++) {
                ctx.lineTo(stroke.points[i].x, stroke.points[i].y);
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        });

    }, [ghostStrokes, ghostElements]);

    return (
        <div className="absolute inset-0 pointer-events-none z-50">
            <canvas
                ref={canvasRef}
                width={window.innerWidth}
                height={window.innerHeight}
                className="absolute inset-0"
            />
            {/* Render Annotations */}
            {annotations.map(ann => (
                <div
                    key={ann.id}
                    className={`absolute px-3 py-2 rounded shadow-lg text-xs max-w-[200px] border ${ann.type === 'critique' ? 'bg-red-900/80 border-red-500 text-white' : 'bg-blue-900/80 border-blue-500 text-white'
                        }`}
                    style={{ top: ann.y, left: ann.x }}
                >
                    {ann.text}
                </div>
            ))}
        </div>
    );
}
