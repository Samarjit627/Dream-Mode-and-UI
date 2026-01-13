import React, { useRef, useEffect } from 'react';
import { Stroke } from './hooks/useStrokeIngestion';

// v2.1 Cache Bust
type DreamCanvasProps = {
  strokes: Stroke[];
  onPointerDown: (e: React.PointerEvent) => void;
  onPointerMove: (e: React.PointerEvent) => void;
  onPointerUp: (e: React.PointerEvent) => void;
}

export default function DreamCanvas({ strokes, onPointerDown, onPointerMove, onPointerUp }: DreamCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear transparently
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw all strokes
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    strokes.forEach(stroke => {
      if (stroke.points.length < 2) return;
      ctx.beginPath();
      ctx.strokeStyle = stroke.color;
      ctx.lineWidth = stroke.width;

      const first = stroke.points[0];
      ctx.moveTo(first.x, first.y);

      for (let i = 1; i < stroke.points.length; i++) {
        const p = stroke.points[i];
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    });

  }, [strokes]);

  return (
    <div className="w-full h-full relative bg-transparent overflow-hidden touch-none">
      <canvas
        ref={canvasRef}
        width={window.innerWidth}
        height={window.innerHeight}
        className="absolute inset-0 cursor-crosshair touch-none"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
      />
      <div className="absolute top-4 left-4 p-4 pointer-events-none select-none">
        <h1 className="text-white/50 text-sm font-mono">DREAM RUNTIME v2.1</h1>
        <p className="text-white/30 text-xs">Strokes Captured: {strokes.length}</p>
      </div>
    </div>
  );
}
