import { useState, useRef, useCallback } from 'react';

export type Point = {
    x: number;
    y: number;
    p: number; // Pressure
    t: number; // Timestamp
};

export type Stroke = {
    id: string;
    points: Point[];
    color: string;
    width: number;
    completed: boolean;
};

export function useStrokeIngestion() {
    const [strokes, setStrokes] = useState<Stroke[]>([]);
    const currentStrokeId = useRef<string | null>(null);

    const startStroke = useCallback((x: number, y: number, pressure: number = 0.5) => {
        const id = crypto.randomUUID();
        currentStrokeId.current = id;
        const newStroke: Stroke = {
            id,
            points: [{ x, y, p: pressure, t: Date.now() }],
            color: '#ffffff',
            width: 2,
            completed: false
        };
        setStrokes(prev => [...prev, newStroke]);
        return id;
    }, []);

    const addPoint = useCallback((x: number, y: number, pressure: number = 0.5) => {
        if (!currentStrokeId.current) return;

        setStrokes(prev => prev.map(s => {
            if (s.id === currentStrokeId.current) {
                return {
                    ...s,
                    points: [...s.points, { x, y, p: pressure, t: Date.now() }]
                };
            }
            return s;
        }));
    }, []);

    const endStroke = useCallback(() => {
        if (!currentStrokeId.current) return;

        setStrokes(prev => prev.map(s => {
            if (s.id === currentStrokeId.current) {
                return { ...s, completed: true };
            }
            return s;
        }));
        currentStrokeId.current = null;
    }, []);

    return {
        strokes,
        startStroke,
        addPoint,
        endStroke
    };
}
