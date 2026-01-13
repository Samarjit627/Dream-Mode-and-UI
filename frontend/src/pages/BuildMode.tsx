import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import BuildViewer from '../features/builder/BuildViewer';

export default function BuildMode() {
    const location = useLocation();
    const navigate = useNavigate();
    const lockedGraph = location.state?.lockedGraph;

    console.log("Locked Graph State:", lockedGraph);

    const [stlBlob, setStlBlob] = useState<Blob | null>(null);
    const [status, setStatus] = useState<string>("Initializing Build Engine...");
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!lockedGraph) {
            console.error("No locked graph found, redirecting to Dream Mode.");
            navigate('/dream/mode');
            return;
        }

        const buildGeometry = async () => {
            setStatus("Extruding Solids...");
            try {
                const res = await fetch('/api/v1/dream/build', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(lockedGraph)
                });

                if (!res.ok) throw new Error("Build Failed");

                const blob = await res.blob();
                setStlBlob(blob);
                setStatus("Complete");
            } catch (e) {
                console.error(e);
                setError("Failed to generate geometry. Check the console.");
            }
        };

        buildGeometry();
    }, [lockedGraph, navigate]);

    if (error) {
        return (
            <div className="w-full h-full flex items-center justify-center bg-neutral-950 text-red-500 font-mono">
                Error: {error}
            </div>
        )
    }

    return (
        <div className="w-full h-full relative bg-neutral-950 flex flex-col">
            {/* Header / HUD */}
            <div className="absolute top-0 left-0 right-0 p-4 z-10 flex justify-between items-center pointer-events-none">
                <h1 className="text-white font-bold text-xl tracking-wider opacity-80">BUILD MODE</h1>
                <div className="font-mono text-sm text-green-500">
                    {status}
                </div>
            </div>

            {/* Viewer */}
            <div className="flex-1 w-full h-full">
                {stlBlob && <BuildViewer stlBlob={stlBlob} />}
            </div>

            {/* Footer Controls */}
            {stlBlob && (
                <div className="absolute bottom-8 left-0 right-0 flex justify-center gap-4 pointer-events-none">
                    <button
                        className="pointer-events-auto px-6 py-2 bg-neutral-800 text-white border border-neutral-600 rounded hover:bg-neutral-700"
                        onClick={() => {
                            const url = URL.createObjectURL(stlBlob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = "dream_build_v1.stl";
                            a.click();
                            URL.revokeObjectURL(url);
                        }}
                    >
                        DOWNLOAD STL
                    </button>
                    <button
                        className="pointer-events-auto px-6 py-2 bg-neutral-800 text-white border border-neutral-600 rounded hover:bg-neutral-700"
                        onClick={() => navigate('/dream/mode')}
                    >
                        BACK TO DREAM
                    </button>
                </div>
            )}
        </div>
    );
}
