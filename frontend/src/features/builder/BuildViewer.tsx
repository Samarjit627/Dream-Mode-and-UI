import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
// @ts-ignore
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
// @ts-ignore
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

type BuildViewerProps = {
    stlBlob: Blob | null;
};

export default function BuildViewer({ stlBlob }: BuildViewerProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (!containerRef.current || !stlBlob) return;

        const width = containerRef.current.clientWidth;
        const height = containerRef.current.clientHeight;

        // Setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111); // Dark background

        // Grid helper
        const grid = new THREE.GridHelper(500, 50, 0x444444, 0x222222);
        // scene.add(grid); // Grid often obscures the view if model is small. Let's keep it but maybe lower opacity.

        const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
        camera.position.set(50, 50, 50);
        camera.lookAt(0, 0, 0);

        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        containerRef.current.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;

        // Lights
        const light = new THREE.DirectionalLight(0xffffff, 1);
        light.position.set(100, 100, 100);
        scene.add(light);

        const ambient = new THREE.AmbientLight(0x404040);
        scene.add(ambient);

        // Load STL
        const loader = new STLLoader();
        const url = URL.createObjectURL(stlBlob);

        console.log("BuildViewer: Loading STL from blob", stlBlob.size);

        loader.load(url, (geometry: any) => {
            console.log("BuildViewer: STL Loaded", geometry);
            // Center geometry
            geometry.center();
            geometry.computeVertexNormals();

            const material = new THREE.MeshPhongMaterial({
                color: 0x00ff00,
                specular: 0x111111,
                shininess: 200,
                side: THREE.DoubleSide
            });
            const mesh = new THREE.Mesh(geometry, material);

            // Fix orientation: build123d exports Z-up. Three.js is Y-up.
            // Rotating -90 deg (-PI/2) around X aligns Z to Y.
            mesh.rotation.x = -Math.PI / 2;

            scene.add(mesh);
            scene.add(new THREE.AxesHelper(50)); // Add axes to confirm orientation

            // Fit camera to object?
            const box = new THREE.Box3().setFromObject(mesh);
            const size = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());

            // Move camera back
            const maxDim = Math.max(size.x, size.y, size.z);
            const fov = camera.fov * (Math.PI / 180);
            let cameraZ = Math.abs(maxDim / 2 * Math.tan(fov * 2)); // rough extimate

            camera.position.set(center.x + maxDim, center.y + maxDim, center.z + maxDim);
            camera.lookAt(center);
            controls.target.copy(center);

        }, undefined, (error: any) => {
            console.error("BuildViewer Error:", error);
        });

        // Animation Loop
        const animate = () => {
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        };
        animate();

        // Handle Resize
        const handleResize = () => {
            if (!containerRef.current) return;
            const w = containerRef.current.clientWidth;
            const h = containerRef.current.clientHeight;
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
            renderer.setSize(w, h);
        };
        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            URL.revokeObjectURL(url);
            window.removeEventListener('resize', handleResize);
            if (containerRef.current && renderer.domElement) {
                containerRef.current.removeChild(renderer.domElement);
            }
            renderer.dispose();
        };
    }, [stlBlob]);

    return <div ref={containerRef} className="w-full h-full" />;
}
