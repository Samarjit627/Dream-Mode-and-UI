import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
// @ts-ignore
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
// @ts-ignore
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export default function GltfCanvas({ url }: { url: string }) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene>()
  const cameraRef = useRef<THREE.PerspectiveCamera>()
  const rendererRef = useRef<THREE.WebGLRenderer>()
  const controlsRef = useRef<any>()
  const overlayGroupRef = useRef<THREE.Group | null>(null)
  const lastBBoxRef = useRef<THREE.Box3 | null>(null)

  useEffect(() => {
    const wrap = wrapRef.current!
    const width = wrap.clientWidth || 600
    const height = wrap.clientHeight || 400

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(width, height)
    renderer.setClearColor(0x000000, 0)
    // @ts-ignore
    ;(renderer as any).outputColorSpace = (THREE as any).SRGBColorSpace || (THREE as any).sRGBEncoding
    wrap.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const scene = new THREE.Scene()
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 1000)
    camera.position.set(0, 0, 3)
    cameraRef.current = camera

    scene.add(new THREE.AmbientLight(0xffffff, 0.8))
    const d = new THREE.DirectionalLight(0xffffff, 1.2)
    d.position.set(5, 5, 5)
    scene.add(d)

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controlsRef.current = controls

    const disposeObject = (obj: THREE.Object3D) => {
      obj.traverse((n: any) => {
        if (n.isMesh) {
          if (n.geometry) n.geometry.dispose()
          const mats = Array.isArray(n.material) ? n.material : [n.material]
          mats?.forEach((m: any) => m && m.dispose && m.dispose())
        }
      })
    }

    const fitCameraToBox = (box: THREE.Box3, offset = 1.4) => {
      const size = box.getSize(new THREE.Vector3())
      const center = box.getCenter(new THREE.Vector3())
      const maxSize = Math.max(size.x, size.y, size.z)
      const fitHeightDistance = maxSize / (2 * Math.tan((Math.PI * camera.fov) / 360))
      const distance = offset * fitHeightDistance
      const dir = camera.position.clone().sub(controls.target).normalize()
      controls.target.copy(center)
      camera.position.copy(center.clone().add(dir.multiplyScalar(distance)))
      camera.near = distance / 100
      camera.far = distance * 100
      camera.updateProjectionMatrix()
      controls.update()
    }

    const addOverlays = (bbox: THREE.Box3) => {
      if (overlayGroupRef.current) {
        scene.remove(overlayGroupRef.current)
        overlayGroupRef.current = null
      }
      const g = new THREE.Group()
      const { min, max } = bbox
      const midX = (min.x + max.x) / 2
      const midZ = (min.z + max.z) / 2
      // centerline
      const clGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(midX, min.y, midZ),
        new THREE.Vector3(midX, max.y, midZ)
      ])
      const cl = new THREE.Line(
        clGeom,
        new THREE.LineBasicMaterial({ color: 0x22c55e, depthTest: false, transparent: true, opacity: 1 })
      )
      cl.name = 'centerline'
      cl.renderOrder = 5
      g.add(cl)
      // beltline at 60%
      const y = min.y + 0.6 * (max.y - min.y)
      const blGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(min.x, y, midZ),
        new THREE.Vector3(max.x, y, midZ)
      ])
      const bl = new THREE.Line(
        blGeom,
        new THREE.LineBasicMaterial({ color: 0x10b981, depthTest: false, transparent: true, opacity: 0.95 })
      )
      bl.name = 'beltline'
      bl.renderOrder = 6
      g.add(bl)
      overlayGroupRef.current = g
      scene.add(g)
    }

    const loader = new GLTFLoader()
    loader.load(url, (gltf) => {
      const root = gltf.scene
      // add simple edges/points overlay per mesh
      root.traverse((child: any) => {
        if (child.isMesh) {
          const g: THREE.BufferGeometry = child.geometry
          try {
            const edges = new THREE.EdgesGeometry(g, 30)
            const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x16a34a, opacity: 0.8, transparent: true, depthTest: false }))
            line.renderOrder = 2
            child.add(line)
          } catch {}
        }
      })
      scene.add(root)
      const bbox = new THREE.Box3().setFromObject(root)
      lastBBoxRef.current = bbox
      addOverlays(bbox)
      fitCameraToBox(bbox)
    }, undefined, (err) => {
      console.warn('[glb] load error', err)
    })

    let raf = 0
    const tick = () => {
      controls.update()
      renderer.render(scene, camera)
      raf = requestAnimationFrame(tick)
    }
    tick()

    const onResize = () => {
      const w = wrap.clientWidth || 600
      const h = wrap.clientHeight || 400
      renderer.setSize(w, h)
      camera.aspect = w / h
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(onResize)
    ro.observe(wrap)

    const onBelt = (e: any) => {
      const yn = Math.min(1, Math.max(0, e?.detail?.y ?? 0.6))
      const grp = overlayGroupRef.current
      const bbox = lastBBoxRef.current
      if (!grp || !bbox) return
      const old = grp.getObjectByName('beltline') as THREE.Line
      if (old) grp.remove(old)
      const { min, max } = bbox
      const midZ = (min.z + max.z) / 2
      const y = min.y + yn * (max.y - min.y)
      const blGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(min.x, y, midZ),
        new THREE.Vector3(max.x, y, midZ)
      ])
      const bl = new THREE.Line(blGeom, new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.95 }))
      bl.name = 'beltline'
      bl.renderOrder = 7
      grp.add(bl)
    }

    const onOverlay = (e: any) => {
      const detail = e?.detail || {}
      const grp = overlayGroupRef.current
      if (!grp) return
      const cl = grp.getObjectByName('centerline') as THREE.Line
      const bl = grp.getObjectByName('beltline') as THREE.Line
      if (cl) cl.visible = !!detail.centerline
      if (bl) bl.visible = !!detail.band
    }

    window.addEventListener('axis5:beltline:set', onBelt as any)
    window.addEventListener('axis5:overlay:set', onOverlay as any)

    return () => {
      window.removeEventListener('axis5:beltline:set', onBelt as any)
      window.removeEventListener('axis5:overlay:set', onOverlay as any)
      ro.disconnect()
      controls.dispose()
      const sc = sceneRef.current
      if (sc) {
        sc.traverse((o: any) => { if (o.isMesh) { o.geometry?.dispose(); if (o.material?.dispose) o.material.dispose() } })
      }
      renderer.dispose()
      wrap.removeChild(renderer.domElement)
    }
  }, [url])

  return <div ref={wrapRef} style={{ width: '100%', height: '100%', position: 'relative' }} />
}
