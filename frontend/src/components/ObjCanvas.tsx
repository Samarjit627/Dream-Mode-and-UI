import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
// @ts-ignore: shipped with three
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js'
// @ts-ignore: shipped with three
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
// @ts-ignore: shipped with three
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js'

export default function ObjCanvas({ url, file, mtlUrl, mtlFile }: { url?: string, file?: File, mtlUrl?: string, mtlFile?: File }){
  const wrapRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<THREE.WebGLRenderer>()
  const sceneRef = useRef<THREE.Scene>()
  const cameraRef = useRef<THREE.PerspectiveCamera>()
  const controlsRef = useRef<any>()
  const msgRef = useRef<HTMLDivElement>(null)
  const currentObjRef = useRef<THREE.Object3D | null>(null)
  const helperRef = useRef<THREE.Box3Helper | null>(null)
  const overlayGroupRef = useRef<THREE.Group | null>(null)
  const lastBBoxRef = useRef<THREE.Box3 | null>(null)

  useEffect(()=>{
    const wrap = wrapRef.current!
    const width = wrap.clientWidth || 600
    const height = wrap.clientHeight || 400

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(width, height)
    renderer.setClearColor(0x000000, 0)
    // @ts-ignore - modern three
    ;(renderer as any).outputColorSpace = (THREE as any).SRGBColorSpace || (THREE as any).sRGBEncoding
    wrap.appendChild(renderer.domElement)
    rendererRef.current = renderer

    const scene = new THREE.Scene()
    scene.background = null
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(45, width/height, 0.01, 1000)
    camera.position.set(0, 0, 3)
    cameraRef.current = camera

    const light1 = new THREE.DirectionalLight(0xffffff, 1.2)
    light1.position.set(5,5,5)
    scene.add(light1)
    scene.add(new THREE.AmbientLight(0xffffff, 0.8))

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

    const fitCameraToBox = (camera: THREE.PerspectiveCamera, box: THREE.Box3, offset = 1.4) => {
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

    const loader = new OBJLoader()
    const handleObject = (obj: THREE.Object3D) => {
      // Remove previous model
      if (currentObjRef.current) {
        scene.remove(currentObjRef.current)
        disposeObject(currentObjRef.current)
        currentObjRef.current = null
      }
      if (helperRef.current) {
        scene.remove(helperRef.current)
        helperRef.current = null
      }

      // compute bbox
      let bbox = new THREE.Box3().setFromObject(obj)
      const size = bbox.getSize(new THREE.Vector3())
      const center = bbox.getCenter(new THREE.Vector3())
      obj.position.sub(center)
      const maxDim = Math.max(size.x, size.y, size.z)
      const scale = maxDim > 0 ? 1.8 / maxDim : 1
      obj.scale.setScalar(scale)
      let meshCount = 0, lineCount = 0, pointCount = 0, vertSum = 0
      obj.traverse((child: any)=>{
        if (child.isMesh) {
          // Material
          child.material = new THREE.MeshNormalMaterial({ side: THREE.DoubleSide })
          // Geometry normals
          const g: THREE.BufferGeometry = child.geometry
          if (!g.getAttribute('normal')) {
            g.computeVertexNormals()
          }
          if (!g.boundingBox) g.computeBoundingBox()
          if (!g.boundingSphere) g.computeBoundingSphere()
          // Edges overlay for visibility
          try {
            const edges = new THREE.EdgesGeometry(g, 30)
            const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x16a34a, opacity: 0.8, transparent: true, depthTest: false }))
            line.renderOrder = 2
            child.add(line)
            const pts = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x22c55e, size: 0.03, sizeAttenuation: true }))
            pts.renderOrder = 3
            child.add(pts)
          } catch {}
          child.castShadow = false
          child.receiveShadow = false
          child.frustumCulled = false
          const pos = g.getAttribute('position')
          vertSum += pos ? pos.count : 0
          meshCount++
        } else if (child.isLine) {
          child.material = new THREE.LineBasicMaterial({ color: 0x16a34a })
          const pos = child.geometry?.getAttribute?.('position')
          vertSum += pos ? pos.count : 0
          lineCount++
        } else if (child.isPoints) {
          child.material = new THREE.PointsMaterial({ color: 0x16a34a, size: 0.01 })
          const pos = child.geometry?.getAttribute?.('position')
          vertSum += pos ? pos.count : 0
          pointCount++
        }
      })
      scene.add(obj)
      currentObjRef.current = obj
      // recompute bbox after transform
      bbox = new THREE.Box3().setFromObject(obj)
      lastBBoxRef.current = bbox
      const helper = new THREE.Box3Helper(bbox, 0x16a34a)
      scene.add(helper)
      helperRef.current = helper
      fitCameraToBox(camera, bbox)
      if (msgRef.current) msgRef.current.textContent = (meshCount+lineCount+pointCount) > 0
        ? `${meshCount} mesh, ${lineCount} line, ${pointCount} point; ${vertSum} verts`
        : 'OBJ had no drawable primitives'

      // overlays: centerline + beltline band
      if (overlayGroupRef.current) {
        scene.remove(overlayGroupRef.current)
        overlayGroupRef.current = null
      }
      const g = new THREE.Group()
      const { min, max } = bbox
      const midX = (min.x + max.x) / 2
      const midZ = (min.z + max.z) / 2
      // centerline (vertical)
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
      // beltline (horizontal at 60%)
      const makeBelt = (yn: number) => {
        const y = min.y + yn * (max.y - min.y)
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
        return bl
      }
      g.add(makeBelt(0.6))

      // hotspots (three small spheres at interesting bands)
      const hotspots = new THREE.Group()
      hotspots.name = 'hotspots'
      const rad = (max.x - min.x) / 2
      const yVals = [0.25, 0.5, 0.75].map(t => min.y + t * (max.y - min.y))
      const sx = 0.35 * rad
      const geomS = new THREE.SphereGeometry(0.03 * (max.y - min.y), 16, 12)
      const matS = new THREE.MeshBasicMaterial({ color: 0xf59e0b })
      yVals.forEach((y) => {
        const s1 = new THREE.Mesh(geomS, matS)
        s1.position.set(midX - sx, y, midZ)
        const s2 = new THREE.Mesh(geomS, matS)
        s2.position.set(midX + sx, y, midZ)
        hotspots.add(s1, s2)
      })
      hotspots.visible = false
      g.add(hotspots)

      // tangent flags (arrows near shoulder area, both sides)
      const tangents = new THREE.Group()
      tangents.name = 'tangents'
      const yT = min.y + 0.82 * (max.y - min.y)
      const len = 0.25 * rad
      const originL = new THREE.Vector3(midX - rad * 0.9, yT, midZ)
      const originR = new THREE.Vector3(midX + rad * 0.9, yT, midZ)
      tangents.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0).normalize(), originL, len, 0x38bdf8))
      tangents.add(new THREE.ArrowHelper(new THREE.Vector3(-1, 0, 0).normalize(), originR, len, 0x38bdf8))
      tangents.visible = false
      g.add(tangents)

      overlayGroupRef.current = g
      scene.add(g)
    }

    const loadWithMaterialsThenObj = (doLoadObj: (materials?: any) => void) => {
      if (mtlFile) {
        const frM = new FileReader()
        frM.onload = () => {
          try {
            const text = String(frM.result || '')
            const mats = new MTLLoader().parse(text, '')
            mats.preload()
            loader.setMaterials(mats)
            doLoadObj(mats)
          } catch (err) {
            console.warn('[mtl] parse error', err)
            doLoadObj(undefined)
          }
        }
        frM.onerror = () => { console.warn('[mtl] filereader error', frM.error); doLoadObj(undefined) }
        frM.readAsText(mtlFile)
        return
      }
      if (mtlUrl) {
        new MTLLoader().load(mtlUrl, (mats: any) => { mats.preload(); loader.setMaterials(mats); doLoadObj(mats) }, undefined, () => doLoadObj(undefined))
        return
      }
      doLoadObj(undefined)
    }

    const loadFromFile = () => {
      const fr = new FileReader()
      fr.onload = () => {
        try {
          const text = String(fr.result || '')
          const parseAndHandle = () => {
            const obj = loader.parse(text)
            handleObject(obj)
          }
          loadWithMaterialsThenObj(() => parseAndHandle())
        } catch (err) {
          console.error('[obj] parse error', err)
          if (msgRef.current) msgRef.current.textContent = 'Failed to parse OBJ'
        }
      }
      fr.onerror = () => {
        console.error('[obj] filereader error', fr.error)
        if (msgRef.current) msgRef.current.textContent = 'Failed to read OBJ file'
      }
      fr.readAsText(file as File)
    }

    const loadFromUrl = () => {
      loadWithMaterialsThenObj(() => {
        loader.load(url as string, handleObject, undefined, (err: any)=>{
          console.error('[obj] load error', err)
          if (msgRef.current) msgRef.current.textContent = 'Failed to load OBJ'
        })
      })
    }

    if (file) loadFromFile()
    else if (url) loadFromUrl()
    else {
      if (msgRef.current) msgRef.current.textContent = 'No OBJ provided'
    }

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
      camera.aspect = w/h
      camera.updateProjectionMatrix()
    }
    const ro = new ResizeObserver(onResize)
    ro.observe(wrap)

    return ()=>{
      cancelAnimationFrame(raf)
      ro.disconnect()
      controls.dispose()
      if (currentObjRef.current) {
        disposeObject(currentObjRef.current)
        currentObjRef.current = null
      }
      if (helperRef.current) {
        scene.remove(helperRef.current)
        helperRef.current = null
      }
      renderer.dispose()
      wrap.removeChild(renderer.domElement)
    }
  },[url, file, mtlUrl, mtlFile])

  // listen for beltline updates
  useEffect(() => {
    const onBelt = (e: any) => {
      const yn = Math.min(1, Math.max(0, e?.detail?.y ?? 0.6))
      const scene = sceneRef.current
      const grp = overlayGroupRef.current
      const bbox = lastBBoxRef.current
      if (!scene || !grp || !bbox) return
      const old = grp.getObjectByName('beltline') as THREE.Line
      if (old) grp.remove(old)
      const { min, max } = bbox
      const midZ = (min.z + max.z) / 2
      const y = min.y + yn * (max.y - min.y)
      const blGeom = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(min.x, y, midZ),
        new THREE.Vector3(max.x, y, midZ)
      ])
      const bl = new THREE.Line(
        blGeom,
        new THREE.LineBasicMaterial({ color: 0x10b981, depthTest: false, transparent: true, opacity: 0.95 })
      )
      bl.name = 'beltline'
      bl.renderOrder = 7
      grp.add(bl)
    }
    window.addEventListener('axis5:beltline:set', onBelt as any)
    return () => window.removeEventListener('axis5:beltline:set', onBelt as any)
  }, [])

  // listen for overlay toggles
  useEffect(() => {
    const onOverlay = (e: any) => {
      const detail = e?.detail || {}
      const grp = overlayGroupRef.current
      if (!grp) return
      const cl = grp.getObjectByName('centerline') as THREE.Line
      const bl = grp.getObjectByName('beltline') as THREE.Line
      const hs = grp.getObjectByName('hotspots') as THREE.Group
      const tg = grp.getObjectByName('tangents') as THREE.Group
      if (cl) cl.visible = !!detail.centerline
      if (bl) bl.visible = !!detail.band
      if (hs) hs.visible = !!detail.hotspots
      if (tg) tg.visible = !!detail.tangents
    }
    window.addEventListener('axis5:overlay:set', onOverlay as any)
    return () => window.removeEventListener('axis5:overlay:set', onOverlay as any)
  }, [])

  return (
    <div ref={wrapRef} style={{ width: '100%', height: '100%', position:'relative' }}>
      <div ref={msgRef} style={{position:'absolute', bottom:8, right:12, fontSize:12, color:'var(--muted)'}} />
    </div>
  )
}
