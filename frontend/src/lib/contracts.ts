export type OverlayLayer =
  | { type: 'band_ratio'; params: { top: number; bottom: number } }
  | { type: 'tangent_flags'; params: { points: [number, number][] } }
  | { type: 'signifier_hotspots'; params: { areas: { x: number; y: number; r: number }[] } }

export type OverlayDoc = {
  artifactType: 'sketch' | 'cad'
  layers: OverlayLayer[]
}
