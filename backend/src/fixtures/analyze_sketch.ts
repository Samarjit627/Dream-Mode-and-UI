export default {
  type: 'sketch-analysis',
  metrics: {
    edgeDensity: 0.42,
    dominantAngleDeg: 0
  },
  layers: [
    { type: 'band_ratio' },
    { type: 'guide_lines' },
    { type: 'signifier_hotspots' },
    { type: 'tangent_flags' }
  ],
  insights: [
    { key: 'symmetry', value: 'bilateral' },
    { key: 'primary-shape', value: 'bottle-profile' }
  ]
}
