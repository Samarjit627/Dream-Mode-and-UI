export default {
  cards: [
    {
      id: 'beltline_harmony_041',
      track: 'form_readability',
      title: 'Beltline Harmony (0.38–0.42 H)',
      why: 'Centers the primary read to match hand placement and visual weight.',
      overlay: { type: 'band_ratio', suggest: [0.40, 0.42] },
      actions: [ { label: 'Set 0.41', params: { belt_ratio: 0.41 } } ],
      refs: ['Ching','Robertson']
    },
    {
      id: 'fillet_radii_guideline',
      track: 'detail',
      title: 'Fillet Radii 6–10mm',
      why: 'Balances mold flow and tactile comfort.',
      overlay: { type: 'tangent_flags' },
      actions: [ { label: 'Prefer R6–R10', params: { fillet_min: 6, fillet_max: 10 } } ],
      refs: ['Baxter']
    },
  ],
}
