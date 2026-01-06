export type StylePack = {
  id: string
  finishes?: string[]
  detail_cadence?: string
  texture?: { touch?: string }
  bias_rules?: string[]
}

export type KnowledgeCard = {
  id: string
  track?: string
  title: string
  why?: string
  overlay?: { type: string; suggest?: number[] }
  actions?: Array<{ label: string; params: Record<string, any> }>
  refs?: string[]
}
