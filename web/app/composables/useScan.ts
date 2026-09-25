export type SearchResult = {
  status: 'confident' | 'uncertain' | 'unsure'
  match: Record<string, any> | null
  alternatives: Record<string, any>[]
  confidence: {
    probability: number
    probability_top5: number
    dominance: number
    inliers: number
    gap: number
    cv_margin: number
  }
  quality?: { f1_top1: number; f1_top5: number; subset: string; n: number }
  latency_ms: { cv_ms: number; rerank_ms: number; total: number }
}

// Результат сканирования живёт в общем состоянии, а не в странице: с
// результата можно уйти на карточку похожего вина и вернуться назад,
// не сканируя этикетку заново.
export function useScan() {
  return {
    preview: useState<string | undefined>('scan-preview', () => undefined),
    result: useState<SearchResult | undefined>('scan-result', () => undefined),
    details: useState<{ pairings: string[]; similar: any[] } | undefined>(
      'scan-details', () => undefined),
  }
}
