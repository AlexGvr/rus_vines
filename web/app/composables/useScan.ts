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
  quality?: { f1_top1: number; f1_top5: number; subset: string; n: number; source?: string }
  latency_ms: { cv_ms: number; rerank_ms: number; total: number }
}

export type WineDetails = {
  wine: Record<string, any>
  style?: WineStyle
  pairings: string[]
  similar: WineItem[]
}

// Ответ /v1/analogs: что прочитано на этикетке и что из каталога похоже.
export type LabelAnalogs = {
  label: {
    producer: string | null
    producer_ru?: string | null
    name: string | null
    color: string | null
    sweetness: string | null
    sparkling: boolean | null
    grapes: string[]
    country: string | null
  }
  maker: string | null
  maker_wines: WineItem[]
  analogs: WineItem[]
}

// Результат сканирования живёт в общем состоянии, а не в странице: с
// результата можно уйти на карточку похожего вина и вернуться назад,
// не сканируя этикетку заново.
export function useScan() {
  return {
    preview: useState<string | undefined>('scan-preview', () => undefined),
    // Сам снимок: по нему после отказа читаются стиль и аналоги.
    file: useState<File | undefined>('scan-file', () => undefined),
    result: useState<SearchResult | undefined>('scan-result', () => undefined),
    details: useState<WineDetails | undefined>('scan-details', () => undefined),
    analogs: useState<LabelAnalogs | 'loading' | 'failed' | undefined>(
      'scan-analogs', () => undefined),
  }
}
