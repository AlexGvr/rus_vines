// Мелкая чистка данных каталога при показе. Сами данные не трогаем:
// карточка должна совпадать с выгрузкой платформы.

// «Рубин кларет . Красная стрелка» — пробел перед знаком из выгрузки.
export function cleanTitle(title?: string): string {
  return (title || '').replace(/\s+([.,!?;:])/g, '$1').replace(/\s{2,}/g, ' ').trim()
}

const PAIRING = /\s*(?:гастрономические\s+)?сочетания\s*[:—-]\s*/iu

// У части вин описание заканчивается абзацем «Сочетания: …». Рядом уже есть
// блок «К чему подать», поэтому абзац переезжает туда, а не повторяется.
export function splitDescription(text?: string): { body: string; pairing: string } {
  if (!text) return { body: '', pairing: '' }
  const match = PAIRING.exec(text)
  if (!match) return { body: text, pairing: '' }
  return {
    body: text.slice(0, match.index).trim(),
    pairing: text.slice(match.index + match[0].length).trim(),
  }
}

export type WineStyle = { sweetness?: string | null; sparkling?: boolean | null }
export type WineItem = { wine: Record<string, any>; reasons?: string[] }
