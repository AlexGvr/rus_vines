// Недавно найденные вина — на главном экране, чтобы вернуться к карточке
// без повторного сканирования. Живёт в браузере пользователя; если
// хранилище недоступно (приватный режим), список просто пустой.
type Recent = {
  slug: string
  title: string
  manufacturer?: string
  region?: string
  rating?: number
  photo?: string
}

const KEY = 'svoe-vino-scanner:recent'
const LIMIT = 5

function load(): Recent[] {
  try {
    const items = JSON.parse(localStorage.getItem(KEY) || '[]')
    return Array.isArray(items) ? items : []
  } catch {
    return []
  }
}

export function useHistory() {
  const items = useState<Recent[]>('scan-history', load)

  function add(wine: Record<string, any>) {
    const entry: Recent = {
      slug: wine.slug,
      title: wine.title,
      manufacturer: wine.manufacturer,
      region: wine.region,
      rating: wine.rating,
      photo: wine.photo,
    }
    items.value = [entry, ...items.value.filter((i) => i.slug !== entry.slug)].slice(0, LIMIT)
    try {
      localStorage.setItem(KEY, JSON.stringify(items.value))
    } catch {
      // без хранилища история живёт до перезагрузки
    }
  }

  return { items, add }
}
