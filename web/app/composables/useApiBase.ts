// Адрес API: из NUXT_PUBLIC_API_BASE, а если он не задан — тот же хост, что
// у страницы, порт 8080. Интерфейс, открытый с телефона по адресу компьютера
// в локальной сети, тогда ходит в API на том же компьютере без пересборки.
export function useApiBase(): string {
  const configured = useRuntimeConfig().public.apiBase as string
  if (configured) return configured
  return `${window.location.protocol}//${window.location.hostname}:8080`
}
