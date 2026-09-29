export default defineNuxtConfig({
  compatibilityDate: '2026-09-01',
  devtools: { enabled: false },
  // Порт из README; 3000 по умолчанию часто занят.
  devServer: { port: 3300 },
  // Модуль встраивается в каталог портала, поэтому рендерится на клиенте:
  // страница живёт внутри чужой оболочки и своего SSR не требует.
  ssr: false,
  runtimeConfig: {
    public: {
      // Пусто — тот же хост, что у страницы, порт 8080 (composables/useApiBase).
      apiBase: process.env.NUXT_PUBLIC_API_BASE || '',
    },
  },
  app: {
    head: {
      title: 'Сканер вин — Своё Вино',
      meta: [
        { name: 'viewport', content: 'width=device-width, initial-scale=1, viewport-fit=cover' },
        { name: 'theme-color', content: '#bc6267' },
      ],
      link: [
        { rel: 'preconnect', href: 'https://fonts.googleapis.com' },
        { rel: 'preconnect', href: 'https://fonts.gstatic.com', crossorigin: '' },
        {
          rel: 'stylesheet',
          href: 'https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700&display=swap',
        },
      ],
    },
  },
})
