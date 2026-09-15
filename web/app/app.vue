<template>
  <div class="shell">
    <header class="masthead">
      <div class="brand">Своё&nbsp;Вино</div>
      <div class="section">Сканер этикеток</div>
    </header>
    <NuxtPage />
  </div>
</template>

<style>
/* Палитра и шрифт сняты со стилистики портала vino-svoe.ru:
   Playfair Display в заголовках, винный #bc6267 в акцентах. */
:root {
  --wine: #bc6267;
  --wine-deep: #8f3d42;
  --wine-darkest: #723135;
  --ink: #2c2a28;
  --paper: #fefdfa;
  --line: #eaeaea;
  --muted: #9a948f;
  --display: 'Playfair Display', Georgia, 'Times New Roman', serif;
  --sans: system-ui, -apple-system, 'Segoe UI', sans-serif;
  --radius: 14px;
}
* { box-sizing: border-box; }
html, body, #__nuxt { height: 100%; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 16px;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
.shell {
  max-width: 520px;
  margin: 0 auto;
  min-height: 100dvh;
  display: flex;
  flex-direction: column;
  padding: 0 18px 40px;
}
.masthead {
  padding: 20px 0 16px;
  border-bottom: 1px solid var(--line);
  margin-bottom: 22px;
}
.brand {
  font-family: var(--display);
  font-size: 26px;
  font-weight: 600;
  letter-spacing: 0.01em;
  color: var(--wine-darkest);
}
.section {
  font-size: 13px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--muted);
  margin-top: 2px;
}
h1, h2, h3 { font-family: var(--display); font-weight: 600; margin: 0; }
button { font-family: inherit; }
:focus-visible { outline: 2px solid var(--wine); outline-offset: 2px; border-radius: 6px; }

.btn {
  border: 0;
  border-radius: 999px;
  padding: 14px 26px;
  font-size: 16px;
  font-weight: 600;
  background: var(--wine);
  color: #fff;
  cursor: pointer;
  transition: background .15s;
}
.btn:hover { background: var(--wine-deep); }
.btn:active { background: var(--wine-darkest); }
.btn.ghost {
  background: transparent;
  color: var(--wine-deep);
  border: 1.5px solid var(--wine);
}
.btn.ghost:hover { background: rgba(188, 98, 103, .08); }

.chip {
  display: inline-block;
  font-size: 13px;
  padding: 5px 12px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: #fff;
  color: var(--ink);
}
.chips { display: flex; flex-wrap: wrap; gap: 7px; }
.muted { color: var(--muted); }
</style>
