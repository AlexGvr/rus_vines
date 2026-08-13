const VERSION = "v1";
const SHELL = `shell-${VERSION}`;
const PHOTOS = `photos-${VERSION}`;

const PRECACHE = [
  ".",
  "index.html",
  "manifest.webmanifest",
  "icon.svg",
  "css/style.css",
  "js/app.js",
  "js/scan.js",
  "js/matcher.js",
  "js/normalize.js",
  "data/wines.json",
  "vendor/tesseract/tesseract.min.js",
  "vendor/tesseract/worker.min.js",
  "vendor/tesseract/tesseract-core-simd-lstm.wasm.js",
  "vendor/tesseract/tesseract-core-lstm.wasm.js",
  "vendor/tesseract/lang/rus.traineddata.gz",
  "vendor/tesseract/lang/eng.traineddata.gz",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => ![SHELL, PHOTOS].includes(k)).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== location.origin) return;
  if (url.pathname.endsWith("/data/wines.json")) {
    // network-first: при сети подтягиваем свежий датапак, офлайн — кэш
    e.respondWith(
      fetch(e.request).then((resp) => {
        if (resp.ok) {
          const copy = resp.clone();
          caches.open(SHELL).then((c) => c.put(e.request, copy));
        }
        return resp;
      }).catch(() => caches.match(e.request))
    );
    return;
  }
  if (url.pathname.includes("/photos/")) {
    e.respondWith(
      caches.open(PHOTOS).then(async (c) => {
        const hit = await c.match(e.request);
        if (hit) return hit;
        const resp = await fetch(e.request);
        if (resp.ok) c.put(e.request, resp.clone());
        return resp;
      })
    );
    return;
  }
  e.respondWith(
    caches.match(e.request, { ignoreSearch: true }).then((hit) => hit || fetch(e.request))
  );
});
