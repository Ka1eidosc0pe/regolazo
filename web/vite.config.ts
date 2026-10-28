import { defineConfig } from "vitest/config";

// `?live=1` connects through these proxies to the live engine (same origin, so no CORS setup is needed).
// REGOLAZO_API picks the server (default http://127.0.0.1:8000).
const api = process.env.REGOLAZO_API ?? "http://127.0.0.1:8000";
const proxy = {
  "/api": { target: api, changeOrigin: true },
  "/ws": { target: api.replace(/^http/, "ws"), ws: true, changeOrigin: true },
};

export default defineConfig({
  server: { port: 5173, strictPort: true, proxy },
  preview: { proxy },
  test: { include: ["tests/**/*.test.ts"], environment: "node" },
});
