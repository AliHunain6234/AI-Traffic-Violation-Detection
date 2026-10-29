import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// In dev, /api (REST, MJPEG, WebSocket) is proxied to the FastAPI server.
const API = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { "@": path.resolve(__dirname, "src") } },
  server: {
    host: true,
    port: 5173,
    allowedHosts: true, // lets tunnels (ngrok / cloudflared) reach the dev server
    proxy: { "/api": { target: API, changeOrigin: true, ws: true } },
  },
  build: { outDir: "dist", sourcemap: false, chunkSizeWarningLimit: 900 },
});
