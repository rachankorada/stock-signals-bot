import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the site runs on :5173 and forwards API calls to the backend on :8000
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
