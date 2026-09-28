import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    chunkSizeWarningLimit: 550,
  },
  optimizeDeps: {
    include: ["echarts/core", "echarts/charts", "echarts/components", "echarts/renderers"],
  },
  server: {
    allowedHosts: [".myide.io"],
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
