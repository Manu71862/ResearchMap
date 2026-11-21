import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: false,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('error', (err, _req, res) => {
            if (res && !res.headersSent) {
              res.writeHead(503, { 'Content-Type': 'application/json' });
              res.end(
                JSON.stringify({
                  detail: `Backend verification service is unreachable at http://127.0.0.1:8000 (${err.code || 'ECONNREFUSED'}). Ensure FastAPI is running on port 8000.`,
                })
              );
            }
          });
        },
      },
    },
  },
});
