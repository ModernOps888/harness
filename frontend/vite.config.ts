import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/v1': {
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
      },
      '/metrics': {
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
      },
      '/proof': {
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
      },
      '/report': {
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
      },
    },
  },
});
