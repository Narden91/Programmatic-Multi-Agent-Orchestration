import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        configure: (proxy, options) => {
          proxy.on('error', (err, req, res) => {
            // Suppress verbose ECONNREFUSED terminal dumping during slow backend startup
            if (err.code !== 'ECONNREFUSED') {
              console.log('proxy error', err);
            }
          });
        }
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom'],
          'vendor-motion': ['framer-motion'],
          'vendor-charts': ['recharts'],
          'vendor-syntax': [
            'react-syntax-highlighter',
            'react-syntax-highlighter/dist/esm/styles/prism',
          ],
          'vendor-markdown': ['react-markdown', 'remark-gfm'],
          'vendor-icons': ['lucide-react'],
        },
      },
    },
  },
})
