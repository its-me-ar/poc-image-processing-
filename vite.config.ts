import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  server: {
    watch: {
      ignored: ['**/addons/**'],
    },
  },
  optimizeDeps: {
    exclude: ['addons'],
  },
  build: {
    rollupOptions: {
      external: (id: string) => id.startsWith('addons/') || id === 'addons',
    },
  },
})
