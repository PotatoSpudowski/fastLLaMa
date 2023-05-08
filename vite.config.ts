import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig((config) => {
    const { mode } = config;
    const isDev = mode === 'development';
    return {
        plugins: [vue({
            template: {
                compilerOptions: {
                    isCustomElement: tag => tag.startsWith('sp-') || tag.startsWith('ion-')
                }
            }
        })],
        resolve: {
            alias: {
                '@': fileURLToPath(new URL('./src', import.meta.url))
            }
        },
        base: isDev ? '/' : '/fastLLaMa/'
    }
})
