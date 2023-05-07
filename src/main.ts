import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

import { MotionPlugin } from '@vueuse/motion'

import './assets/main.css'
import { notifyError } from './lib/notification'

const app = createApp(App);

app.config.errorHandler = (err, vm, info) => {
    console.log('errorHandler', err, vm, info)
    const name = vm?.$options.name ?? vm?.$options._componentTag ?? 'Unknown';
    const message = err instanceof Error ? err.message : String(err);
    notifyError(`Error[${name}]: ${message}; ${info}`);
}

app.use(createPinia())
app.use(router)
app.use(MotionPlugin)

app.mount('#app')
