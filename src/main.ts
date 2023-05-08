import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'
import '@spectrum-web-components/bundle/elements.js';
import 'ionicons/dist/ionicons.js';

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

// Motion does not work properly with chrome for some reason
app.use(MotionPlugin)

app.mount('#app')
