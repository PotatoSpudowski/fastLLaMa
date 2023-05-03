import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'

import '@spectrum-web-components/theme/sp-theme.js';
import '@spectrum-web-components/theme/src/themes.js';
import '@spectrum-web-components/sidenav/sp-sidenav.js';
import '@spectrum-web-components/sidenav/sp-sidenav-heading.js';
import '@spectrum-web-components/sidenav/sp-sidenav-item.js';
import '@spectrum-web-components/button/sp-button.js';
import '@spectrum-web-components/button/sp-clear-button.js';
import '@spectrum-web-components/button/sp-close-button.js';
import '@spectrum-web-components/button-group/sp-button-group.js';
import '@spectrum-web-components/dialog/sp-dialog.js';
import '@spectrum-web-components/textfield/sp-textfield.js';
import '@spectrum-web-components/field-label/sp-field-label.js';
import '@spectrum-web-components/search/sp-search.js';
import '@spectrum-web-components/menu/sp-menu.js';
import '@spectrum-web-components/menu/sp-menu-group.js';
import '@spectrum-web-components/menu/sp-menu-item.js';
import '@spectrum-web-components/menu/sp-menu-divider.js';
import '@spectrum-web-components/icon/sp-icon.js';
import '@spectrum-web-components/progress-circle/sp-progress-circle.js';
// import '@spectrum-web-components/table/elements.js';

import { MotionPlugin } from '@vueuse/motion'

import './assets/main.css'

const app = createApp(App);

app.use(createPinia())
app.use(router)
app.use(MotionPlugin)

app.mount('#app')
