import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
    history: createWebHistory(import.meta.env.BASE_URL),
    routes: [
        {
            path: '/',
            name: 'home',
            component: () => import('../views/HomeView.vue')
        },
        {
            path: '/chat',
            name: 'chat',
            component: () => import('../views/ChatView.vue')
        },
        {
            path: '/error/:message',
            name: 'error',
            component: () => import('../views/ErrorView.vue')
        }
    ]
})

export default router
