<template>
    <AppMainLayout>
        <template #aside>
            <TheSideNavProvider />
        </template>

        <RouterView />
    </AppMainLayout>
</template>

<script setup lang="ts">
import { RouterView } from 'vue-router'
import AppMainLayout from './layout/AppMainLayout.vue';
import TheSideNavProvider from './components/side-nav/TheSideNavProvider.vue';

</script>
