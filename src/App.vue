<template>
    <AppMainLayout>
        <template #aside>
            Test
        </template>

        <RouterView />
    </AppMainLayout>
</template>

<script setup lang="ts">
import { RouterView } from 'vue-router'
import AppMainLayout from './layout/AppMainLayout.vue';

</script>
