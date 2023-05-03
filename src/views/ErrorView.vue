<template>
    <div class="flex flex-col items-center justify-center h-full">
        <ion-icon name="alert-circle-outline" class="text-6xl text-gray-700"></ion-icon>
        <div class="text-4xl font-bold text-gray-700" v-if="!errorMessage">404</div>
        <div class="text-2xl font-bold text-gray-700" v-if="!errorMessage">Page not found</div>
        <div class="text-2xl font-bold text-gray-700" v-else>{{ errorMessage }}</div>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import { useRouter } from 'vue-router';

const router = useRouter();

const errorMessage = computed(() => {
    const { params } = router.currentRoute.value;
    console.log(params);
    return params.message;
});
</script>
