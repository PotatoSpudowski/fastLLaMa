<template>
    <RouterView :key="$route.fullPath" />
</template>

<script setup lang="ts">
import { RouterView } from 'vue-router'

</script>
