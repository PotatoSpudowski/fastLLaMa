<template>
    <sp-theme
        theme="spectrum"
        color="darkest"
        scale="medium"
        class="h-full">
    <div class="w-full h-full" style="display: grid; grid-template-columns: 15rem 1fr;">
        <aside v-if="$slots.aside" class="bg-zinc-900">
            <slot name="aside"></slot>
        </aside>
        <main>
            <slot></slot>
        </main>
    </div>
    </sp-theme>
</template>

<script setup lang="ts">

</script>
