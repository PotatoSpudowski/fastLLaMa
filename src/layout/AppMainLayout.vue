<template>
    <sp-theme theme="spectrum" color="darkest" scale="medium" class="h-full">
        <div class="w-full h-full" style="display: flex;">
            <aside v-if="$slots.aside" class="bg-zinc-900 h-full overflow-hidden"
                :style="{ 'flex-basis': isWebsocketConnected ? '15rem' : '0', 'max-width': isWebsocketConnected ? '15rem' : '0' }"
                style="transition: flex-basis 150ms ease-in-out;">
                <slot name="aside"></slot>
            </aside>
            <main class="h-full overflow-hidden flex-grow">
                <slot></slot>
            </main>
        </div>
        <div role="region" id="alert-container" class="fixed z-10 top-0 left-1/2 -translate-x-1/2"></div>
    </sp-theme>
</template>

<script setup lang="ts">
import useSocketStore from '@/stores/socketStore';
import { storeToRefs } from 'pinia';


const { isConnected: isWebsocketConnected } = storeToRefs(useSocketStore())
</script>
