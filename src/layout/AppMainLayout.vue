<template>
    <sp-theme theme="spectrum" color="darkest" scale="medium" class="h-full">
        <div class="w-full h-full" style="display: grid;"
            :style="{ 'grid-template-columns': isWebsocketConnected ? '15rem 1fr' : '1fr' }">
            <aside v-if="$slots.aside && isWebsocketConnected" class="bg-zinc-900 h-full overflow-hidden">
                <slot name="aside"></slot>
            </aside>
            <main class="h-full overflow-hidden">
                <slot></slot>
            </main>
        </div>
    </sp-theme>
</template>

<script setup lang="ts">
import useSocketStore from '@/stores/socketStore';
import { storeToRefs } from 'pinia';


const { isConnected: isWebsocketConnected } = storeToRefs(useSocketStore())
</script>
