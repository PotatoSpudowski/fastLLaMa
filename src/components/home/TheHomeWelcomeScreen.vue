<template>
    <div class="flex flex-col items-center justify-center">
        <img :src="LlamaBg" class="h-[30rem] aspect-[1.66/1]" />
        <sp-button-group class="min-w-fit my-2">
            <sp-button static="white" @click="emits('select-model')" v-if="isConnectedToWebsocket">Select Model</sp-button>
            <sp-button static="white" @click="emits('connect-websocket')" v-else>Connect</sp-button>
        </sp-button-group>
    </div>
</template>

<script setup lang="ts">
import useSocketStore from '@/stores/socketStore';
import { storeToRefs } from 'pinia';
import LlamaBg from '@/assets/img/image.png';

interface Emits {
    (e: 'select-model'): void
    (e: 'connect-websocket'): void
}

const emits = defineEmits<Emits>()

// ---------------- Websocket ----------------

const socketStore = useSocketStore();
const { isConnected: isConnectedToWebsocket } = storeToRefs(socketStore);

// ---------------- Websocket ----------------

</script>
