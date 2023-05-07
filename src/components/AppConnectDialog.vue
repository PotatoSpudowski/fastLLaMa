<template>
    <sp-dialog v-if="show" class="w-[25rem] bg-zinc-900" no-divider dismissable @close="onClose">
        <div slot="heading" class="">
            <h2>Websocket Configuration</h2>
            <sp-status-light variant="positive" v-if="isConnected">Connected</sp-status-light>
            <sp-status-light variant="negative" v-else>Disconnected</sp-status-light>
        </div>
        <form onsubmit.prevent="">
            <div class="w-full">
                <sp-field-label for="websocket-address">Websocket Address</sp-field-label>
                <div class="flex items-center gap-2 justify-between">
                    <sp-textfield id="websocket-address" placeholder="ws://localhost:8080" class="mb-2 flex-grow"
                        :readonly="isConnecting" v-model="address" :invalid="hasError"></sp-textfield>
                    <sp-progress-circle label="Loading" indeterminate static="white" class="mb-2"
                        :style="{ opacity: isConnecting ? '1' : '0' }"></sp-progress-circle>
                </div>
                <sp-help-text variant="negative" icon v-show="hasError">
                    {{ errorMessage }}
                </sp-help-text>
            </div>

            <sp-button-group class="min-w-fit mt-2">
                <sp-button static="white" :disabled="isConnecting" v-if="!isConnected"
                    @click="onConnect">Connect</sp-button>
                <sp-button variant="negative" :disabled="isConnecting" v-else @click="disconnect">Disconnect</sp-button>
            </sp-button-group>
        </form>
        <AppLoading v-model="showLoading" message="Connecting to websocket..."></AppLoading>
    </sp-dialog>
</template>

<script setup lang="ts">
import type { WebSocketMessage } from '@/model/schema';
import useSocketStore from '@/stores/socketStore';
import { storeToRefs } from 'pinia';
import { ref } from 'vue';
import AppLoading from './AppLoading.vue';


interface Props {
    show: boolean,
}

interface Emits {
    (e: 'update:show', value: boolean): void,
}

defineProps<Props>();
const emits = defineEmits<Emits>();

const socketStore = useSocketStore();
const { isConnecting, isConnected, errorMessage, hasError, address } = storeToRefs(socketStore);
const { connect, disconnect } = useSocketStore();
const showLoading = ref(false);

function onClose() {
    emits('update:show', false);
}


// ----------------- Websocket -----------------

const initAckPromise = {
    resolve: (_value: boolean) => { void _value; },
    reject: (_reason?: any) => { void _reason; },
    timeout: 5000, // 5 seconds
};


function onWebsocketMessage(message: WebSocketMessage) {
    if (message.type === 'init-ack') {
        initAckPromise.resolve(true);
        socketStore.off(onWebsocketMessage);
    }
}

async function initHandshake() {
    if (!socketStore.socket) return false;
    socketStore.on(onWebsocketMessage);
    return new Promise<boolean>((resolve, reject) => {
        initAckPromise.resolve = resolve;
        initAckPromise.reject = reject;
        socketStore.message({
            type: 'init',
            version: '1'
        });
        setTimeout(() => {
            reject('websocket did not respond in time');
        }, initAckPromise.timeout);
    });
}

async function onConnect() {
    showLoading.value = true;
    connect(async (err) => {
        if (err) return;
        try {
            if (!await initHandshake()) {
                disconnect();
                hasError.value = true;
                errorMessage.value = 'Websocket does not follow fastLLaMa protocol';
            }
        } catch {
            disconnect();
            hasError.value = true;
            errorMessage.value = 'Websocket does not respond in time, or does not support fastLLaMa protocol';
        } finally {
            showLoading.value = false;
        }
    });

}

// ----------------- Websocket -----------------

</script>
