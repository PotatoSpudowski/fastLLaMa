<template>
    <sp-dialog v-if="show" class="w-[25rem] bg-zinc-900" no-divider dismissable @close="onClose">
        <div slot="heading" class="">
            <h2>Websocket Configuration</h2>
            <sp-status-light variant="positive" v-if="isConnected">Connected</sp-status-light>
            <sp-status-light variant="negative" v-else>Disconnected</sp-status-light>
        </div>
        <form>
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
                <sp-button static="white" :disabled="isConnecting" v-if="!isConnected" @click="connect">Connect</sp-button>
                <sp-button variant="negative" :disabled="isConnecting" v-else @click="disconnect">Disconnect</sp-button>
            </sp-button-group>
        </form>
    </sp-dialog>
</template>

<script setup lang="ts">
import useSocketStore from '@/stores/socketStore';
import { storeToRefs } from 'pinia';


interface Props {
    show: boolean,
}

interface Emits {
    (e: 'update:show', value: boolean): void,
}

const props = defineProps<Props>();
const emits = defineEmits<Emits>();

const socketStore = useSocketStore();
const { isConnecting, isConnected, errorMessage, hasError, address } = storeToRefs(socketStore);
const { connect, disconnect } = useSocketStore();

function onClose() {
    emits('update:show', false);
}

</script>
