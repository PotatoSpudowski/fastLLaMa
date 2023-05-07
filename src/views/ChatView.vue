<template>
    <AppMainLayout>
        <template #aside>
            <TheSideNavProvider />
        </template>
        <TheChatProvider v-if="filepath" ref="chatProviderRef">
            <template v-for="message in messages" :key="message.id">
                <TheChatSystemMessage v-if="message.type === 'system-message'" :message="message" />
                <TheChatUserMessage v-else-if="message.type === 'user-message'" :message="message" />
                <TheChatModelMessage v-else-if="message.type === 'model-message'" :message="message" />
            </template>
        </TheChatProvider>
    </AppMainLayout>
</template>

<script setup lang="ts">
import { computed, onBeforeMount, onMounted, ref } from 'vue';
import TheChatModelMessage from '@/components/chat/TheChatModelMessage.vue';
import TheChatProvider from '@/components/chat/TheChatProvider.vue';
import TheChatSystemMessage from '@/components/chat/TheChatSystemMessage.vue';
import TheChatUserMessage from '@/components/chat/TheChatUserMessage.vue';
import AppMainLayout from '@/layout/AppMainLayout.vue';
import TheSideNavProvider from '@/components/side-nav/TheSideNavProvider.vue';
import { dummyMessages } from '@/model/dummy';
import { useRouter } from 'vue-router';
import type { ConversationMessage, Message, SystemMessageProgress, WebSocketMessage } from '@/model/schema';
import { pause } from '@/lib/utils';
import useSocketStore from '@/stores/socketStore';

const messages = ref<Message[]>([]);
const router = useRouter();
const chatProviderRef = ref<InstanceType<typeof TheChatProvider> | null>(null);
const messagesKey = new Set<string>();

const filepath = computed(() => {
    const { query } = router.currentRoute.value;
    const temp = Array.isArray(query.path) ? query.path[0] : query.path;
    if (!temp) return null;
    return String(temp);
});

onBeforeMount(() => {
    if (filepath.value) return;
    router.replace({ name: 'error', params: { message: 'Model path is not specified' } });
});

function addMessage(message: Message) {
    if (messagesKey.has(message.id)) return;
    messagesKey.add(message.id);
    messages.value.push(message);
}

function searchMessageFromEnd(id: string) {
    for (let i = messages.value.length - 1; i >= 0; --i) {
        if (messages.value[i].id === id) return i;
    }
    return -1;
}

function updateMessage(message: Message) {
    if (!messagesKey.has(message.id)) {
        addMessage(message);
        return;
    }
    const index = searchMessageFromEnd(message.id);
    messages.value[index] = message;
}

// ----------------- Model Params -----------------

const websocketStore = useSocketStore();

const modelParams = computed(() => {
    const { query } = router.currentRoute.value;
    const temp = Array.isArray(query.model_params) ? query.model_params[0] : query.model_params;
    if (!temp) return null;
    return JSON.parse(String(temp));
});

function onError(message: string) {
    // TODO: Show error message
}

function onWebsocketMessage(message: WebSocketMessage) {
    switch (message.type) {
        case 'model-message': case 'system-message': case 'user-message': {
            updateMessage(message);
            break;
        }
        case 'message-ack': {
            const temp = searchMessageFromEnd(message.webui_id);
            if (temp === -1) return;
            const mess = messages.value[temp];
            if (mess.type !== 'user-message') return;
            mess.status = {
                kind: message.status,
            };
            mess.id = message.id;
            break;
        }
        default: return;
    }
}

onBeforeMount(async () => {
    if (!websocketStore.isConnected) {
        await router.replace({ name: 'error', params: { message: 'Websocket is not connected' } });
    } else {
        if (modelParams.value) return;
        await router.replace({ name: 'error', params: { message: 'Model params is not specified' } });
    }
});

onMounted(() => {
    if (!modelParams.value) return;
    websocketStore.on(onWebsocketMessage)
    websocketStore.message({
        type: 'init-model',
        model_path: filepath.value!,
        ...modelParams.value,
    });
});

// ----------------- Model Params -----------------

// ----------------- Testing -----------------

async function simulateProgress(callback: (progress: number) => void) {
    for (let i = 0; i < 100; i += 10) {
        callback(i);
        await pause(100 + Math.floor(Math.random() * 200))
    }
    callback(100);
}

async function simulateMessage() {
    for (let i = 0; i < dummyMessages.length; ++i) {
        const message = dummyMessages[i];
        if (message.type === 'system-message' && message.kind === 'progress') {
            const progressMessage = message;
            progressMessage.progress = 0;
            messages.value.push(progressMessage);
            const lastPos = messages.value.length - 1;
            await simulateProgress(progress => {
                (messages.value[lastPos] as SystemMessageProgress).progress = progress;
            });
            continue;
        }
        if (message.type === 'model-message') {
            const orgMessageData = structuredClone(message.message);
            const messageLen = orgMessageData.length;
            message.message = '';
            message.status = {
                kind: 'progress',
                progress: 0,
            }
            messages.value.push(message);
            const lastPos = messages.value.length - 1;

            let step = 10;
            for (let j = 0; j < messageLen; j += step) {
                step = Math.floor(Math.random() * 10 + 10);
                (messages.value[lastPos] as ConversationMessage).message += orgMessageData.substring(j, j + step);
                const progress = Math.floor(j / messageLen * 100);
                (messages.value[lastPos] as ConversationMessage).status = {
                    kind: 'progress',
                    progress,
                }
                await pause(100 + Math.floor(Math.random() * 100))
            }
            (messages.value[lastPos] as ConversationMessage).status = {
                kind: 'success'
            }
        } else {
            messages.value.push(message);
        }
        await pause(100 + Math.floor(Math.random() * 100))
    }
}

onMounted(async () => {
    simulateMessage();
})

// ----------------- Testing -----------------

</script>
