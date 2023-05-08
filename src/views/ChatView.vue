<template>
    <AppMainLayout>
        <template #aside>
            <TheSideNavProvider />
        </template>
        <TheChatProvider v-if="filepath" ref="chatProviderRef" @message="onUserMessage" @command="onUserCommand">
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
import { useRouter } from 'vue-router';
import type { Command, ConversationMessage, Message, WebSocketMessage } from '@/model/schema';
import useSocketStore from '@/stores/socketStore';
import { v4 as uuidv4 } from '@lukeed/uuid';
import type { Token } from '@/lib/parser';
import useAppStore from '@/stores/appStore';

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
    chatProviderRef.value?.scrollToLatestMessage();
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
    chatProviderRef.value?.scrollToLatestMessage();
}

// ----------------- Model Params -----------------

const websocketStore = useSocketStore();

const modelParams = computed(() => {
    const { query } = router.currentRoute.value;
    const temp = Array.isArray(query.model_params) ? query.model_params[0] : query.model_params;
    if (!temp) return null;
    return JSON.parse(String(temp));
});

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
                kind: 'loading',
            };
            mess.id = message.id;
            messagesKey.add(message.id);
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

// ----------------- User Message -----------------

function onUserMessage(message: string) {
    const id = uuidv4();
    const temp_message: ConversationMessage = {
        type: 'user-message',
        message,
        webui_id: id,
        id,
        title: 'user',
        status: {
            kind: 'loading',
        },
    };
    updateMessage(temp_message);
    websocketStore.message(temp_message);
}

// ----------------- User Message -----------------


// ----------------- User Command -----------------

function normalizeCommandValue(value: string | undefined | null, type: Command['args'][0]['type']) {
    switch (type) {
        case 'float': {
            if (value == null) return 0;
            const num = parseFloat(value);
            return Number.isNaN(num) ? 0 : num;
        }
        case 'int': {
            if (value == null) return 0;
            const num = parseInt(value);
            return Number.isNaN(num) ? 0 : num;
        }
        case 'string': return String(value ?? '');
        case 'boolean': return value == null || value === 'true';
        default: return value;
    }
}

function normalizeCommand(token: Token[]) {
    if (token[0].type !== 'cmd') return undefined;
    const cmd_name = token[0].value;
    const cmdArgs = [] as Command['args'];
    const { commands } = useAppStore();
    const cmd = commands.find((cmd) => cmd.name === cmd_name);
    if (!cmd) return undefined;
    for (let i = 1; i < token.length; ++i) {
        const arg = token[i];
        if (arg.type === 'cmd') return undefined;
        const argName = arg.type === 'arg' ? arg.name : arg.value;
        const argWitness = cmd.args.find((arg) => arg.name === argName);
        if (!argWitness) continue;
        cmdArgs.push({
            name: argName,
            value: normalizeCommandValue(arg.value, argWitness.type),
            type: argWitness.type,
        });
    }
    return {
        name: cmd_name,
        args: cmdArgs,
    };
}

function onUserCommand(command: Token[]) {
    const cmd = normalizeCommand(command);
    if (!cmd) return;

    websocketStore.message({
        type: 'invoke-command',
        command: cmd.name,
        args: cmd.args,
    })
}

// ----------------- User Command -----------------

</script>
