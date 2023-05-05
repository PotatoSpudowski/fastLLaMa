<template>
    <AppMainLayout>
        <template #aside>
            <TheSideNavProvider />
        </template>
        <TheChatProvider v-if="filepath" ref="chatProviderRef">
            <template v-for="message in messages" :key="message.id">
                <TheChatSystemMessage v-if="message.type === 'system'" :message="message" />
                <TheChatUserMessage v-else-if="message.type === 'user'" :message="message" />
                <TheChatModelMessage v-else-if="message.type === 'model'" :message="message" />
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
import type { ConversationMessage, Message, SystemMessageProgress } from '@/model/schema';
import { pause } from '@/lib/utils';

const messages = ref<Message[]>([]);
const router = useRouter();
const chatProviderRef = ref<InstanceType<typeof TheChatProvider> | null>(null);

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
        if (message.type === 'system' && message.kind === 'progress') {
            const progressMessage = message;
            progressMessage.progress = 0;
            messages.value.push(progressMessage);
            const lastPos = messages.value.length - 1;
            await simulateProgress(progress => {
                (messages.value[lastPos] as SystemMessageProgress).progress = progress;
            });
            continue;
        }
        if (message.type === 'model') {
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
