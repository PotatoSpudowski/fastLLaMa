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

const messages = ref(dummyMessages);
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

onMounted(async () => {
    const lastElement = messages.value[messages.value.length - 1];
    if (!lastElement || lastElement.type === 'system') return;
    const message = lastElement.message;
    const len = message.length

    lastElement.message = '';

    for (let i = 0; i < len;) {
        const step = 10 + Math.floor(Math.random() * 10);
        lastElement.message += message.substring(i, i + step);

        if (lastElement.message.length === len) {
            lastElement.status = { kind: 'success' }
        } else {
            lastElement.status = {
                kind: 'progress',
                progress: (i / len) * 100
            }
        }
        chatProviderRef.value?.scrollToLatestMessage({
            behavior: 'smooth', block: 'center', inline: 'center'
        });
        await new Promise(resolve => setTimeout(resolve, 100 + Math.floor(Math.random() * 100)));

        i += step;
    }
})

</script>
