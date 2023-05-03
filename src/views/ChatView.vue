<template>
    <TheChatProvider v-if="filepath">
        <template v-for="message in messages" :key="message.id">
            <TheChatSystemMessage v-if="message.type === 'system'" :message="message.message" :kind="message.kind"
                :function-name="message.function_name" />
            <TheChatUserMessage v-else-if="message.type === 'user'" :message="message.message" :title="message.title"
                :status="message.status" />
            <TheChatModelMessage v-else-if="message.type === 'model'" :message="message.message" :title="message.title"
                :status="message.status" />
        </template>
    </TheChatProvider>
</template>

<script setup lang="ts">
import { computed, onBeforeMount, onMounted, ref } from 'vue';
import TheChatModelMessage from '@/components/chat/TheChatModelMessage.vue';
import TheChatProvider from '@/components/chat/TheChatProvider.vue';
import TheChatSystemMessage from '@/components/chat/TheChatSystemMessage.vue';
import TheChatUserMessage from '@/components/chat/TheChatUserMessage.vue';
import { dummyMessages } from '@/model/dummy';
import { useRouter } from 'vue-router';

const messages = ref(dummyMessages);
const router = useRouter();

const filepath = computed(() => {
    const { query } = router.currentRoute.value;
    return String(Array.isArray(query.path) ? query.path[0] : query.path);
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
        await new Promise(resolve => setTimeout(resolve, 100 + Math.floor(Math.random() * 100)));

        i += step;
    }
})

</script>
