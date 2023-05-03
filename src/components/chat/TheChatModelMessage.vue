<template>
    <li class="message-element message-element-model my-2">
        <div class="flex items-center gap-1 message__model-title">
            <ion-icon name="school"></ion-icon>
            <span class="text-xs">{{ title }}</span>
            <TheChatMessageStatus :status="status" />
        </div>
        <output>
            <pre class="text-left message__content">{{ message.trim() }}</pre>
        </output>
    </li>
</template>

<script setup lang="ts">
import type { MessageStatus } from '@/model/schema';
import TheChatMessageStatus from './TheChatMessageStatus.vue';

interface Props {
    title: string
    message: string
    status: MessageStatus
}

defineProps<Props>()
</script>
