<template>
    <li class="message-element message-element-right message-element-user my-2">
        <div class="flex items-center gap-1 message__user-title">
            <TheChatMessageStatus :status="message.status" />
            <ion-icon name="person"></ion-icon>
            <span class="text-xs">{{ message.title }}</span>
        </div>
        <output>
            <pre class="text-left message__content">{{ message.message.trim() }}</pre>
        </output>
    </li>
</template>

<script setup lang="ts">
import type { ConversationMessage } from '@/model/schema';
import TheChatMessageStatus from './TheChatMessageStatus.vue';

interface Props {
    message: ConversationMessage
}

defineProps<Props>()
</script>
