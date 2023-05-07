<template>
    <li class="flex items-center gap-2 font-mono">
        <span class="system-message-tag" :data-kind="message.kind">
            [<span class="flex-grow">{{ message.kind }}</span>]
        </span>
        <span class="text-xs font-semibold text-slate-400">&lt;{{ message.function_name }}&gt;</span>
        <output v-if="message.kind !== 'progress'">
            <pre class="message__content font-mono">{{ chatMessage }}</pre>
        </output>
        <sp-meter v-else :progress="progress" class="my-2 w-full" static="white">{{ message.message }}</sp-meter>
    </li>
</template>

<script setup lang="ts">
import type { SystemMessage } from '@/model/schema';
import { computed } from 'vue';

interface Props {
    message: SystemMessage
}

const props = defineProps<Props>()

const chatMessage = computed(() => {
    const { message } = props;
    return message.kind === 'progress' ? undefined : message.message.trim();
});

const progress = computed(() => {
    const { message } = props;
    return message.kind === 'progress' ? Math.round(message.progress) : undefined;
});

</script>
