<template>
    <li class="flex items-center gap-2 font-mono">
        <span class="system-message-tag" :data-kind="kind">
            [<span class="flex-grow">{{ kind }}</span>]
        </span>
        <span class="text-xs font-semibold text-slate-400">&lt;{{ functionName }}&gt;</span>
        <output>
            <pre class="message__content font-mono">{{ message.trim() }}</pre>
        </output>
    </li>
</template>

<script setup lang="ts">
import type { SystemMessage } from '@/model/schema';

interface Props {
    kind: SystemMessage['kind']
    functionName: string
    message: string
}

defineProps<Props>()
</script>
