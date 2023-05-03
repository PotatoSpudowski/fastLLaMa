<template>
    <ion-icon v-if="status.kind === 'failure'" class="text-rose-400 text-lg" name="alert-circle"></ion-icon>
    <sp-progress-circle v-else-if="status.kind === 'loading'" label="Loading" indeterminate static="white"
        size="s"></sp-progress-circle>
    <sp-progress-circle v-else-if="status.kind === 'progress'" label="Loading" static="white" :progress="status.progress"
        size="s"></sp-progress-circle>
</template>

<script setup lang="ts">
import type { MessageStatus } from '@/model/schema';


interface Props {
    status: MessageStatus
}

defineProps<Props>()

</script>
