<template>
    <AppMainLayout>
        <template #aside>
            <TheSideNavProvider />
        </template>

        <TheHomeProvider v-slot="{ styleData }">
            <TheHomeWelcomeScreen :style="styleData" v-if="!isFilePickerShowing && !showConnectionDialog"
                @select-model="isFilePickerShowing = true" v-motion-slide-visible-left
                @connect-websocket="showConnectionDialog = true" />
            <AppFilePicker :style="styleData" v-model:show="isFilePickerShowing" v-motion-slide-visible-right
                @confirm="onConfirm"></AppFilePicker>
            <AppConnectDialog v-model:show="showConnectionDialog" v-motion-slide-visible-right :style="styleData" />
            <TheHomeModelParameters v-model:show="showModelParameters" @confirm="onModelParameterInitialized">
            </TheHomeModelParameters>
        </TheHomeProvider>
    </AppMainLayout>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import AppFilePicker from '@/components/AppFilePicker.vue';
import TheHomeProvider from '@/components/home/TheHomeProvider.vue';
import TheHomeWelcomeScreen from '@/components/home/TheHomeWelcomeScreen.vue';
import { dummyFiles } from '@/model/dummy';
import type { FileStructure, modelParamsSchema } from '@/model/schema';
import AppMainLayout from '@/layout/AppMainLayout.vue';
import TheSideNavProvider from '@/components/side-nav/TheSideNavProvider.vue';
import { useRouter } from 'vue-router';
import AppConnectDialog from '@/components/AppConnectDialog.vue';
import { useEventListener } from '@vueuse/core';
import TheHomeModelParameters from '@/components/home/TheHomeModelParameters.vue';
import type { z } from 'zod';

const isFilePickerShowing = ref(false);
const router = useRouter();

function onConfirm(file: FileStructure) {
    modelFile.value = file;
    showModelParameters.value = true;
}

// ---------------- Websocket ----------------
const showConnectionDialog = ref(false);

// ---------------- Websocket ----------------

// ---------------- Close All ----------------

useEventListener(document, 'keyup', (e) => {
    if (e.key === 'Escape') {
        isFilePickerShowing.value = false;
        showConnectionDialog.value = false;
    }
});

// ---------------- Close All ----------------

// ---------------- Model Parameters ----------------
const showModelParameters = ref(false);
const modelFile = ref<FileStructure | null>(null);

async function onModelParameterInitialized(args: Omit<z.infer<typeof modelParamsSchema>, 'id' | 'model_path' | 'type'>) {
    if (!modelFile.value) return;
    showModelParameters.value = false;
    const model_params = JSON.stringify({
        n_threads: args.n_threads,
        n_ctx: args.n_ctx,
        last_n_size: args.last_n_size,
        seed: args.seed,
        tokens_to_keep: args.tokens_to_keep,
        n_batch: args.n_batch,
        use_mmap: args.use_mmap,
        use_mlock: args.use_mlock,
        load_parallel: args.load_parallel,
        n_load_parallel_blocks: args.n_load_parallel_blocks,
    });
    router.push({
        name: 'chat', query: {
            model_params,
            path: modelFile.value.path,
        }
    });
}

// ---------------- Model Parameters ----------------

</script>
