<template>
    <AppMainLayout>
        <template #aside>
            <TheSideNavProvider />
        </template>

        <TheHomeProvider v-slot="{ styleData }">
            <TheHomeWelcomeScreen :style="styleData" v-if="!isFilePickerShowing && !showConnectionDialog"
                @select-model="isFilePickerShowing = true" v-motion-slide-visible-left
                @connect-websocket="showConnectionDialog = true" />
            <AppFilePicker :style="styleData" v-model:show="isFilePickerShowing" :files="dummyFiles" filepath="/"
                v-motion-slide-visible-right @confirm="onConfirm"></AppFilePicker>
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
    console.log(file);
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

function onModelParameterInitialized(args: Omit<z.infer<typeof modelParamsSchema>, 'id' | 'model_path' | 'type'>) {
    if (!modelFile.value) return;
    showModelParameters.value = false;
    router.push({
        name: 'chat', query: {
            path: modelFile.value.path,
            n_threads: String(args.n_threads),
            n_ctx: String(args.n_ctx),
            last_n_size: String(args.last_n_size),
            seed: String(args.seed),
            tokens_to_keep: String(args.tokens_to_keep),
            n_batch: String(args.n_batch),
            use_mmap: String(args.use_mmap),
            use_mlock: String(args.use_mlock),
            load_parallel: String(args.load_parallel),
            n_load_parallel_blocks: String(args.n_load_parallel_blocks),
        }
    });
}

// ---------------- Model Parameters ----------------

</script>
