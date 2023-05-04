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
        </TheHomeProvider>
    </AppMainLayout>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import AppFilePicker from '@/components/AppFilePicker.vue';
import TheHomeProvider from '@/components/home/TheHomeProvider.vue';
import TheHomeWelcomeScreen from '@/components/home/TheHomeWelcomeScreen.vue';
import { dummyFiles } from '@/model/dummy';
import type { FileStructure } from '@/model/schema';
import AppMainLayout from '@/layout/AppMainLayout.vue';
import TheSideNavProvider from '@/components/side-nav/TheSideNavProvider.vue';
import { useRouter } from 'vue-router';
import AppConnectDialog from '@/components/AppConnectDialog.vue';
import { useEventListener } from '@vueuse/core';

const isFilePickerShowing = ref(false);
const router = useRouter();

function onConfirm(file: FileStructure) {
    console.log(file);
    router.push({ name: 'chat', query: { path: file.path } });
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

</script>
