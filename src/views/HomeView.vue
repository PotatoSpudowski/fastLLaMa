<template>
    <TheHomeProvider v-slot="{ styleData }">
        <TheHomeWelcomeScreen :style="styleData" v-if="!isFilePickerShowing" @select-model="isFilePickerShowing = true"
            v-motion-slide-visible-left />
        <AppFilePicker :style="styleData" v-model:show="isFilePickerShowing" :files="dummyFiles" filepath="/"
            v-motion-slide-visible-right @confirm="onConfirm"></AppFilePicker>
    </TheHomeProvider>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import AppFilePicker from '@/components/AppFilePicker.vue';
import TheHomeProvider from '@/components/home/TheHomeProvider.vue';
import TheHomeWelcomeScreen from '@/components/home/TheHomeWelcomeScreen.vue';
import { dummyFiles } from '@/model/dummy';
import type { FileStructure } from '@/model/schema';
import { useRouter } from 'vue-router';

const isFilePickerShowing = ref(true);
const router = useRouter();

function onConfirm(file: FileStructure) {
    console.log(file);
    router.push({ name: 'chat', query: { path: file.path } });
}

</script>
