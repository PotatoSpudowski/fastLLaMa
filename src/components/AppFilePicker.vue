<template>
    <sp-dialog v-if="show" class="w-[25rem] bg-zinc-900" no-divider dismissable @close="emits('update:show', false)">
        <div slot="heading">
            <h2 class="mb-2">Select a Modal</h2>
            <sp-search :value="filepath" class="w-[15rem]"></sp-search>
        </div>

        <div class="max-h-[10rem] overflow-y-auto">
            <sp-menu class="w-full" selects="single" @change="onItemSelect">
                <sp-menu-item v-for="file in files" :key="file.type + file.path" :value="`${file.path};${file.type}`"
                    @dblclick="emits('open', file)">
                    <ion-icon slot="icon" :name="file.type === 'directory' ? 'folder' : 'document'" />
                    <div class="flex items-center gap-2 w-full justify-between">
                        <span>{{ file.name }}</span>
                        <ion-icon aria-hidden="true" name="chevron-forward-outline"
                            v-if="file.type === 'directory'"></ion-icon>
                    </div>
                </sp-menu-item>
            </sp-menu>
        </div>

        <div class="flex flex-col">
            <sp-button-group class="min-w-fit mt-2">
                <sp-button variant="secondary" treatment="outline">Go Back</sp-button>
                <sp-button static="white" id="file-dialog-confirm-btn" :disabled="disableConfirmBtn"
                    @click="onConfirm">Confirm</sp-button>
            </sp-button-group>
        </div>
    </sp-dialog>
</template>

<script setup lang="ts">
import type { FileStructure } from '@/model/schema';
import { ref } from 'vue';
import { Menu } from '@spectrum-web-components/menu';

interface Props {
    show: boolean,
    files: FileStructure[],
    filepath: string
}

interface Emits {
    (e: 'confirm', file: FileStructure): void
    (e: 'update:show', show: boolean): void
    (e: 'open', file: FileStructure): void
}

const props = defineProps<Props>()
const emits = defineEmits<Emits>()

const selectedFile = ref<FileStructure>();
const disableConfirmBtn = ref(true);

function onConfirm() {
    if (selectedFile.value) emits('confirm', selectedFile.value);
}

function onItemSelect(e: Event) {
    if (!(e.target instanceof Menu)) return;

    const value = e.target.value as string;
    const [path, type] = value.split(';');
    disableConfirmBtn.value = type === 'directory';
    selectedFile.value = props.files.find(file => file.path === path && file.type === type);
}

</script>
