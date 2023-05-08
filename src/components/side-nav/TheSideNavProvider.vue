<template>
    <div class="h-full" style="display: grid; grid-template-rows: 1fr auto;">
        <sp-accordion allow-multiple class="h-full overflow-y-auto">
            <sp-accordion-item v-for="histories, date in normalizedHistory" :key="date" :label="date" open>
                <sp-menu class="w-full">
                    <sp-menu-item v-for="history in histories" :key="history.id" class="group">
                        <div class="w-full justify-between" style="display: grid; grid-template-columns: 1fr auto;">
                            <sp-menu-item-label class="whitespace-nowrap overflow-x-hidden text-ellipsis"
                                @dblclick="() => onLoad(history)">
                                {{ history.title }}
                            </sp-menu-item-label>
                            <button title="Delete" aria-label="Delete history item" tabindex="1"
                                @click.prevent.stop="() => onDelete(history.id)"
                                class="aspect-square w-6 rounded-full flex justify-center items-center hover:bg-rose-500 active:bg-rose-600 transition-all opacity-0 group-hover:opacity-100">
                                <ion-icon name="close-outline" class="text-rose-50"></ion-icon>
                            </button>
                        </div>
                    </sp-menu-item>
                </sp-menu>
            </sp-accordion-item>
        </sp-accordion>
        <sp-button-group v-if="showSaveHistoryButton" class="flex justify-center items-center py-2">
            <sp-button variant="primary" @click="onSave">
                Save
            </sp-button>
        </sp-button-group>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';
import type { SaveHistoryItem } from '@/model/schema';
import { useRouter } from 'vue-router';
import { storeToRefs } from 'pinia';
import useAppStore from '@/stores/appStore';
import useSocketStore from '@/stores/socketStore';

const { saveHistory } = storeToRefs(useAppStore())
const router = useRouter();

function formatDateWithPadding(date: Date) {
    const year = date.getFullYear();
    const month = date.getMonth() + 1;
    const day = date.getDate();
    return `${year}-${('0' + month.toString()).slice(-2)}-${('0' + day.toString()).slice(-2)}`;
}

const normalizedHistory = computed(() => {
    const res: Record<string, SaveHistoryItem[]> = {};
    const temp = saveHistory.value.slice().sort((a, b) => b.date - a.date);
    temp.forEach((el) => {
        const date = new Date(el.date);
        const key = formatDateWithPadding(date);
        if (!res[key]) res[key] = [];
        res[key].push(el);
    });
    return res;
});

const showSaveHistoryButton = computed(() => {
    return router.currentRoute.value.name === 'chat';
})

function onSave() {
    useSocketStore().message({
        type: 'session-save'
    });
}

function onDelete(id: string) {
    useSocketStore().message({
        type: 'session-delete',
        id,
    });
}

const readQuery = (key: string) => {
    const query = router.currentRoute.value.query;
    if (query[key]) {
        const val = query[key];
        return (Array.isArray(val) ? val[0] : val) as string;
    }
    return null;
}

async function onLoad(saveItem: SaveHistoryItem) {
    if (router.currentRoute.value.name === 'chat') {
        const filepath = readQuery('path');
        if (filepath === saveItem.model_path) {
            console.log(filepath, saveItem.model_path);
            useSocketStore().message({
                type: 'session-load',
                id: saveItem.id,
            });
            return;
        }
    }
    const routeResult = await router.push({
        name: 'chat',
        query: {
            path: saveItem.model_path,
            model_params: JSON.stringify(saveItem.model_args),
        },
    });
    if (routeResult) return;

    useSocketStore().message({
        type: 'session-load',
        id: saveItem.id,
    })
}

</script>
