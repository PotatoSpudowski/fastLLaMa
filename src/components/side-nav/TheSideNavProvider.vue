<template>
    <div class="h-full" style="display: grid; grid-template-rows: 1fr auto;">
        <sp-accordion allow-multiple class="h-full overflow-y-auto">
            <sp-accordion-item v-for="histories, date in normalizedHistory" :key="date" :label="date" open>
                <sp-menu class="w-full">
                    <sp-menu-item v-for="history, i in histories" :key="history.id + i" class="group">
                        <div class="w-full justify-between" style="display: grid; grid-template-columns: 1fr auto;">
                            <sp-menu-item-label class="whitespace-nowrap overflow-x-hidden text-ellipsis">
                                {{ history.title }}
                            </sp-menu-item-label>
                            <button title="Delete" aria-label="Delete history item" tabindex="1"
                                class="aspect-square w-6 rounded-full flex justify-center items-center hover:bg-rose-500 active:bg-rose-600 transition-all opacity-0 group-hover:opacity-100">
                                <ion-icon name="close-outline" class="text-rose-50"></ion-icon>
                            </button>
                        </div>
                    </sp-menu-item>
                </sp-menu>
            </sp-accordion-item>
        </sp-accordion>
        <sp-button-group class=" flex justify-center items-center py-2">
            <sp-button variant="primary" @click="history.push({ id: '1', title: 'test', date: Date.now() })">
                Save
            </sp-button>
        </sp-button-group>
    </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { dummyHistory } from '@/model/dummy';
import type { SaveHistoryItem } from '@/model/schema';

const history = ref(dummyHistory)

function formatDateWithPadding(date: Date) {
    const year = date.getFullYear();
    const month = date.getMonth() + 1;
    const day = date.getDate();
    return `${year}-${('0' + month.toString()).slice(-2)}-${('0' + day.toString()).slice(-2)}`;
}

const normalizedHistory = computed(() => {
    const res: Record<string, SaveHistoryItem[]> = {};
    const temp = history.value.slice().sort((a, b) => b.date - a.date);
    temp.forEach((el) => {
        const date = new Date(el.date);
        const key = formatDateWithPadding(date);
        if (!res[key]) res[key] = [];
        res[key].push(el);
    });
    return res;
});

</script>
