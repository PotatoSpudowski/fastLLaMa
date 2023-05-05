<template>
    <sp-underlay :open="show"></sp-underlay>
    <sp-dialog v-if="show" class="w-[25rem] bg-zinc-900" no-divider dismissable @close="onClose">
        <div slot="heading" class="">
            <h2>Model Parameters</h2>
        </div>
        <form onsubmit.prevent="">
            <div class="flex flex-wrap items-center" style="gap: 0.5rem 2rem">
                <div class="flex-grow">
                    <!-- n_threads: z.number().optional(), -->
                    <sp-field-label for="model_init_n_threads">Number of threads</sp-field-label>
                    <sp-number-field name="model_init_n_threads" id="model_init_n_threads" min="0"
                        :max="Number.MAX_SAFE_INTEGER" class="w-[8rem]"
                        v-model.number="modelParams.n_threads"></sp-number-field>
                </div>

                <div class="flex-grow">
                    <!-- n_ctx: z.number().optional(), -->
                    <sp-field-label for="model_init_n_ctx">Context Size</sp-field-label>
                    <sp-number-field name="model_init_n_ctx" id="model_init_n_ctx" min="0" :max="Number.MAX_SAFE_INTEGER"
                        class="w-[8rem]" v-model.number="modelParams.n_ctx"></sp-number-field>
                </div>

                <div class="flex-grow">
                    <!-- last_n_size: z.number().optional(), -->
                    <sp-field-label for="model_init_last_n_size">Token history size</sp-field-label>
                    <sp-number-field name="model_init_last_n_size" id="model_init_last_n_size" min="0"
                        :max="Number.MAX_SAFE_INTEGER" class="w-[8rem]"
                        v-model.number="modelParams.last_n_size"></sp-number-field>
                </div>

                <div class="flex-grow">
                    <!-- tokens_to_keep: z.number().optional(), -->
                    <sp-field-label for="model_init_tokens_to_keep">Token to keep</sp-field-label>
                    <sp-number-field name="model_init_tokens_to_keep" id="model_init_tokens_to_keep"
                        :max="Number.MAX_SAFE_INTEGER" min="0" class="w-[8rem]"
                        v-model.number="modelParams.tokens_to_keep"></sp-number-field>
                </div>

                <div class="flex-grow">
                    <!-- n_batch: z.number().optional(), -->
                    <sp-field-label for="model_init_n_batch">Batch Size</sp-field-label>
                    <sp-number-field name="model_init_n_batch" id="model_init_n_batch" min="0"
                        :max="Number.MAX_SAFE_INTEGER" class="w-[8rem]"
                        v-model.number="modelParams.n_batch"></sp-number-field>
                </div>

                <div class="flex-grow">
                    <!-- n_load_parallel_blocks: z.number().optional(), -->
                    <sp-field-label for="model_init_n_load_parallel_blocks">Parallel load Batch Size</sp-field-label>
                    <sp-number-field name="model_init_n_load_parallel_blocks" id="model_init_n_load_parallel_blocks" min="0"
                        :max="Number.MAX_SAFE_INTEGER" class="w-[8rem]"
                        v-model.number="modelParams.n_load_parallel_blocks"></sp-number-field>
                </div>

                <div class="flex-grow mt-5 w-[8rem]">
                    <!-- use_mmap: z.boolean().optional(), -->
                    <sp-checkbox :checked="modelParams.use_mmap"
                        @change="onItemSelected('use_mmap', $event)">Mmap</sp-checkbox>
                </div>

                <div class="flex-grow mt-5 w-[8rem]">
                    <!-- load_parallel: z.boolean().optional(), -->
                    <sp-checkbox :checked="modelParams.load_parallel"
                        @change="onItemSelected('load_parallel', $event)">Parallel load</sp-checkbox>
                </div>

            </div>
            <div class="w-full">
                <!-- seed: z.number().optional(), -->
                <sp-field-label for="model_init_seed">Seed</sp-field-label>
                <sp-number-field name="model_init_seed" id="model_init_seed" min="0" :max="Number.MAX_SAFE_INTEGER"
                    class="w-full" v-model.number="modelParams.seed"></sp-number-field>
            </div>

            <sp-button-group class="min-w-fit mt-8">
                <sp-button static="white" @click="emits('confirm', modelParams)">Initialize</sp-button>
            </sp-button-group>
        </form>
    </sp-dialog>
</template>

<script setup lang="ts">
import type { modelParamsSchema } from '@/model/schema';
import { reactive } from 'vue';
import type { z } from 'zod';

type ModelParameters = Omit<z.infer<typeof modelParamsSchema>, 'id' | 'model_path' | 'type'>;

interface Props {
    show: boolean,
}

interface Emits {
    (e: 'update:show', value: boolean): void,
    (e: 'confirm', params: ModelParameters): void,
}

defineProps<Props>();
const emits = defineEmits<Emits>();

const modelParams = reactive<ModelParameters>({
    n_threads: 4,
    n_ctx: 512,
    last_n_size: 64,
    seed: 0,
    tokens_to_keep: 200,
    n_batch: 16,
    use_mmap: false,
    use_mlock: false,
    load_parallel: false,
    n_load_parallel_blocks: 1,
})

function onClose() {
    emits('update:show', false);
}

function onItemSelected(key: keyof typeof modelParams, e: Event) {
    const target = e.target as HTMLInputElement;
    (modelParams[key] as boolean) = target.checked;
}

</script>

<style scoped>
sp-underlay:not([open])+sp-dialog {
    display: none;
}

sp-underlay+sp-dialog {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
    background: var(--spectrum-gray-100);
}
</style>
