<template>
    <div class="flex flex-col items-center justify-center">
        <img src="img/image.png" class="h-[30rem] aspect-[46/50]" />
        <sp-button-group class="min-w-fit my-2">
            <sp-button static="white" @click="emits('select-model')">Select Modal</sp-button>
        </sp-button-group>
    </div>
</template>

<script setup lang="ts">

interface Emits {
    (e: 'select-model'): void
}

const emits = defineEmits<Emits>()

</script>
