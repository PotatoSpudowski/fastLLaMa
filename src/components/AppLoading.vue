<template>
    <Teleport to="body" v-if="modelValue">
        <sp-theme theme="spectrum" color="darkest" scale="medium" class="h-full fixed top-0 left-0">
            <sp-underlay class="loading-underlay" v-motion-fade open></sp-underlay>
            <div class="loading-container" v-motion-pop-visible>
                <sp-progress-circle :label="message" indeterminate></sp-progress-circle>
                <span>{{ message }}</span>
            </div>
        </sp-theme>
    </Teleport>
</template>

<script setup lang="ts">
import { onMounted } from 'vue';


interface Props {
    modelValue: boolean,
    message: string,
    timeout?: number,
}

interface Emits {
    (e: 'update:modelValue', value: boolean): void,
}

const props = defineProps<Props>();
const emits = defineEmits<Emits>();

onMounted(() => {
    if (props.timeout == null) return;
    setTimeout(() => {
        emits('update:modelValue', false);
    }, props.timeout);
})

</script>

<style>
sp-underlay.loading-underlay:not([open])+sp-dialog {
    display: none;
}

.loading-container {
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    z-index: 1;
    background: var(--spectrum-gray-100);
    padding: 1rem;
    display: flex;
    color: white;
    gap: 1rem;
    align-items: center;
}
</style>
