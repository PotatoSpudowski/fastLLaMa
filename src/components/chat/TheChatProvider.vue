<template>
    <div class="h-full w-full" style="display: grid; grid-template-rows: 1fr auto;">
        <article class="w-full overflow-y-auto px-4 pt-2 flex flex-col last:mb-8" ref="messageContainerRef"
            @scroll="onScroll">
            <slot></slot>
        </article>
        <form class="w-full p-2 relative" @submit.prevent="onSubmit">
            <sp-field-label for="message-box">Message</sp-field-label>
            <sp-popover open v-if="nextPossibleSuggestions.length > 0 && showSuggestions"
                style="position: absolute; bottom: 100%;" class="border border-zinc-600 max-h-[10rem]"
                :style="{ left: `min(${cursorPosition * inputFontSize}px, 100%)` }">
                <sp-menu>
                    <sp-menu-item v-for="sugg in nextPossibleSuggestions" :key="sugg.value + sugg.type" :value="sugg.value">
                        {{ tokenToString(sugg) }}
                    </sp-menu-item>
                </sp-menu>
            </sp-popover>
            <div class="flex flex-grow items-center gap-2">
                <sp-textfield grows multiline class="w-full max-h-[15rem] min-w-[70%] overflow-hidden resize-none"
                    placeholder="Write a message..." @keydown="onKeyUp" ref="inputRef"
                    v-model.trim="inputValue"></sp-textfield>
                <button type="submit"
                    class="flex aspect-square w-10 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-zinc-100 transition-colors hover:bg-zinc-700 active:bg-zinc-900"
                    aria-label="Send Message" title="Send Message">
                    <ion-icon name="send" aria-hidden="true"></ion-icon>
                </button>
            </div>
        </form>
    </div>
</template>

<script setup lang="ts">
import { parseCommand, type Token } from '@/lib/parser';
import { computed, nextTick, ref, watchEffect } from 'vue';
import { dummyCommands } from '@/model/dummy';
import { watchDebounced } from '@vueuse/core';

interface Emits {
    (e: 'message', message: string): void,
    (e: 'command', command: Token[]): void,
}

const emits = defineEmits<Emits>();

const inputRef = ref<HTMLInputElement | null>(null);
const inputValue = ref('');

function onSubmit() {
    if (!inputRef.value) return;
    const value = inputRef.value.value.trim();
    if (!value) return;
    const command = parseCommand(value);
    if (!command) {
        emits('message', value);
    } else {
        emits('command', command);
        console.log(command);
    }
    inputValue.value = '';
}

function onKeyUp(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        onSubmit();
    }
}

function getTextElement() {
    if (!inputRef.value) return null;
    return inputRef.value.shadowRoot?.querySelector('textarea') as HTMLTextAreaElement;
}

// Fix for textarea not resizing properly
watchEffect(async (cleanUp) => {
    if (!inputRef.value) return;

    const timerId = setInterval(async () => {
        if (!inputRef.value) return;
        await nextTick();
        const textEl = getTextElement();

        if (!textEl) return;
        textEl.style.maxHeight = '15rem';
        textEl.style.overflow = 'scroll';
        clearInterval(timerId);
    }, 200);
    cleanUp(() => {
        clearInterval(timerId);
    });
});


// ------------------ Command Suggestions ------------------
const cursorPosition = ref(0);
const commandSuggestions = ref<Token[]>([]);
const possibleCommands = dummyCommands;
const showSuggestions = ref(false);
const inputFontSize = computed(() => {
    if (!inputRef.value) return 0;
    const fontSize = getComputedStyle(inputRef.value as HTMLElement).fontSize;
    return parseFloat(fontSize) * 0.5;
});

watchDebounced(inputValue, () => {
    const textEl = getTextElement();
    if (!textEl) return;
    cursorPosition.value = textEl.selectionStart ?? 0;

    const value = inputValue.value;

    if (!value.startsWith('/')) {
        showSuggestions.value = false;
        commandSuggestions.value = [];
        return;
    }
    const command = parseCommand(value);
    if (!command) {
        commandSuggestions.value = [];
        showSuggestions.value = false;
        return;
    }
    showSuggestions.value = true;
    if (command[0].value === '') return;
    commandSuggestions.value = command;
});

const nextPossibleSuggestions = computed(() => {
    if (!showSuggestions.value) return [];
    if (commandSuggestions.value.length === 0) return Object.keys(possibleCommands).map((key) => {
        return {
            type: 'cmd',
            value: key,
        } as Token;
    });

    const command = possibleCommands[commandSuggestions.value[0].value];
    if (!command) return [];

    const unusedCommands: Token[] = [];

    for (let i = 0; i < command.length; ++i) {
        const el = command[i];
        const argName = el.type === 'arg' ? el.name : el.value;
        const cmdExists = commandSuggestions.value.some((cmd) => {
            if (cmd.type === 'arg') return cmd.name === argName;
            return cmd.value === argName;
        });
        if (!cmdExists) unusedCommands.push(el);
    }
    return unusedCommands;
});

function tokenToString(token: Token) {
    if (token.type === 'cmd') return `/${token.value}`;
    if (token.type === 'arg') return `${token.name}=`;
    return `${token.value}`;
}

// ------------------ Command Suggestions ------------------

// ------------------ Message ------------------------------

const messageContainerRef = ref<HTMLElement | null>(null);

const userScrollState = {
    isAtBottom: true,
    threshold: 0.02,
};

function onScroll() {
    requestAnimationFrame(() => {
        if (!messageContainerRef.value) return;
        const el = messageContainerRef.value;
        const scrollHeight = el.scrollHeight;
        const scrollTop = el.scrollTop;
        const clientHeight = el.clientHeight;
        const scrollBottom = scrollHeight - scrollTop - clientHeight;

        if (scrollBottom <= scrollHeight * userScrollState.threshold) {
            userScrollState.isAtBottom = true;
        } else {
            userScrollState.isAtBottom = false;
        }
    })
}

function scrollToLatestMessage(options?: Parameters<HTMLElement['scrollIntoView']>[0]) {
    requestAnimationFrame(() => {
        const el = messageContainerRef.value;
        if (!el || !userScrollState.isAtBottom) return;
        (el.lastElementChild as HTMLElement).scrollIntoView(options);
    })
}

// ------------------ Message ------------------------------

defineExpose({
    scrollToLatestMessage,
});

</script>
