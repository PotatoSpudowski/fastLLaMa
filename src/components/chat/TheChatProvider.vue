<template>
    <div class="h-full w-full" style="display: grid; grid-template-rows: 1fr auto;">
        <article class="my-2 w-full overflow-y-auto px-4 pt-2 flex flex-col">
            <slot></slot>
        </article>
        <form class="w-full p-2" @submit.prevent="">
            <sp-field-label for="message-box">Message</sp-field-label>
            <div class="flex flex-grow items-center gap-2">
                <sp-textfield grows multiline resize="none" class="w-full max-h-[15rem] overflow-scroll"
                    placeholder="Write a message..."></sp-textfield>
                <button type="submit"
                    class="flex aspect-square w-10 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-zinc-100 transition-colors hover:bg-zinc-700 active:bg-zinc-900"
                    aria-label="Send Message" title="Send Message">
                    <ion-icon name="send" aria-hidden="true"></ion-icon>
                </button>
            </div>
        </form>
    </div>
</template>
