import { html, TemplateResult } from 'lit-html';

function textAreaAdjustHelper(element: HTMLTextAreaElement) {
    requestAnimationFrame(() => {
        element.style.height = 'auto'
        element.style.height = `${element.scrollHeight}px`
    });
}

function textAreaAdjust(event: KeyboardEvent) {
    const element = event.target as HTMLTextAreaElement;
    textAreaAdjustHelper(element);
}

function onFormSubmit(e: SubmitEvent) {
    e.preventDefault();
    const textArea = document.getElementById('message-box') as HTMLTextAreaElement;

    const message = textArea.value.trim();
    if (message === '') return;
    textArea.value = '';
    textAreaAdjustHelper(textArea);
    console.log(message);
}


export function chatBoxTemplate(...children: TemplateResult[]) {
    return html`
    <article
    class="my-2 w-full overflow-y-auto px-4"
    id="message-container"
    >${children}</article>
    <form class="w-full" id="message-form" @submit="${onFormSubmit}">
        <label for="message-box" class="text-xs text-zinc-300"
            >Message</label
        >
        <div class="flex flex-grow items-center gap-2">
            <textarea
                name="message-input"
                id="message-box"
                class="max-h-20 w-full resize-none overflow-hidden rounded border border-zinc-700 bg-zinc-800 py-1 indent-1 text-sm"
                rows="1"
                placeholder="Type a message..."
                @keydown="${textAreaAdjust}"
            ></textarea>
            <button
                type="submit"
                class="flex aspect-square w-10 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-zinc-100 transition-colors hover:bg-zinc-700 active:bg-zinc-900"
                aria-label="Send Message"
                title="Send Message"
            >
                <ion-icon name="send" aria-hidden="true"></ion-icon>
            </button>
        </div>
    </form>
    `;
}
