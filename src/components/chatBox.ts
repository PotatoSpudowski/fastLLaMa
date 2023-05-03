import { html, TemplateResult } from 'lit-html';

function onFormSubmit(e: SubmitEvent) {
    e.preventDefault();
    const textArea = document.getElementById('message-box') as HTMLTextAreaElement;

    const message = textArea.value.trim();
    if (message === '') return;
    textArea.value = '';
}


export function chatBoxTemplate(...children: TemplateResult[]) {
    return html`
    <article
    class="my-2 w-full overflow-y-auto px-4 pt-2"
    id="message-container"
    >${children}</article>
    <form class="w-full" id="message-form" @submit="${onFormSubmit}">
        <sp-field-label for="message-box">Message</sp-field-label>
        <div class="flex flex-grow items-center gap-2">
            <sp-textfield
                id="message-box"
                grows
                multiline
                resize="none"
                class="w-full max-h-[15rem] overflow-scroll"
                placeholder="Write a message..."
                value="By default the text area has a fixed height and will scroll when text entry goes beyond the available space. With the use of the 'grows' attribute the text area will grow to accomidate the full content of the element."
            ></sp-textfield>
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
