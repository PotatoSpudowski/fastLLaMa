import { modelMessageTemplate, systemMessageTemplate, userMessageTemplate } from './components/messages';
import './index.css'
import { render, html } from 'lit-html';
import { Message, SystemMessage, dummyMessages } from './model/dummy';
import { FAST_LLAMA_WATERMARK } from './lib/watermark';
import { chatBoxTemplate } from './components/chatBox';

type UserState = {
    currentScrollHeight: number
    scrollThreshold: number
}

function watermarkTemplate() {
    return html` <pre aria-hidden="true" id="water" class="fastllama-watermark">${FAST_LLAMA_WATERMARK}</pre>`
}

function scrollToTheLastElement(container: HTMLElement, userState: UserState) {
    const ulElelemnt = container.querySelector<HTMLUListElement>('ul');
    if (ulElelemnt === null) return;
    const lastElement = ulElelemnt.lastElementChild;
    if (lastElement === null) return;
    requestAnimationFrame(() => {
        if (userState.currentScrollHeight === -1) userState.currentScrollHeight = container.scrollHeight;
        else if (userState.currentScrollHeight === container.scrollHeight) return;
        lastElement.scrollIntoView({ behavior: 'smooth' });
    })
}

function renderMessages(container: HTMLElement, messages: Message[], userState: UserState) {
    const messageElements = messages.map((message) => {
        if (message.type === 'user') return userMessageTemplate(message.title, message.message);
        else if (message.type === 'model') return modelMessageTemplate(message.title, message.message);
        const systemMessage = message as SystemMessage;
        return systemMessageTemplate(systemMessage.kind, systemMessage.function_name, systemMessage.message);
    })
    const list = chatBoxTemplate(
        watermarkTemplate(),
        html`
            <ul class="flex flex-col">
                ${messageElements}
            </ul>
        `
    );
    render(list, container);
    scrollToTheLastElement(container, userState);
}

async function main() {

    const mainContainer = document.getElementById('main-body') as HTMLElement;

    const userState: UserState = {
        currentScrollHeight: -1,
        scrollThreshold: 10
    };

    const lastMessage = dummyMessages[dummyMessages.length - 1];
    const message = lastMessage.message;
    const len = message.length;

    lastMessage.message = '';
    for (let i = 0; i < len; i += 10) {
        lastMessage.message += message.substring(i, i + 10);
        renderMessages(mainContainer, dummyMessages, userState);
        await new Promise((resolve) => setTimeout(resolve, 200));
    }
}

window.onload = main;

