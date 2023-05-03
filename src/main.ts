import { modelMessageTemplate, systemMessageTemplate, userMessageTemplate } from './components/messages';
import './index.css'
import { render, html } from 'lit-html';
import { Message, SystemMessage, dummyFiles, dummyMessages } from './model/dummy';
import { chatBoxTemplate } from './components/chatBox';
import '@spectrum-web-components/theme/theme-darkest.js';
import '@spectrum-web-components/theme/scale-medium.js';
import '@spectrum-web-components/theme/sp-theme.js';
import '@spectrum-web-components/theme/src/themes.js';
import '@spectrum-web-components/sidenav/sp-sidenav.js';
import '@spectrum-web-components/sidenav/sp-sidenav-heading.js';
import '@spectrum-web-components/sidenav/sp-sidenav-item.js';
import '@spectrum-web-components/button/sp-button.js';
import '@spectrum-web-components/button/sp-clear-button.js';
import '@spectrum-web-components/button/sp-close-button.js';
import '@spectrum-web-components/button-group/sp-button-group.js';
import '@spectrum-web-components/dialog/sp-dialog.js';
import '@spectrum-web-components/textfield/sp-textfield.js';
import '@spectrum-web-components/field-label/sp-field-label.js';
import '@spectrum-web-components/search/sp-search.js';
import '@spectrum-web-components/menu/sp-menu.js';
import '@spectrum-web-components/menu/sp-menu-group.js';
import '@spectrum-web-components/menu/sp-menu-item.js';
import '@spectrum-web-components/menu/sp-menu-divider.js';
import '@spectrum-web-components/icon/sp-icon.js';
// import '@spectrum-web-components/table/elements.js';
import { HomePage } from './components/HomePage';
import { FileStructure } from './model/schema';

type UserState = {
    currentScrollHeight: number
    scrollThreshold: number
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
        html`
            <ul class="flex flex-col">
                ${messageElements}
            </ul>
        `
    );
    render(list, container);
    scrollToTheLastElement(container, userState);
}

async function chatPage() {
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

function renderHomePage() {
    const mainContainer = document.getElementById('main-body') as HTMLElement;
    render(HomePage("test/test/ste", dummyFiles), mainContainer);
}

async function main() {
    const body = html`
        <aside style="background-color: var(--spectrum-gray-75)">
            <sp-sidenav defaultValue="Docs">
                <sp-sidenav-item value="Docs" href="/components/SideNav">
                    Docs
                </sp-sidenav-item>
            </sp-sidenav>
        </aside>
        <main id="main-body"></main>
    `

    const app = document.getElementById('app') as HTMLElement;
    render(body, app);

    renderHomePage();
    // chatPage();

    document.addEventListener('file-select', (e: CustomEvent<FileStructure>) => {
        render(body, app);
        chatPage();
    });
}

window.onload = main;

