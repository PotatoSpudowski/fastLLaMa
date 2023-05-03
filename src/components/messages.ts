import { html } from 'lit-html';
import { MessageStatus, SystemMessage } from '../model/dummy';

function statusTemplate(status: MessageStatus) {
    switch (status.kind) {
        case 'failure':
            return html`<ion-icon class="text-rose-400 text-lg" name="alert-circle"></ion-icon>`;
        case 'loading':
            return html`<sp-progress-circle
            label="Loading"
            indeterminate
            static="white"
            size="s"
        ></sp-progress-circle>`;
        case 'progress':
            return html`<sp-progress-circle
            label="Loading"
            static="white"
            progress="${status.progress}"
            size="s"
        ></sp-progress-circle>`;
        default: return html``;
    }
}

export function userMessageTemplate(title: string, message: string, status: MessageStatus) {
    return html`
        <li class="message-element message-element-right message-element-user my-2">
            <div class="flex items-center gap-1 message__user-title">
                <ion-icon name="person"></ion-icon>
                <span>${title}</span>
                ${statusTemplate(status)}
            </div>
            <pre class="text-right message__content">${message.trim()}</pre>
        </li>
    `;
}

export function modelMessageTemplate(title: string, message: string, status: MessageStatus) {
    return html`
        <li class="message-element message-element-model my-2">
            <div class="flex items-center gap-1 message__model-title">
                <ion-icon name="school"></ion-icon>
                <span class="text-xs">${title}</span>
                ${statusTemplate(status)}
            </div>
            <pre class="text-left message__content">${message.trim()}</pre>
        </li>
    `;
}

export function systemMessageTemplate(kind: SystemMessage['kind'], function_name: string, message: string) {
    return html`
        <li class="flex items-center gap-2 font-mono">
            <span class="system-message-tag" data-kind="${kind}">
                [<span class="flex-grow">${kind}</span>]
            </span>
            <span class="text-xs font-semibold text-slate-400">&lt;${function_name}&gt;</span>
            <pre class="message__content font-mono">${message.trim()}</pre>
        </li>
    `;
}
