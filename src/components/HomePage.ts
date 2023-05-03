import { html } from 'lit-html';
import { FileStructure } from '../model/schema';
import { FAST_LLAMA_WATERMARK } from '../lib/watermark';

function onSelectModelClick() {
    const dialog = document.getElementById('file-dialog');
    const welcome = document.getElementById('welcome-llama');

    requestAnimationFrame(() => {
        if (!dialog || !welcome) return;

        const dAnimation = dialog.animate([
            { opacity: 0, pointerEvents: 'none', transform: 'translateX(-100%)' },
            { opacity: 1, pointerEvents: 'all', transform: 'translateX(0)' }
        ], {
            duration: 150,
            easing: 'ease-in-out',
            fill: 'forwards'
        });

        const wAnimation = welcome.animate([
            { opacity: 1, pointerEvents: 'all', transform: 'translateX(0)' },
            { opacity: 0, pointerEvents: 'none', transform: 'translateX(100%)' }
        ], {
            duration: 150,
            easing: 'ease-in-out',
            fill: 'forwards'
        });

        wAnimation.currentTime = dAnimation.currentTime;

        dAnimation.onfinish = () => {
            dialog.classList.remove('opacity-0');
            dialog.classList.remove('pointer-events-none');
            welcome.classList.add('opacity-0');
            welcome.classList.add('pointer-events-none');
            dAnimation.cancel();
            wAnimation.cancel();
        };
    });
}

function onDialogClose() {
    const dialog = document.getElementById('file-dialog');
    const welcome = document.getElementById('welcome-llama');

    requestAnimationFrame(() => {
        if (!dialog || !welcome) return;

        const dAnimation = dialog.animate([
            { opacity: 1, pointerEvents: 'all', transform: 'translateX(0)' },
            { opacity: 0, pointerEvents: 'none', transform: 'translateX(-100%)' }
        ], {
            duration: 150,
            easing: 'ease-in-out',
            fill: 'forwards'
        });

        const wAnimation = welcome.animate([
            { opacity: 0, pointerEvents: 'none', transform: 'translateX(100%)' },
            { opacity: 1, pointerEvents: 'all', transform: 'translateX(0)' }
        ], {
            duration: 150,
            easing: 'ease-in-out',
            fill: 'forwards'
        });

        wAnimation.currentTime = dAnimation.currentTime;

        dAnimation.onfinish = () => {
            dialog.classList.add('opacity-0');
            dialog.classList.add('pointer-events-none');
            welcome.classList.remove('opacity-0');
            welcome.classList.remove('pointer-events-none');
            dAnimation.cancel();
            wAnimation.cancel();
        };
    });
}


function onFileDBClick(file: FileStructure) {
    if (file.type !== 'directory') return;
    console.log('open directory', file);
}

function FileDialog(filepath: string, files: FileStructure[]) {
    const dir_right_icon = html`<ion-icon aria-hidden="true" name="chevron-forward-outline"></ion-icon>`;
    let selectedFile: FileStructure | null = null;

    const onFileItemSelect = (file: FileStructure) => {
        const btn = document.getElementById('file-dialog-confirm-btn') as HTMLButtonElement;
        if (file.type === 'directory') {
            btn.disabled = true;
        } else {
            btn.disabled = false;
            selectedFile = file;
        }
    };

    const onConfirmFile = () => {
        if (!selectedFile) return;
        document.dispatchEvent(new CustomEvent('file-select', { detail: selectedFile }));
    };

    const fileElements = files.map(file => {
        return html`
            <sp-menu-item @click="${() => onFileItemSelect(file)}" @dblclick="${() => onFileDBClick(file)}">
                ${file.type === 'directory' ? html`<ion-icon slot="icon" name="folder"></ion-icon>` : html`<ion-icon slot="icon" name="document"></ion-icon>`}
                <div class="flex items-center gap-2 w-full justify-between">
                    <span>${file.name}</span>
                    ${file.type === 'directory' ? dir_right_icon : html``}
                </div>
            </sp-menu-item>
        `;
    });
    return html`
        <sp-dialog id="file-dialog" class="w-[25rem] opacity-0 pointer-events-none" no-divider class="" dismissable @close="${onDialogClose}" style="background-color: var(--spectrum-gray-75)">
            <div slot="heading">
                <h2 class="mb-2">Select a Modal</h2>
                <sp-search value="${filepath}" class="w-[15rem]"></sp-search>
            </div>

            <div class="max-h-[10rem] overflow-y-auto">
                <sp-menu class="w-full" selects="single">
                    ${fileElements}
                </sp-menu>
            </div>

            <div class="flex flex-col">
                <sp-button-group class="min-w-fit mt-2">
                    <sp-button variant="secondary" treatment="outline">Go Back</sp-button>
                    <sp-button static="white" id="file-dialog-confirm-btn" disabled @click="${onConfirmFile}">Confirm</sp-button>
                </sp-button-group>
            </div>
        </sp-dialog>
    `
}

function watermarkTemplate() {
    return html` <pre aria-hidden="true" id="water" class="fastllama-watermark">${FAST_LLAMA_WATERMARK}</pre>`
}

export function HomePage(filepath: string, files: FileStructure[]) {
    return html`
        <div class="grid items-center justify-center">
            <div class="flex flex-col items-center justify-center" id="welcome-llama" style="grid-area: 1/1">
                <img src="img/image.png" width="400" height="300" />
                <sp-button-group class="min-w-fit my-2">
                    <sp-button static="white" @click="${onSelectModelClick}">Select Modal</sp-button>
                </sp-button-group>
            </div>
            <div style="grid-area: 1 / 1;">
                ${FileDialog(filepath, files)}
            </div>
        </div>
    `
}

declare global {
    interface DocumentEventMap {
        'file-select': CustomEvent<FileStructure>;
    }
}
