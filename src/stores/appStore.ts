import type { FileStructure, InitAck, WebSocketMessage } from '@/model/schema';
import { defineStore } from 'pinia';
import useSocketStore from './socketStore';
import { notifyError } from '@/lib/notification';

const useAppStore = defineStore('useAppStore', {
    state: () => ({
        currentPath: '',
        files: [] as FileStructure[],
        commands: [] as InitAck['commands'],
    }),

    actions: {
        openDir(path: string) {
            useSocketStore().message({
                type: 'file-manager',
                path,
                kind: 'open-dir',
            });
        },

        goBack() {
            useSocketStore().message({
                type: 'file-manager',
                path: this.currentPath,
                kind: 'go-back',
            });
        },

        _handleWebsocketMessage(data: WebSocketMessage) {
            if (data.type === 'init-ack') {
                useAppStore().commands = data.commands;
                useAppStore().files = data.files;
                useAppStore().currentPath = data.currentPath;
            } else if (data.type === 'file-manager-ack') {
                useAppStore().files = data.files;
                useAppStore().currentPath = data.currentPath;
            } else if (data.type === 'error') {
                notifyError(data.message);
            } else if (data.type === 'system-message') {
                // console.log(data);
            }
        },
    }
});

export default useAppStore;
