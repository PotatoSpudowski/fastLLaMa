import type { FileStructure, InitAck, WebSocketMessage } from '@/model/schema';
import { defineStore } from 'pinia';
import useSocketStore from './socketStore';
import { notifyError, notifyInfo, notifySuccess, notifyWarning } from '@/lib/notification';

const useAppStore = defineStore('useAppStore', {
    state: () => ({
        currentPath: '',
        files: [] as FileStructure[],
        commands: [] as InitAck['commands'],
        saveHistory: [] as InitAck['saveHistory'],
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
                this.commands = data.commands;
                this.files = data.files;
                this.currentPath = data.currentPath;
                this.saveHistory = data.saveHistory;
            } else if (data.type === 'file-manager-ack') {
                this.files = data.files;
                this.currentPath = data.currentPath;
            } else if (data.type === 'error-notification') {
                notifyError(data.message);
            } else if (data.type === 'warning-notification') {
                notifyWarning(data.message);
            } else if (data.type === 'info-notification') {
                notifyInfo(data.message);
            } else if (data.type === 'success-notification') {
                notifySuccess(data.message);
            } else if (data.type === 'session-list-ack') {
                this.saveHistory = data.sessions;
            }
        },
    }
});

export default useAppStore;
