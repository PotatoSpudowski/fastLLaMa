import { webSocketMessageSchema, type WebSocketMessage } from '@/model/schema';
import { defineStore } from 'pinia';
import useAppStore from './appStore';

export type SocketCallback = (this: WebSocket, data: WebSocketMessage) => void;

type SocketStoreState = {
    socket: WebSocket | null;
    address: string;
    isConnected: boolean;
    isConnecting: boolean;
    errorMessage: string | null;
    hasError: boolean;
    _callbacks: SocketCallback[];
};

const useSocketStore = defineStore('useSocketStore', {
    state: (): SocketStoreState => ({
        socket: null,
        address: 'ws://localhost:8000/ws',
        isConnected: false,
        isConnecting: false,
        errorMessage: null,
        hasError: false,
        _callbacks: [],
    }),
    actions: {
        connect(callback?: (err?: any) => void) {
            this.isConnecting = true;
            this.errorMessage = null;
            this.hasError = false;
            this.socket = new WebSocket(this.address);
            this.socket.onopen = () => {
                this.isConnected = true;
                this.isConnecting = false;
                callback?.();
            };
            this.socket.onclose = (e) => {
                this.isConnected = false;
                this.isConnecting = false;
                if (e.wasClean) return;
                this.hasError = true;
                if (e.reason) this.errorMessage = e.reason;
            };

            this.socket.onerror = () => {
                this.isConnecting = false;
                this.hasError = true;
                this.errorMessage = 'Error occurred while connecting to the WebSocket server.';
                callback?.(this.errorMessage);
            }
            const callbacks = this._callbacks;
            this.socket!.onmessage = function (event) {
                try {
                    const json = JSON.parse(event.data);
                    const data: WebSocketMessage = webSocketMessageSchema.parse(json);
                    const self = this;
                    useAppStore()._handleWebsocketMessage(data);
                    callbacks.forEach((callback) => {
                        callback.call(self, data);
                    });
                } catch (e) {
                    console.error(e);
                }
            };
        },

        on(callback: SocketCallback) {
            this._callbacks.push(callback);
        },
        off(callback: SocketCallback) {
            const idx = this._callbacks.indexOf(callback);
            if (idx === -1) return;
            this._callbacks.splice(idx, 1);
        },
        disconnect() {
            if (this.socket) this.socket.close();
            this._callbacks = [];
            this.isConnected = false;
            this.isConnecting = false;
            this.errorMessage = null;
            this.hasError = false;
        },
        message(message: WebSocketMessage) {
            if (!this.socket) return;
            this.socket.send(JSON.stringify(message));
        },
    }
});

export default useSocketStore;
