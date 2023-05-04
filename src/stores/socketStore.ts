import { defineStore } from 'pinia';

export type SocketCallback<T = any> = (this: WebSocket, data: T) => void;

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
        address: 'ws://localhost:8080',
        isConnected: false,
        isConnecting: false,
        errorMessage: null,
        hasError: false,
        _callbacks: [],
    }),
    actions: {
        connect() {
            this.isConnecting = true;
            this.errorMessage = null;
            this.hasError = false;
            this.socket = new WebSocket(this.address);
            this.socket.onopen = () => {
                this.isConnected = true;
                this.isConnecting = false;
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
            }
            const callbacks = this._callbacks;
            this.socket!.onmessage = function (event) {
                const data = JSON.parse(event.data);
                const self = this;
                callbacks.forEach((callback) => {
                    callback.apply(self, data);
                });
            };
        },

        on<T>(callback: SocketCallback<T>) {
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
        }
    }
});

export default useSocketStore;
