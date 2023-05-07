type Init = {
    type: 'init',
    version: string,
}

type InitAck = {
    type: 'init-ack',
    currentPath: string,
    files: Array<{
        type: 'file' | 'directory',
        name: string,
        path: string,
    }>
    saveHistory: Array<{
        id: string | number,
        title: string,
        date: number,
    }>,
}

type Close = {
    type: 'close',
}

type SessionLoad = {
    type: 'load-save',
    id: string,
}
type SessionLoadAck = {
    type: 'load-save-ack',
    kind: 'success' | 'failure',
}

type SessionSave = {
    type: 'session-save',
}

type SessionSaveAck = {
    type: 'session-save-ack',
    kind: 'success' | 'failure',
}

type SystemMessageProgress = {
    id: string,
    type: 'system',
    kind: 'progress',
    function_name: string,
    message: string,
    progress: number,
}

type SystemMessage = {
    id: string,
    type: 'system',
    kind: 'info' | 'warning' | 'error',
    function_name: string,
    message: string,
} | SystemMessageProgress

type MessageStatusKind = 'loading' | 'progress' | 'success' | 'failure';

type MessageStatus = {
    kind: Exclude<MessageStatusKind, 'progress'>,
} | {
    kind: 'progress',
    progress: number,
}

type ConversationMessage = {
    id: string,
    type: 'user' | 'model',
    title: string,
    message: string,
    status: MessageStatus,
}

type MessageAck = {
    type: 'message-ack',
    kind: 'success' | 'failure',
}

type Message = ConversationMessage | SystemMessage

type SaveHistoryItem = {
    id: string,
    title: string,
    date: number,
}

type FileTraversal = {
    type: 'file-traversal',
    kind: 'go-back' | 'open-directory',
    path: string,
}

type FileTraversalAck = {
    type: 'file-traversal-ack',
    currentPath: string,
    files: InitAck['files'],
}

type LoadModel = {
    type: 'load-model',
    path: string,
    params: {
        n_threads?: number,
        n_ctx?: number,
        last_n_size?: number,
        seed?: number,
        tokens_to_keep?: number,
        n_batch?: number,
        use_mmap?: boolean,
        use_mlock?: boolean,
        load_parallel?: boolean,
        n_load_parallel_blocks?: number
    }
}

type Command = {
    type: 'command',
    command: string,
    args: Array<{
        name: string,
        value: string | boolean | number,
    }>,
}
