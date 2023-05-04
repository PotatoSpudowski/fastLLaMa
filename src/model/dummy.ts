import type { Token } from '@/lib/parser';
import type { FileStructure, Message, SaveHistoryItem } from './schema';

export const dummyMessages: Message[] = [
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'warning',
        function_name: 'Model',
        message: 'loading model from ./models/VICUNA-7B/ggml-vicuna-7b-1.0-uncensored-q4_2.bin - please wait ...'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'error',
        function_name: 'Model',
        message: 'n_vocab    = 32001'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_ctx      = 2048'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_embd     = 4096'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_mult     = 256'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_head     = 32'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_layer    = 32'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_rot      = 128'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'ftype      = 5 (mostly Q4_2)'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_ff       = 11008'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_parts    = 1'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'model_id   = 1'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'KVCacheBuffer::init',
        message: 'kv self size  = 2.00 GiB'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'ggml ctx size = 59.11 KiB'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'mem required  = 7.17 GiB (+ 2.00 GiB per state)'
    },
    {
        id: new Date().toISOString(),
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'time to load all data = 1334.79 ms'
    },
    {
        id: new Date().toISOString(),
        type: 'user',
        title: 'John Doe',
        message: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Quisquam, voluptatum.',
        status: {
            kind: 'success',
        }
    },
    {
        id: new Date().toISOString(),
        type: 'model',
        title: 'Alpaca',
        message: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Quisquam, voluptatum.',
        status: {
            kind: 'success',
        }
    },
];

export const dummyFiles: FileStructure[] = [
    {
        name: 'root',
        type: 'directory',
        path: 'root/root',
    },
    {
        name: 'file1.txt',
        type: 'file',
        path: 'root/file1.txt',
    },
    {
        name: 'file2.txt',
        type: 'file',
        path: 'root/file2.txt',
    }
];


export const dummyHistory: SaveHistoryItem[] = [
    {
        id: new Date().toISOString(),
        title: 'John Doe 1',
        date: Date.now()
    },
    {
        id: new Date().toISOString(),
        title: 'John Doe 2',
        date: Date.now()
    },
    {
        id: new Date().toISOString(),
        title: 'John Doe 3',
        date: Date.now() - 1000 * 60 * 60 * 24
    },
    {
        id: new Date().toISOString(),
        title: 'Sint id dolor anim sint nostrud magna deserunt ullamco sint ex ex commodo dolore labore. Cillum qui ea quis reprehenderit velit sunt laboris dolor reprehenderit. Mollit nisi dolor cupidatat officia Lorem adipisicing et deserunt enim ut qui fugiat amet occaecat.',
        date: Date.now() - 1000 * 60 * 60 * 24 * 2
    }
];

export const dummyCommands: Record<string, Token[]> = {
    load: [
        {
            type: 'string',
            value: 'lora'
        },
        {
            type: 'arg',
            name: 'n_thread',
            value: '',
        }
    ]
};
