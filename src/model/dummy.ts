export type SystemMessage = {
    type: 'system',
    kind: 'info' | 'warning' | 'error',
    function_name: string,
    message: string
}

export type Message = {
    type: 'user' | 'model',
    title: string,
    message: string
} | SystemMessage

export const dummyMessages: Message[] = [
    {
        type: 'system',
        kind: 'warning',
        function_name: 'Model',
        message: 'loading model from ./models/VICUNA-7B/ggml-vicuna-7b-1.0-uncensored-q4_2.bin - please wait ...'
    },
    {
        type: 'system',
        kind: 'error',
        function_name: 'Model',
        message: 'n_vocab    = 32001'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_ctx      = 2048'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_embd     = 4096'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_mult     = 256'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_head     = 32'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_layer    = 32'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_rot      = 128'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'ftype      = 5 (mostly Q4_2)'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_ff       = 11008'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'n_parts    = 1'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'model_id   = 1'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'KVCacheBuffer::init',
        message: 'kv self size  = 2.00 GiB'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'ggml ctx size = 59.11 KiB'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'mem required  = 7.17 GiB (+ 2.00 GiB per state)'
    },
    {
        type: 'system',
        kind: 'info',
        function_name: 'Model',
        message: 'time to load all data = 1334.79 ms'
    },
    {
        type: 'user',
        title: 'John Doe',
        message: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Quisquam, voluptatum.'
    },
    {
        type: 'model',
        title: 'Alpaca',
        message: 'Lorem ipsum dolor sit amet consectetur adipisicing elit. Quisquam, voluptatum.'
    },
]
