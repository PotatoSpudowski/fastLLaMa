import { z } from 'zod';

export const modelParamsSchema = z.object({
    id: z.string(),
    type: z.literal('init-model'),
    model_path: z.string(),
    n_threads: z.number().optional(),
    n_ctx: z.number().optional(),
    last_n_size: z.number().optional(),
    seed: z.number().optional(),
    tokens_to_keep: z.number().optional(),
    n_batch: z.number().optional(),
    use_mmap: z.boolean().optional(),
    use_mlock: z.boolean().optional(),
    load_parallel: z.boolean().optional(),
    n_load_parallel_blocks: z.number().optional()
})

export const packetSchema = z.object({
    id: z.string(),
    type: z.literal('join'),
}).or(modelParamsSchema).or(z.object({
    id: z.string(),
    type: z.literal('fetch-all-models'),
    search_dirs: z.string().array().optional(),
})).or(z.object({
    id: z.string(),
    type: z.literal('ingest-message'),
    message: z.string(),
    is_system_prompt: z.boolean().optional(),
})).or(z.object({
    id: z.string(),
    type: z.literal('generate-message'),
    num_of_tokens: z.number().optional(),
    temperature: z.number().optional(),
    top_p: z.number().optional(),
    top_k: z.number().optional(),
    repetition_penalty: z.number().optional(),
    stop_words: z.string().array().optional(),
})).or(z.object({
    id: z.string(),
    seq: z.number(),
    type: z.literal('generate-response'),
    message: z.string().optional(),
})).or(z.object({
    id: z.string(),
    type: z.literal('error'),
    message: z.string(),
})).or(z.object({
    id: z.string(),
    type: z.literal('save-model'),
    file_name: z.string(),
})).or(z.object({
    id: z.string(),
    type: z.literal('load-model'),
    file_name: z.string(),
}));


export const fileSchema = z.object({
    type: z.literal('file'),
    name: z.string(),
    path: z.string(),
}).or(z.object({
    type: z.literal('directory'),
    name: z.string(),
    path: z.string(),
}));

export type FileStructure = z.infer<typeof fileSchema>;

export type SystemMessageProgress = {
    id: string,
    type: 'system',
    kind: 'progress',
    function_name: string,
    message: string,
    progress: number,
}

export type SystemMessage = {
    id: string,
    type: 'system',
    kind: 'info' | 'warning' | 'error',
    function_name: string,
    message: string,
} | SystemMessageProgress

export type MessageStatusKind = 'loading' | 'progress' | 'success' | 'failure';

export type MessageStatus = {
    kind: Exclude<MessageStatusKind, 'progress'>,
} | {
    kind: 'progress',
    progress: number,
}

export type ConversationMessage = {
    id: string,
    type: 'user' | 'model',
    title: string,
    message: string,
    status: MessageStatus,
}

export type Message = ConversationMessage | SystemMessage


export type SaveHistoryItem = {
    id: string,
    title: string,
    date: number,
}
