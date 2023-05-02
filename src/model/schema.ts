import { z } from 'zod';

export const packetSchema = z.object({
    id: z.string(),
    type: z.literal('join'),
}).or(z.object({
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
    n_load_parallel_blocks: z.number().optional(),
})).or(z.object({
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
