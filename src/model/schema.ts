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
});

export const fileSchema = z.object({
    type: z.enum(['file', 'directory']),
    name: z.string(),
    path: z.string(),
});

export const commandAckSchema = z.object({
    type: z.literal('command-ack'),
    command: z.string(),
    args: z.object({
        name: z.string(),
        value: z.string().or(z.number()).or(z.boolean()).optional(),
    }).array()
});

export const commandSchema = z.object({
    type: z.literal('command'),
});

export const initSchema = z.object({
    type: z.literal('init'),
    version: z.string(),
});

export const saveHistorySchema = z.object({
    id: z.string().or(z.number()),
    title: z.string(),
    date: z.number(),
});

export const initAckSchema = z.object({
    type: z.literal('init-ack'),
    currentPath: z.string(),
    files: fileSchema.array(),
    saveHistory: saveHistorySchema.array(),
    commands: commandAckSchema
});

export const closeSchema = z.object({
    type: z.literal('close'),
});

export const sessionLoadSchema = z.object({
    type: z.literal('session-load'),
    id: z.string().or(z.number()),
});

export const sessionLoadAckSchema = z.object({
    type: z.literal('session-load-ack'),
    status: z.enum(['success', 'failure']),
});

export const sessionSaveSchema = z.object({
    type: z.literal('session-save'),
});

export const sessionSaveAckSchema = z.object({
    type: z.literal('session-save-ack'),
    status: z.enum(['success', 'failure']),
    id: z.string().or(z.number()),
});

export const sessionDeleteSchema = z.object({
    type: z.literal('session-delete'),
    id: z.string().or(z.number()),
});

export const sessionDeleteAckSchema = z.object({
    type: z.literal('session-delete-ack'),
    status: z.enum(['success', 'failure']),
});

export const sessionListSchema = z.object({
    type: z.literal('session-list'),
});

export const sessionListAckSchema = z.object({
    type: z.literal('session-list-ack'),
    sessions: saveHistorySchema.array(),
});

const systemMessageBaseSchema = z.object({
    id: z.string(),
    type: z.literal('system-message'),
    message: z.string(),
    function_name: z.string(),
});

const systemMessageWithoutProgressSchema = systemMessageBaseSchema.merge(z.object({
    kind: z.enum(['info', 'warning', 'error']),
}));

export const systemMessageProgressSchema = systemMessageBaseSchema.merge(z.object({
    kind: z.literal('progress'),
    progress: z.number(),
}))

export const systemMessageSchema = systemMessageWithoutProgressSchema.or(systemMessageProgressSchema);

export const conversationMessageStatus = z.object({
    kind: z.enum(['loading', 'success', 'failure']),
}).or(z.object({
    kind: z.literal('progress'),
    progress: z.number(),
}));

export const conversationMessageSchema = z.object({
    id: z.string(),
    type: z.enum(['user-message', 'model-message']),
    webui_id: z.string().optional(), // temp id generated for user messages before they are sent to the server and assigned an id
    title: z.string(),
    message: z.string(),
    status: conversationMessageStatus,
})

export const messageAckSchema = z.object({
    type: z.literal('message-ack'),
    id: z.string(),
    webui_id: z.string(), // temp id generated for user messages before they are sent to the server and assigned an id
    status: z.enum(['success', 'failure']),
});

export const fileManagerSchema = z.object({
    type: z.literal('file-manager'),
    path: z.string(),
    kind: z.enum(['go-back', 'open-directory'])
});

export const fileManagerAckSchema = z.object({
    type: z.literal('file-manager-ack'),
    currentPath: z.string(),
    files: fileSchema.array(),
});

export type FileStructure = z.infer<typeof fileSchema>;

export type SystemMessageProgress = z.infer<typeof systemMessageProgressSchema>;

export type SystemMessage = z.infer<typeof systemMessageSchema>;

export type MessageStatusKind = z.infer<typeof conversationMessageStatus>['kind'];

export type MessageStatus = z.infer<typeof conversationMessageStatus>;

export type ConversationMessage = z.infer<typeof conversationMessageSchema>;

export type Message = ConversationMessage | SystemMessage

export type SaveHistoryItem = z.infer<typeof saveHistorySchema>;

export const webSocketMessageSchema = z.union([
    initSchema,
    initAckSchema,
    closeSchema,
    sessionLoadSchema,
    sessionLoadAckSchema,
    sessionSaveSchema,
    sessionSaveAckSchema,
    sessionDeleteSchema,
    sessionDeleteAckSchema,
    sessionListSchema,
    sessionListAckSchema,
    commandSchema,
    commandAckSchema,
    systemMessageSchema,
    messageAckSchema,
    fileManagerSchema,
    fileManagerAckSchema,
    conversationMessageSchema,
]);
export type WebSocketMessage = z.infer<typeof webSocketMessageSchema>;
