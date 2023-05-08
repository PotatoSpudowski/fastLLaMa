import { z } from 'zod';

export const modelParamsSchema = z.object({
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

export const modelInitCompleteSchema = z.object({
    type: z.literal('model-init-complete'),
});

export const fileSchema = z.object({
    kind: z.enum(['file', 'directory']),
    name: z.string(),
    path: z.string(),
});

export const supportedCommandAckSchema = z.object({
    type: z.literal('supported-command-ack'),
    command: z.string(),
    args: z.object({
        name: z.string(),
        type: z.enum(['string', 'float', 'int', 'boolean']),
        value: z.string().or(z.number()).or(z.boolean()).optional().nullable(),
    }).array()
});

export const supportedCommandSchema = z.object({
    type: z.literal('supported-command'),
});

export const invokeCommandSchema = z.object({
    type: z.literal('invoke-command'),
    command: z.string(),
    args: supportedCommandAckSchema.shape.args
});

export const initSchema = z.object({
    type: z.literal('init'),
    version: z.string(),
});

export const saveHistorySchema = z.object({
    id: z.string(),
    title: z.string(),
    date: z.number(),
    model_path: z.string(),
    model_args: modelParamsSchema.omit({ type: true, model_path: true }),
});

export const initAckSchema = z.object({
    type: z.literal('init-ack'),
    currentPath: z.string(),
    files: fileSchema.array(),
    saveHistory: saveHistorySchema.array(),
    commands: z.object({
        name: z.string(),
        args: supportedCommandAckSchema.shape.args
    }).array()
});

export const closeSchema = z.object({
    type: z.literal('close'),
});

export const sessionLoadSchema = z.object({
    type: z.literal('session-load'),
    id: z.string().or(z.number()),
});

export const sessionSaveSchema = z.object({
    type: z.literal('session-save'),
});

export const sessionDeleteSchema = z.object({
    type: z.literal('session-delete'),
    id: z.string().or(z.number()),
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
    kind: z.enum(['go-back', 'open-dir'])
});

export const fileManagerAckSchema = z.object({
    type: z.literal('file-manager-ack'),
    currentPath: z.string(),
    files: fileSchema.array(),
});

export const notificationSchema = z.object({
    type: z.enum(['error-notification', 'success-notification', 'warning-notification', 'info-notification']),
    message: z.string(),
});

export const resetMessageSchema = z.object({
    type: z.literal('reset-messages'),
    messages: conversationMessageSchema.or(systemMessageSchema).array(),
});

export type FileStructure = z.infer<typeof fileSchema>;

export type SystemMessageProgress = z.infer<typeof systemMessageProgressSchema>;

export type SystemMessage = z.infer<typeof systemMessageSchema>;

export type MessageStatusKind = z.infer<typeof conversationMessageStatus>['kind'];

export type MessageStatus = z.infer<typeof conversationMessageStatus>;

export type ConversationMessage = z.infer<typeof conversationMessageSchema>;

export type Message = ConversationMessage | SystemMessage

export type SaveHistoryItem = z.infer<typeof saveHistorySchema>;

export type InitAck = z.infer<typeof initAckSchema>;

export type Command = Omit<z.infer<typeof supportedCommandAckSchema>, 'type'>;

export const webSocketMessageSchema = z.union([
    initSchema,
    initAckSchema,
    closeSchema,
    sessionLoadSchema,
    sessionSaveSchema,
    sessionDeleteSchema,
    sessionListSchema,
    sessionListAckSchema,
    supportedCommandSchema,
    supportedCommandAckSchema,
    invokeCommandSchema,
    systemMessageSchema,
    messageAckSchema,
    fileManagerSchema,
    fileManagerAckSchema,
    conversationMessageSchema,
    notificationSchema,
    modelInitCompleteSchema,
    resetMessageSchema
]);
export type WebSocketMessage = z.infer<typeof webSocketMessageSchema>;
