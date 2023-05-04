export type Token = {
    type: 'cmd' | 'string';
    value: string;
} | {
    type: 'arg';
    name: string;
    value: string;
}

// command: "load '/models/lora'"
// command: "save path='/models/lora' format='json'"
// command: "save path='/models/lora' format='json' pretty=true test"
//           [{type: 'cmd', value: 'save'}, {type: 'arg', name: 'path', value: '/models/lora'}, {type: 'arg', name: 'format', value: 'json'}, {type: 'arg', name: 'pretty', value: 'true'}, {type: 'string', value: 'test'}]
export function parseCommand(command: string): Array<Token> | undefined {
    const res: Token[] = [];
    if (!command.startsWith('/')) return;

    const tokens = command.split(' ');

    res.push({ type: 'cmd', value: tokens[0].substring(1) });

    for (let i = 1; i < tokens.length; i++) {
        const token = tokens[i];
        if (token.startsWith("'") || token.startsWith('"')) {
            const temp = token.slice(1, -1);
            res.push({ type: 'string', value: temp });
        } else if (token.includes('=')) {
            const [name, value] = token.split('=');
            if (value.startsWith("'") || value.startsWith('"')) {
                res.push({ type: 'arg', name, value: value.slice(1, -1) });
            } else {
                res.push({ type: 'arg', name, value });
            }
        } else {
            res.push({ type: 'string', value: token });
        }
    }
    return res;
}
