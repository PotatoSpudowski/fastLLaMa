module.exports = {
    env: {
        browser: true,
        es2021: true
    },
    extends: 'standard-with-typescript',
    overrides: [
    ],
    parserOptions: {
        ecmaVersion: 'latest',
        sourceType: 'module'
    },
    rules: {
        indent: ['error', 4, {
            SwitchCase: 1
        }],
        quotes: ['warn', 'single', {
            avoidEscape: true
        }],
        '@typescript-eslint/no-var-requires': 'off',
        'no-unused-vars': 'off',
        'no-debugger': process.env.NODE_ENV === 'production' ? 'error' : 'off',
        'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off'
    }
}
