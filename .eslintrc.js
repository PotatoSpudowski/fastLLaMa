/* eslint-env node */
require('@rushstack/eslint-patch/modern-module-resolution')

module.exports = {
    root: true,
    'extends': [
        'plugin:vue/vue3-essential',
        'eslint:recommended',
        '@vue/eslint-config-typescript',
        '@vue/eslint-config-prettier/skip-formatting'
    ],
    parserOptions: {
        ecmaVersion: 'latest'
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
        'no-console': process.env.NODE_ENV === 'production' ? 'warn' : 'off',
        'vue/no-deprecated-slot-attribute': 'off',
    }
}
