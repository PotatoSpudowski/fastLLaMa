export class InputHistory {
    private _history: string[];
    private _maxHistoryLength = 100;
    private _top = 0;
    private _bottom = 0;
    private _iterCurr = 0;

    constructor(maxHistoryLength = 100) {
        this._maxHistoryLength = maxHistoryLength;
        this._history = new Array<string>(this._maxHistoryLength);
    }

    get history() {
        return this._history;
    }

    get maxHistoryLength() {
        return this._maxHistoryLength;
    }

    get size() {
        return this._top > this._bottom ? this._top - this._bottom : this._maxHistoryLength - this._bottom + this._top;
    }

    addHistory(input: string) {
        this._history[this._top] = input;
        this._top = (this._top + 1) % this._maxHistoryLength;
        if (this._top == this._bottom) {
            this._bottom = (this._bottom + 1) % this._maxHistoryLength;
        }
        this._history[this._top] = input;
        this._iterCurr = this._top;
    }

    getPreviousHistory() {
        if (this._iterCurr == this._bottom) return this._history[this._iterCurr];
        this._iterCurr = (this._maxHistoryLength + this._iterCurr - 1) % this._maxHistoryLength;
        if (this._iterCurr == this._bottom) return this._history[this._iterCurr];
        return this._history[this._iterCurr];
    }

    getNextHistory() {
        if (this._iterCurr == this._top) return undefined;
        this._iterCurr = (this._iterCurr + 1) % this._maxHistoryLength;
        if (this._iterCurr == this._top) return undefined;
        return this._history[this._iterCurr];
    }
}
