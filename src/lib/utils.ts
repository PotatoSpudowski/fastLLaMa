export function pause(milliseconds: number) {
    return new Promise(resolve => setTimeout(resolve, milliseconds))
}

export function deepCopy<T>(obj: T): T {
    return JSON.parse(JSON.stringify(obj))
}
