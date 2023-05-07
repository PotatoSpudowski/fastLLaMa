export type NotificationConfig = {
    message: string,
    type: 'success' | 'error' | 'warning' | 'info',
    duration?: number,
};

function flipNotification(container: HTMLDivElement, notification: HTMLElement) {
    const first = container.offsetHeight;

    container.appendChild(notification);

    const last = container.offsetHeight;

    const delta = last - first;

    container.animate([
        { transform: `translate(-50%, -${delta}px)` },
        { transform: 'translateY(-50%, 0)' },
    ], {
        duration: 150,
        easing: 'ease-out',
    });
};

function removeNotification(container: HTMLDivElement, notification: HTMLElement) {
    container.removeChild(notification);
}

export function showNotification(config: NotificationConfig) {
    const container = document.getElementById('alert-container') as HTMLDivElement;
    if (!container) return;

    const { message, type, duration = 3000 } = config;
    const toast = document.createElement('sp-toast');
    toast.append(message);
    toast.setAttribute('open', '');
    toast.style.setProperty('--on-screen-duration', `${duration}ms`);
    toast.classList.add('fade-animation');
    toast.onclose = () => removeNotification(container, toast);
    switch (type) {
        case 'success':
            toast.setAttribute('variant', 'positive');
            break;
        case 'error':
            toast.setAttribute('variant', 'negative');
            break;
        case 'warning':
            toast.setAttribute('variant', 'warning');
            break;
        case 'info':
            toast.setAttribute('variant', 'info');
            break;
    }

    requestAnimationFrame(async () => {
        flipNotification(container, toast);

        await Promise.allSettled(toast.getAnimations().map(a => a.finished));

        removeNotification(container, toast);
    });
}

export function notifyError(message: string, duration?: number) {
    showNotification({ message, type: 'error', duration });
}

export function notifySuccess(message: string, duration?: number) {
    showNotification({ message, type: 'success', duration });
}

export function notifyWarning(message: string, duration?: number) {
    showNotification({ message, type: 'warning', duration });
}

export function notifyInfo(message: string, duration?: number) {
    showNotification({ message, type: 'info', duration });
}

