/**
 * Error Handler Utility
 * Centralized error parsing and display for API calls
 */

/**
 * Parse API error into a readable message
 * @param {Error|Response|Object} error - The error object
 * @returns {string} Human-readable error message
 */
export const parseApiError = (error) => {
    // Network errors
    if (error instanceof TypeError && error.message.includes('fetch')) {
        return 'Backend not reachable. Please check if the server is running.';
    }

    // Already a string
    if (typeof error === 'string') {
        return error;
    }

    // Error with message property
    if (error?.message) {
        // Clean up technical error messages
        const msg = error.message;

        if (msg.includes('Failed to fetch')) {
            return 'Backend not reachable. Please check your connection.';
        }
        if (msg.includes('NetworkError')) {
            return 'Network error. Please check your internet connection.';
        }
        if (msg.includes('CORS')) {
            return 'Cross-origin request blocked. Backend may not be configured correctly.';
        }
        if (msg.includes('401')) {
            return 'Authentication required.';
        }
        if (msg.includes('403')) {
            return 'Permission denied.';
        }
        if (msg.includes('404')) {
            return 'API endpoint not found.';
        }
        if (msg.includes('500')) {
            return 'Server error. Please try again later.';
        }

        return msg;
    }

    // Response object
    if (error?.status) {
        switch (error.status) {
            case 400: return 'Invalid request. Please check your input.';
            case 401: return 'Authentication required.';
            case 403: return 'Permission denied.';
            case 404: return 'API endpoint not found.';
            case 500: return 'Server error. Please try again later.';
            case 502: return 'Backend server is starting up. Please wait and retry.';
            case 503: return 'Service temporarily unavailable.';
            default: return `Server error (${error.status})`;
        }
    }

    return 'An unexpected error occurred. Please try again.';
};

/**
 * Log error in development mode only
 * @param {string} context - Where the error occurred
 * @param {Error} error - The error object
 */
export const logError = (context, error) => {
    if (import.meta.env.DEV) {
        console.error(`[${context}]`, error);
    }
};

/**
 * Create error display data for UI
 * @param {Error} error - The error object
 * @returns {Object} Error data with message and type
 */
export const createErrorState = (error) => {
    const message = parseApiError(error);
    const isNetworkError = message.includes('Backend') || message.includes('Network') || message.includes('connection');

    return {
        message,
        type: isNetworkError ? 'network' : 'api',
        timestamp: new Date().toISOString()
    };
};
