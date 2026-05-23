/**
 * TabManager - Handles tab switching between Interaction Logs and Dialog Window
 * Preserves content state when switching between tabs
 */

class TabManager {
    constructor() {
        this.activeTab = 'logs'; // Default to logs tab
        this.tabs = {
            logs: {
                tabButton: null,
                panel: null,
                initialized: true // Logs panel is always initialized
            },
            dialog: {
                tabButton: null,
                panel: null,
                initialized: false
            }
        };

        // Chat manager instance for dialog tab
        this.chatManager = null;

        this.initializeElements();
        this.setupEventListeners();
        this.initializeDefaultTab();
    }

    initializeElements() {
        // Tab buttons
        this.tabs.logs.tabButton = document.getElementById('tab-logs');
        this.tabs.dialog.tabButton = document.getElementById('tab-dialog');

        // Panels
        this.tabs.logs.panel = document.getElementById('logs-panel');
        this.tabs.dialog.panel = document.getElementById('dialog-panel');

        // Validate elements exist
        for (const [tabName, tab] of Object.entries(this.tabs)) {
            if (!tab.tabButton) {
                console.warn(`Tab button not found for tab: ${tabName}`);
            }
            if (!tab.panel) {
                console.warn(`Panel not found for tab: ${tabName}`);
            }
        }
    }

    setupEventListeners() {
        // Tab button event listeners
        if (this.tabs.logs.tabButton) {
            this.tabs.logs.tabButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchToTab('logs');
            });
        }

        if (this.tabs.dialog.tabButton) {
            this.tabs.dialog.tabButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.switchToTab('dialog');
            });
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch (e.key) {
                    case '1':
                        e.preventDefault();
                        this.switchToTab('logs');
                        break;
                    case '2':
                        e.preventDefault();
                        this.switchToTab('dialog');
                        break;
                }
            }
        });

        // Handle browser navigation (back/forward)
        window.addEventListener('popstate', (e) => {
            if (e.state && e.state.activeTab) {
                this.switchToTab(e.state.activeTab, false); // Don't push state again
            }
        });
    }

    initializeDefaultTab() {
        // Check URL fragment for initial tab
        const hash = window.location.hash.substring(1);
        if (hash === 'dialog' && this.tabs.dialog.tabButton && this.tabs.dialog.panel) {
            this.switchToTab('dialog');
        } else {
            this.switchToTab('logs');
        }

        // Update browser history with initial state
        history.replaceState({ activeTab: this.activeTab }, '',
            `${window.location.pathname}${window.location.search}#${this.activeTab}`);
    }

    async switchToTab(tabName, updateHistory = true) {
        if (!this.tabs[tabName]) {
            console.error(`Unknown tab: ${tabName}`);
            return;
        }

        if (this.activeTab === tabName) {
            return; // Already active
        }

        const previousTab = this.activeTab;
        this.activeTab = tabName;

        console.log(`Switching from ${previousTab} to ${tabName}`);

        try {
            // Hide previous tab
            this.hideTab(previousTab);

            // Initialize new tab if needed
            await this.initializeTab(tabName);

            // Show new tab
            this.showTab(tabName);

            // Update URL and history
            if (updateHistory) {
                const newUrl = `${window.location.pathname}${window.location.search}#${tabName}`;
                history.pushState({ activeTab: tabName }, '', newUrl);
            }

            // Trigger tab change event
            this.triggerTabChangeEvent(previousTab, tabName);

        } catch (error) {
            console.error(`Failed to switch to tab ${tabName}:`, error);

            // Revert to previous tab on error
            this.activeTab = previousTab;
            this.showTab(previousTab);

            // Show error message
            this.showErrorMessage(`Failed to switch to ${tabName} tab: ${error.message}`);
        }
    }

    hideTab(tabName) {
        const tab = this.tabs[tabName];
        if (!tab) return;

        // Hide panel using CSS class
        if (tab.panel) {
            tab.panel.classList.add('hidden');
        }

        // Remove active class from button
        if (tab.tabButton) {
            tab.tabButton.classList.remove('active');
        }

        console.log(`Hidden tab: ${tabName}`);
    }

    showTab(tabName) {
        const tab = this.tabs[tabName];
        if (!tab) return;

        // Show panel by removing hidden class
        if (tab.panel) {
            tab.panel.classList.remove('hidden');
            console.log(`Showed panel for tab: ${tabName}, hidden class removed`);
        }

        // Add active class to button
        if (tab.tabButton) {
            tab.tabButton.classList.add('active');
            console.log(`Activated button for tab: ${tabName}`);
        }

        // Focus handling
        this.focusTab(tabName);

        console.log(`Shown tab: ${tabName}`);
    }

    async initializeTab(tabName) {
        const tab = this.tabs[tabName];
        if (!tab || tab.initialized) {
            return; // Already initialized
        }

        console.log(`Initializing tab: ${tabName}`);

        switch (tabName) {
            case 'dialog':
                await this.initializeDialogTab();
                break;

            case 'logs':
                // Logs tab is always initialized
                break;

            default:
                console.warn(`No initialization logic for tab: ${tabName}`);
        }

        tab.initialized = true;
        console.log(`Tab initialized: ${tabName}`);
    }

    async initializeDialogTab() {
        console.log('Initializing dialog tab...');

        if (this.chatManager) {
            console.log('Dialog tab already initialized');
            return; // Already initialized
        }

        try {
            console.log('Creating ChatManager...');
            this.chatManager = new ChatManager();

            console.log('Initializing ChatManager...');
            await this.chatManager.initialize();

            // Store reference for external access
            window.chatManager = this.chatManager;

            console.log('Dialog tab initialized successfully');

        } catch (error) {
            console.error('Failed to initialize dialog tab:', error);

            // Create a basic chat manager even on error so UI shows status
            if (!this.chatManager) {
                console.log('Creating fallback ChatManager...');
                this.chatManager = new ChatManager();
                window.chatManager = this.chatManager;
            }

            console.log('Dialog tab initialized with basic ChatManager for status display');
        }
    }

    focusTab(tabName) {
        switch (tabName) {
            case 'dialog':
                // Focus on input field if available and chat manager is ready
                setTimeout(() => {
                    const userInput = document.getElementById('user-input');
                    if (userInput && !userInput.disabled) {
                        userInput.focus();
                    }
                }, 100);
                break;

            case 'logs':
                // Scroll to bottom of logs
                const logwrap = document.getElementById('logwrap');
                if (logwrap) {
                    logwrap.scrollTop = logwrap.scrollHeight;
                }
                break;
        }
    }

    triggerTabChangeEvent(fromTab, toTab) {
        const event = new CustomEvent('tabchange', {
            detail: {
                from: fromTab,
                to: toTab,
                timestamp: Date.now()
            }
        });

        document.dispatchEvent(event);

        // Also trigger on window for global listeners
        window.dispatchEvent(new CustomEvent('tabchange', {
            detail: event.detail
        }));

        console.log(`Tab change event: ${fromTab} -> ${toTab}`);
    }

    showErrorMessage(message) {
        // Create error notification
        const errorDiv = document.createElement('div');
        errorDiv.className = 'tab-error-message';
        errorDiv.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #7f1d1d;
            color: #fca5a5;
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid #b91c1c;
            z-index: 1000;
            max-width: 300px;
        `;
        errorDiv.textContent = message;

        document.body.appendChild(errorDiv);

        // Remove after 5 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.parentNode.removeChild(errorDiv);
            }
        }, 5000);
    }

    // Public API

    getActiveTab() {
        return this.activeTab;
    }

    isTabAvailable(tabName) {
        return !!this.tabs[tabName];
    }

    isTabInitialized(tabName) {
        return this.tabs[tabName]?.initialized || false;
    }

    async refreshActiveTab() {
        const currentTab = this.activeTab;

        switch (currentTab) {
            case 'dialog':
                if (this.chatManager) {
                    // Refresh connection status
                    await this.chatManager.checkManualModeStatus();
                }
                break;

            case 'logs':
                // Logs are automatically updated via SSE, no manual refresh needed
                break;
        }
    }

    getChatManager() {
        return this.chatManager;
    }

    // Tab state preservation methods

    preserveScrollPosition(tabName) {
        const tab = this.tabs[tabName];
        if (!tab?.panel) return;

        const scrollableElement = this.getScrollableElement(tabName);
        if (scrollableElement) {
            tab.scrollPosition = scrollableElement.scrollTop;
        }
    }

    restoreScrollPosition(tabName) {
        const tab = this.tabs[tabName];
        if (!tab?.panel || tab.scrollPosition === undefined) return;

        const scrollableElement = this.getScrollableElement(tabName);
        if (scrollableElement) {
            scrollableElement.scrollTop = tab.scrollPosition;
        }
    }

    getScrollableElement(tabName) {
        switch (tabName) {
            case 'logs':
                return document.getElementById('logwrap');
            case 'dialog':
                return document.getElementById('chat-messages');
            default:
                return null;
        }
    }

    // Cleanup method for proper resource management
    destroy() {
        // Clean up chat manager
        if (this.chatManager) {
            this.chatManager.stopSession();
            this.chatManager = null;
            window.chatManager = null;
        }

        // Remove event listeners
        document.removeEventListener('keydown', this.keydownHandler);
        window.removeEventListener('popstate', this.popstateHandler);

        console.log('TabManager destroyed');
    }
}

// Initialize tab manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (!window.tabManager) {
        window.tabManager = new TabManager();
        console.log('TabManager initialized');
    }
});

// Export for external use
window.TabManager = TabManager;