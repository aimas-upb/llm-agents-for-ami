/**
 * ChatManager - WebSocket communication and chat interface management
 * Handles real-time communication with the AMI agent system
 */

class ChatManager {
    constructor() {
        this.ws = null;
        this.connected = false;
        this.connecting = false;
        this.messageQueue = [];
        this.activeThreadId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.manualModeActive = false;
        this.statusPollingInterval = null;

        // Message handlers
        this.messageHandlers = {
            'assistant_message': this.handleAssistantMessage.bind(this),
            'plan_proposal': this.handlePlanProposal.bind(this),
            'execution_status': this.handleExecutionStatus.bind(this),
            'system_status': this.handleSystemStatus.bind(this)
        };

        // DOM elements
        this.elements = {};
        this.initializeElements();
        this.setupEventListeners();
    }

    initializeElements() {
        this.elements = {
            chatMessages: document.getElementById('chat-messages'),
            userInput: document.getElementById('user-input'),
            sendButton: document.getElementById('send-button'),
            connectionStatus: document.getElementById('connection-status'),
            statusIndicator: document.getElementById('status-indicator'),
            statusText: document.getElementById('status-text'),
            modeBadge: document.getElementById('mode-badge'),
            clearSignifiersBtn: document.getElementById('clear-signifiers-btn')
        };

        // Log which elements were found for debugging if needed
        Object.entries(this.elements).forEach(([key, element]) => {
            if (!element) {
                console.warn(`Element ${key} not found`);
            }
        });
    }

    setupEventListeners() {
        // Send button
        if (this.elements.sendButton) {
            this.elements.sendButton.addEventListener('click', () => this.sendMessage());
        }

        // Enter key in input
        if (this.elements.userInput) {
            this.elements.userInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            // Auto-resize textarea
            this.elements.userInput.addEventListener('input', () => {
                this.autoResizeInput();
            });
        }

        // Clear signifiers button
        if (this.elements.clearSignifiersBtn) {
            this.elements.clearSignifiersBtn.addEventListener('click', () => this.clearSignifiers());
        }
    }

    async initialize() {
        console.log('Initializing ChatManager...');
        await this.checkManualModeStatus();

        if (this.manualModeActive) {
            await this.connect();
        } else {
            this.updateConnectionStatus('Manual mode inactive', 'disconnected');
        }

        // Start polling for manual mode status changes
        this.startStatusPolling();
    }

    startStatusPolling() {
        // Poll every 2 seconds to check if manual mode status changed
        this.statusPollingInterval = setInterval(async () => {
            const wasActive = this.manualModeActive;
            await this.checkManualModeStatus();

            // If manual mode became active and we weren't connected, try to connect
            if (!wasActive && this.manualModeActive && !this.connected) {
                console.log('Manual mode became active, attempting to connect...');
                await this.connect();
            }

            // If manual mode became inactive, disconnect
            if (wasActive && !this.manualModeActive && this.connected) {
                console.log('Manual mode became inactive, disconnecting...');
                this.disconnect();
            }
        }, 2000);
    }

    async checkManualModeStatus() {
        try {
            const response = await fetch('/api/manual/status');
            const data = await response.json();
            this.manualModeActive = data.manual_mode_active;
            this.updateModeStatus(this.manualModeActive);
        } catch (error) {
            console.error('Failed to check manual mode status:', error);
            this.manualModeActive = false;
            this.updateModeStatus(false);
        }
    }

    updateModeStatus(active) {
        if (this.elements.modeBadge) {
            this.elements.modeBadge.textContent = active ? 'Manual Mode' : 'Inactive';
            this.elements.modeBadge.className = active ? 'mode-badge manual' : 'mode-badge inactive';
        }
    }

    async startManualMode() {
        try {
            this.updateConnectionStatus('Starting manual mode...', 'connecting');

            const response = await fetch('/api/manual/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to start manual mode');
            }

            this.manualModeActive = true;
            this.updateModeStatus(true);

            // Poll status until system is ready
            await this.waitForSystemReady();

            await this.connect();

        } catch (error) {
            console.error('Failed to start manual mode:', error);
            this.displaySystemMessage(`Failed to start manual mode: ${error.message}`, 'error');
            this.updateConnectionStatus('Failed to start', 'disconnected');
        }
    }

    async waitForSystemReady(maxWaitTime = 120000) { // 2 minutes max wait
        const startTime = Date.now();
        let lastMessage = '';

        while (Date.now() - startTime < maxWaitTime) {
            try {
                const response = await fetch('/api/manual/status');
                if (response.ok) {
                    const status = await response.json();

                    // Update status if message changed
                    if (status.startup_message && status.startup_message !== lastMessage) {
                        this.updateConnectionStatus(status.startup_message, 'connecting');
                        lastMessage = status.startup_message;
                    }

                    // Check if system is ready
                    if (status.ready_for_interaction) {
                        this.updateConnectionStatus('System ready, connecting...', 'connecting');
                        return; // System is ready
                    }

                    // Check for error state
                    if (status.startup_phase === 'error') {
                        throw new Error(status.startup_message || 'System startup failed');
                    }
                }
            } catch (error) {
                if (error.message.includes('System startup failed') || error.message.includes('startup_message')) {
                    throw error; // Re-throw startup errors
                }
                // Ignore network errors during polling
                console.warn('Status check failed:', error);
            }

            // Wait before next check
            await new Promise(resolve => setTimeout(resolve, 1000));
        }

        throw new Error('System startup timed out after 2 minutes');
    }

    async connect() {
        if (this.connecting || this.connected) {
            return;
        }

        this.connecting = true;
        this.updateConnectionStatus('Connecting...', 'connecting');

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/manual/chat`;

        try {
            this.ws = new WebSocket(wsUrl);

            this.ws.onopen = () => {
                console.log('WebSocket connected');
                console.log('Connection count: this should only appear once per connection');
                this.connected = true;
                this.connecting = false;
                this.reconnectAttempts = 0;
                this.reconnectDelay = 1000;
                this.updateConnectionStatus('Connected', 'connected');
                this.enableInput();
                this.flushMessageQueue();
            };

            this.ws.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleIncomingMessage(message);
                } catch (error) {
                    console.error('Failed to parse message:', error);
                }
            };

            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.connected = false;
                this.connecting = false;
                this.updateConnectionStatus('Disconnected', 'disconnected');
                this.disableInput();
                this.scheduleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.connecting = false;
                this.updateConnectionStatus('Connection error', 'disconnected');
            };

        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.connecting = false;
            this.updateConnectionStatus('Connection failed', 'disconnected');
        }
    }

    scheduleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts && this.manualModeActive) {
            this.reconnectAttempts++;
            const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);

            console.log(`Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);
            setTimeout(() => this.connect(), delay);
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.connected = false;
        this.connecting = false;
    }

    sendMessage() {
        const content = this.elements.userInput?.value.trim();
        if (!content) return;

        if (!this.connected) {
            if (!this.manualModeActive) {
                this.displaySystemMessage('Manual mode is not active. Please start manual mode using the button in the header.', 'warning');
                return;
            } else {
                this.displaySystemMessage('Not connected. Attempting to reconnect...', 'warning');
                this.connect();
                return;
            }
        }

        // Create thread ID if not exists
        if (!this.activeThreadId) {
            this.activeThreadId = this.generateUUID();
        }

        const message = {
            type: 'user_message',
            content: content,
            thread_id: this.activeThreadId
        };

        if (this.connected && this.ws.readyState === WebSocket.OPEN) {
            // Set loading state
            this.setLoadingState(true);

            // Send message
            this.ws.send(JSON.stringify(message));
            this.displayUserMessage(content);
            this.elements.userInput.value = '';
            this.autoResizeInput();

            // Show typing indicator
            this.showTypingIndicator();
        } else {
            this.messageQueue.push(message);
            this.displaySystemMessage('Message queued. Reconnecting...', 'warning');
        }
    }

    flushMessageQueue() {
        while (this.messageQueue.length > 0 && this.connected) {
            const message = this.messageQueue.shift();
            this.ws.send(JSON.stringify(message));
        }
    }

    handleIncomingMessage(message) {
        const handler = this.messageHandlers[message.type];
        if (handler) {
            handler(message);
        } else {
            console.warn('Unknown message type:', message.type);
        }
    }

    handleAssistantMessage(message) {
        this.hideTypingIndicator();
        this.displayAssistantMessage(message.content, message.timestamp);
        this.setLoadingState(false);
        this.enableInput();
    }

    handlePlanProposal(message) {
        this.displayPlanProposal(message);
        this.enableInput();
    }

    handleExecutionStatus(message) {
        this.displayExecutionStatus(message);

        // Re-enable input when execution is complete
        if (message.status === 'success' || message.status === 'failed') {
            this.enableInput();
        }
    }

    handleSystemStatus(message) {
        this.displaySystemMessage(message.content, message.level);
    }

    displayUserMessage(content) {
        const messageEl = this.createMessageElement('user', content, new Date());
        this.appendMessage(messageEl);
    }

    displayAssistantMessage(content, timestamp) {
        const messageEl = this.createMessageElement('assistant', content, new Date(timestamp || Date.now()));
        this.appendMessage(messageEl);
    }

    displayPlanProposal(planData) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message plan-proposal';

        const header = document.createElement('div');
        header.className = 'message-header';
        header.innerHTML = `
            <span class="message-sender">Assistant</span>
            <span class="message-timestamp">${this.formatTimestamp(new Date(planData.timestamp))}</span>
        `;

        const content = document.createElement('div');
        content.className = 'message-content';

        const summary = document.createElement('div');
        summary.className = 'plan-summary';

        // Render markdown for plan proposals
        if (typeof marked !== 'undefined') {
            summary.innerHTML = marked.parse(planData.content);
        } else {
            summary.textContent = planData.content;
        }

        const details = document.createElement('div');
        details.className = 'plan-details';

        const toggleBtn = document.createElement('button');
        toggleBtn.className = 'toggle-plan-details';
        toggleBtn.textContent = 'Show Details';

        const planJson = document.createElement('div');
        planJson.className = 'plan-json';
        planJson.style.display = 'none';
        planJson.textContent = JSON.stringify(planData.plan, null, 2);

        toggleBtn.addEventListener('click', () => {
            const isHidden = planJson.style.display === 'none';
            planJson.style.display = isHidden ? 'block' : 'none';
            toggleBtn.textContent = isHidden ? 'Hide Details' : 'Show Details';
        });

        const actions = document.createElement('div');
        actions.className = 'plan-actions';

        const approveBtn = document.createElement('button');
        approveBtn.className = 'btn-approve';
        approveBtn.textContent = '✅ Approve';
        approveBtn.addEventListener('click', () => {
            this.confirmPlan(planData.plan_id, planData.thread_id, true);
            approveBtn.disabled = true;
            rejectBtn.disabled = true;
        });

        const rejectBtn = document.createElement('button');
        rejectBtn.className = 'btn-reject';
        rejectBtn.textContent = '❌ Reject';
        rejectBtn.addEventListener('click', () => {
            this.confirmPlan(planData.plan_id, planData.thread_id, false);
            approveBtn.disabled = true;
            rejectBtn.disabled = true;
        });

        details.appendChild(toggleBtn);
        details.appendChild(planJson);

        actions.appendChild(approveBtn);
        actions.appendChild(rejectBtn);

        content.appendChild(summary);
        content.appendChild(details);
        content.appendChild(actions);

        messageEl.appendChild(header);
        messageEl.appendChild(content);

        this.appendMessage(messageEl);
    }

    displayExecutionStatus(statusData) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message execution-status';

        const header = document.createElement('div');
        header.className = 'message-header';
        header.innerHTML = `
            <span class="message-sender">System</span>
            <span class="message-timestamp">${this.formatTimestamp(new Date(statusData.timestamp))}</span>
        `;

        const content = document.createElement('div');
        content.className = 'message-content';

        const statusText = document.createElement('div');
        statusText.className = 'status-text';
        statusText.textContent = statusData.content;

        content.appendChild(statusText);

        // Add progress bar if progress is specified
        if (statusData.progress !== null && statusData.progress !== undefined) {
            const progressBar = document.createElement('div');
            progressBar.className = 'progress-bar';

            const progressFill = document.createElement('div');
            progressFill.className = `progress-fill ${statusData.status}`;
            progressFill.style.width = `${statusData.progress}%`;

            progressBar.appendChild(progressFill);
            content.appendChild(progressBar);
        }

        messageEl.appendChild(header);
        messageEl.appendChild(content);

        this.appendMessage(messageEl);
    }

    displaySystemMessage(content, level = 'info') {
        const messageEl = document.createElement('div');
        messageEl.className = `message system-message ${level}`;

        const header = document.createElement('div');
        header.className = 'message-header';
        header.innerHTML = `
            <span class="message-sender">System</span>
            <span class="message-timestamp">${this.formatTimestamp(new Date())}</span>
        `;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;

        messageEl.appendChild(header);
        messageEl.appendChild(messageContent);

        this.appendMessage(messageEl);
    }

    createMessageElement(type, content, timestamp) {
        const messageEl = document.createElement('div');
        messageEl.className = `message ${type}-message`;

        const header = document.createElement('div');
        header.className = 'message-header';
        header.innerHTML = `
            <span class="message-sender">${type === 'user' ? 'You' : 'Assistant'}</span>
            <span class="message-timestamp">${this.formatTimestamp(timestamp)}</span>
        `;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        // Add data attribute for short text to help with CSS styling
        const textLength = content.trim().length;
        if (textLength <= 10) {
            messageContent.setAttribute('data-length', 'short');
        } else if (textLength <= 50) {
            messageContent.setAttribute('data-length', 'medium');
        } else {
            messageContent.setAttribute('data-length', 'long');
        }

        // Render markdown for assistant messages, plain text for user messages
        if (type === 'assistant') {
            if (typeof marked !== 'undefined') {
                try {
                    let renderedHtml;
                    // Try modern marked API
                    if (marked.parse) {
                        renderedHtml = marked.parse(content, {
                            breaks: true,
                            gfm: true
                        });
                    } else {
                        renderedHtml = marked(content, {
                            breaks: true,
                            gfm: true
                        });
                    }
                    messageContent.innerHTML = renderedHtml;
                } catch (error) {
                    console.error('Markdown rendering failed:', error);
                    messageContent.textContent = content;
                }
            } else {
                console.warn('marked.js not available, using plain text');
                messageContent.textContent = content;
            }
        } else {
            // For user messages, use plain text to prevent XSS
            messageContent.textContent = content;
        }

        messageEl.appendChild(header);
        messageEl.appendChild(messageContent);

        return messageEl;
    }

    appendMessage(messageEl) {
        if (!this.elements.chatMessages) return;

        this.elements.chatMessages.appendChild(messageEl);
        this.scrollToBottom();
    }

    showTypingIndicator() {
        // Remove existing typing indicator
        this.hideTypingIndicator();

        const typingEl = document.createElement('div');
        typingEl.id = 'typing-indicator';
        typingEl.className = 'message assistant-message typing-message';

        const header = document.createElement('div');
        header.className = 'message-header';
        header.innerHTML = `
            <span class="message-sender">Assistant</span>
            <span class="message-timestamp">${this.formatTimestamp(new Date())}</span>
        `;

        const content = document.createElement('div');
        content.className = 'message-content typing-content';
        content.innerHTML = `
            <div class="typing-dots">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
            <span class="typing-text">Assistant is thinking...</span>
        `;

        typingEl.appendChild(header);
        typingEl.appendChild(content);

        if (this.elements.chatMessages) {
            this.elements.chatMessages.appendChild(typingEl);
            this.scrollToBottom();
        }
    }

    hideTypingIndicator() {
        const existingIndicator = document.getElementById('typing-indicator');
        if (existingIndicator) {
            existingIndicator.remove();
        }
    }

    setLoadingState(loading) {
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = loading;
            this.elements.sendButton.textContent = loading ? 'Sending...' : 'Send';
        }

        if (this.elements.userInput) {
            this.elements.userInput.disabled = loading;
        }

        if (this.elements.clearSignifiersBtn) {
            this.elements.clearSignifiersBtn.disabled = loading;
        }
    }

    confirmPlan(planId, threadId, approved) {
        const message = {
            type: 'plan_confirmation',
            approved: approved,
            thread_id: threadId,
            plan_id: planId
        };

        if (this.connected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
            this.displaySystemMessage(
                `Plan ${approved ? 'approved' : 'rejected'}`,
                approved ? 'info' : 'warning'
            );
        }
    }

    async clearSignifiers() {
        try {
            this.elements.clearSignifiersBtn.disabled = true;

            const message = {
                type: 'clear_signifiers'
            };

            if (this.connected && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify(message));
            } else {
                // Fallback to HTTP endpoint
                const response = await fetch('/api/manual/clear-signifiers', { method: 'POST' });
                const data = await response.json();

                if (response.ok) {
                    this.displaySystemMessage('Signifier memory cleared', 'info');
                } else {
                    this.displaySystemMessage(`Failed to clear signifiers: ${data.error}`, 'error');
                }
            }
        } catch (error) {
            console.error('Failed to clear signifiers:', error);
            this.displaySystemMessage(`Failed to clear signifiers: ${error.message}`, 'error');
        } finally {
            this.elements.clearSignifiersBtn.disabled = false;
        }
    }

    updateConnectionStatus(text, status) {
        if (this.elements.statusIndicator) {
            this.elements.statusIndicator.className = `status-indicator status-${status}`;
        }
        if (this.elements.statusText) {
            this.elements.statusText.textContent = text;
        }
    }

    enableInput() {
        if (this.elements.userInput) {
            this.elements.userInput.disabled = false;
        }
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = false;
        }
    }

    disableInput() {
        if (this.elements.userInput) {
            this.elements.userInput.disabled = true;
        }
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = true;
        }
    }

    autoResizeInput() {
        const input = this.elements.userInput;
        if (!input) return;

        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';

        // Enable/disable send button based on content
        if (this.elements.sendButton) {
            this.elements.sendButton.disabled = !input.value.trim() || !this.connected;
        }
    }

    scrollToBottom() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.scrollTop = this.elements.chatMessages.scrollHeight;
        }
    }

    formatTimestamp(date) {
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    // Public API for external control
    async startSession() {
        if (!this.manualModeActive) {
            throw new Error('Manual mode is not active. Please start manual mode using the header button.');
        }
        if (!this.connected) {
            await this.connect();
        }
    }

    stopSession() {
        this.disconnect();
        this.manualModeActive = false;
        this.updateModeStatus(false);

        // Stop status polling
        if (this.statusPollingInterval) {
            clearInterval(this.statusPollingInterval);
            this.statusPollingInterval = null;
        }
    }

    clearChat() {
        if (this.elements.chatMessages) {
            this.elements.chatMessages.innerHTML = '';
        }
        this.activeThreadId = null;
    }

    getStatus() {
        return {
            connected: this.connected,
            manualMode: this.manualModeActive,
            activeThread: this.activeThreadId,
            messageQueue: this.messageQueue.length
        };
    }
}