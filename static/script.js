// Additional JavaScript functionality for RAG Interface

// Configuration
const CONFIG = {
    AUTO_SAVE_QUERIES: true,
    MAX_QUERY_HISTORY: 50,
    NOTIFICATION_DURATION: 5000,
    AUTO_REFRESH_INTERVAL: 30000
};

// Query history management
class QueryHistory {
    constructor() {
        this.queries = this.loadQueries();
    }

    loadQueries() {
        try {
            const saved = localStorage.getItem('rag_query_history');
            return saved ? JSON.parse(saved) : [];
        } catch (error) {
            console.warn('Failed to load query history:', error);
            return [];
        }
    }

    saveQuery(query, results) {
        if (!CONFIG.AUTO_SAVE_QUERIES) return;

        const queryData = {
            id: Date.now(),
            query: query,
            timestamp: new Date().toISOString(),
            resultsCount: results ? results.sources.length : 0,
            summary: results ? results.summary.substring(0, 200) + '...' : null
        };

        this.queries.unshift(queryData);
        
        // Keep only the latest queries
        if (this.queries.length > CONFIG.MAX_QUERY_HISTORY) {
            this.queries = this.queries.slice(0, CONFIG.MAX_QUERY_HISTORY);
        }

        try {
            localStorage.setItem('rag_query_history', JSON.stringify(this.queries));
        } catch (error) {
            console.warn('Failed to save query history:', error);
        }
    }

    getRecentQueries(limit = 10) {
        return this.queries.slice(0, limit);
    }

    clearHistory() {
        this.queries = [];
        localStorage.removeItem('rag_query_history');
    }
}

// Notification system
class NotificationManager {
    constructor() {
        this.container = this.createContainer();
    }

    createContainer() {
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 400px;
        `;
        document.body.appendChild(container);
        return container;
    }

    show(message, type = 'info', duration = CONFIG.NOTIFICATION_DURATION) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '×';
        closeBtn.style.cssText = `
            float: right;
            background: none;
            border: none;
            font-size: 18px;
            cursor: pointer;
            margin-left: 10px;
        `;
        closeBtn.onclick = () => this.remove(notification);
        notification.appendChild(closeBtn);

        this.container.appendChild(notification);

        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => this.remove(notification), duration);
        }

        return notification;
    }

    remove(notification) {
        if (notification && notification.parentNode) {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }
    }

    success(message, duration) {
        return this.show(message, 'success', duration);
    }

    error(message, duration) {
        return this.show(message, 'error', duration);
    }

    info(message, duration) {
        return this.show(message, 'info', duration);
    }
}

// Search suggestions and autocomplete
class SearchSuggestions {
    constructor(inputElement, historyManager) {
        this.input = inputElement;
        this.history = historyManager;
        this.suggestions = [];
        this.currentIndex = -1;
        this.setupSuggestions();
    }

    setupSuggestions() {
        // Create suggestions dropdown
        this.dropdown = document.createElement('div');
        this.dropdown.className = 'suggestions-dropdown';
        this.dropdown.style.cssText = `
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: white;
            border: 1px solid #e9ecef;
            border-top: none;
            border-radius: 0 0 8px 8px;
            max-height: 200px;
            overflow-y: auto;
            display: none;
            z-index: 100;
        `;

        // Position relative container
        this.input.parentNode.style.position = 'relative';
        this.input.parentNode.appendChild(this.dropdown);

        // Event listeners
        this.input.addEventListener('input', () => this.handleInput());
        this.input.addEventListener('keydown', (e) => this.handleKeydown(e));
        this.input.addEventListener('focus', () => this.showSuggestions());
        document.addEventListener('click', (e) => this.handleClickOutside(e));
    }

    handleInput() {
        const query = this.input.value.trim();
        if (query.length > 2) {
            this.updateSuggestions(query);
        } else {
            this.hideSuggestions();
        }
    }

    updateSuggestions(query) {
        const recentQueries = this.history.getRecentQueries(10);
        this.suggestions = recentQueries
            .filter(item => item.query.toLowerCase().includes(query.toLowerCase()))
            .slice(0, 5);

        this.renderSuggestions();
    }

    renderSuggestions() {
        if (this.suggestions.length === 0) {
            this.hideSuggestions();
            return;
        }

        this.dropdown.innerHTML = '';
        this.suggestions.forEach((suggestion, index) => {
            const item = document.createElement('div');
            item.className = 'suggestion-item';
            item.style.cssText = `
                padding: 10px 15px;
                cursor: pointer;
                border-bottom: 1px solid #f8f9fa;
                font-size: 14px;
            `;
            
            item.innerHTML = `
                <div style="font-weight: 500;">${this.highlightMatch(suggestion.query)}</div>
                <div style="font-size: 12px; color: #6c757d; margin-top: 5px;">
                    ${new Date(suggestion.timestamp).toLocaleDateString()} • 
                    ${suggestion.resultsCount} results
                </div>
            `;

            item.addEventListener('click', () => {
                this.input.value = suggestion.query;
                this.hideSuggestions();
                this.input.focus();
            });

            item.addEventListener('mouseenter', () => {
                this.currentIndex = index;
                this.highlightSuggestion();
            });

            this.dropdown.appendChild(item);
        });

        this.showSuggestions();
    }

    highlightMatch(text) {
        const query = this.input.value.trim();
        if (!query) return text;
        
        const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'gi');
        return text.replace(regex, '<mark>$1</mark>');
    }

    handleKeydown(e) {
        if (!this.isVisible()) return;

        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                this.currentIndex = Math.min(this.currentIndex + 1, this.suggestions.length - 1);
                this.highlightSuggestion();
                break;
            case 'ArrowUp':
                e.preventDefault();
                this.currentIndex = Math.max(this.currentIndex - 1, -1);
                this.highlightSuggestion();
                break;
            case 'Enter':
                if (this.currentIndex >= 0) {
                    e.preventDefault();
                    this.input.value = this.suggestions[this.currentIndex].query;
                    this.hideSuggestions();
                }
                break;
            case 'Escape':
                this.hideSuggestions();
                break;
        }
    }

    highlightSuggestion() {
        const items = this.dropdown.querySelectorAll('.suggestion-item');
        items.forEach((item, index) => {
            item.style.backgroundColor = index === this.currentIndex ? '#f8f9fa' : 'white';
        });
    }

    handleClickOutside(e) {
        if (!this.input.contains(e.target) && !this.dropdown.contains(e.target)) {
            this.hideSuggestions();
        }
    }

    showSuggestions() {
        this.dropdown.style.display = 'block';
    }

    hideSuggestions() {
        this.dropdown.style.display = 'none';
        this.currentIndex = -1;
    }

    isVisible() {
        return this.dropdown.style.display === 'block';
    }
}

// Keyboard shortcuts
class KeyboardShortcuts {
    constructor() {
        this.setupShortcuts();
    }

    setupShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + Enter to search
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const searchBtn = document.getElementById('searchBtn');
                if (searchBtn && !searchBtn.disabled) {
                    searchBtn.click();
                }
            }

            // Ctrl/Cmd + U to upload
            if ((e.ctrlKey || e.metaKey) && e.key === 'u') {
                e.preventDefault();
                const selectFilesBtn = document.getElementById('selectFilesBtn');
                if (selectFilesBtn) {
                    selectFilesBtn.click();
                }
            }

            // Ctrl/Cmd + R to refresh documents
            if ((e.ctrlKey || e.metaKey) && e.key === 'r' && e.shiftKey) {
                e.preventDefault();
                const listDocsBtn = document.getElementById('listDocsBtn');
                if (listDocsBtn) {
                    listDocsBtn.click();
                }
            }

            // Focus search with /
            if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
                const activeElement = document.activeElement;
                if (activeElement.tagName !== 'INPUT' && activeElement.tagName !== 'TEXTAREA') {
                    e.preventDefault();
                    const searchQuery = document.getElementById('searchQuery');
                    if (searchQuery) {
                        searchQuery.focus();
                    }
                }
            }
        });
    }
}

// Export functionality
class ExportManager {
    static exportSearchResults(searchData) {
        const exportData = {
            timestamp: new Date().toISOString(),
            query: searchData.query,
            summary: searchData.summary,
            sources: searchData.sources.map(source => ({
                text_preview: source.text_preview,
                distance: source.distance
            })),
            metadata: {
                total_chunks: searchData.total_chunks,
                sources_count: searchData.sources.length
            }
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `rag_search_results_${Date.now()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }

    static exportQueryHistory(historyManager) {
        const history = historyManager.getRecentQueries(CONFIG.MAX_QUERY_HISTORY);
        const exportData = {
            exported_at: new Date().toISOString(),
            query_count: history.length,
            queries: history
        };

        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = `rag_query_history_${Date.now()}.json`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        URL.revokeObjectURL(url);
    }
}

// Theme manager
class ThemeManager {
    constructor() {
        this.currentTheme = this.loadTheme();
        this.applyTheme(this.currentTheme);
    }

    loadTheme() {
        return localStorage.getItem('rag_theme') || 'default';
    }

    saveTheme(theme) {
        localStorage.setItem('rag_theme', theme);
    }

    applyTheme(theme) {
        document.body.className = `theme-${theme}`;
        this.currentTheme = theme;
        this.saveTheme(theme);
    }

    getAvailableThemes() {
        return ['default', 'dark', 'compact', 'high-contrast'];
    }

    switchTheme(theme) {
        if (this.getAvailableThemes().includes(theme)) {
            this.applyTheme(theme);
        }
    }
}

// Initialize enhanced features when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize managers
    window.queryHistory = new QueryHistory();
    window.notifications = new NotificationManager();
    window.keyboardShortcuts = new KeyboardShortcuts();
    window.themeManager = new ThemeManager();

    // Setup search suggestions
    const searchQuery = document.getElementById('searchQuery');
    if (searchQuery) {
        window.searchSuggestions = new SearchSuggestions(searchQuery, window.queryHistory);
    }

    // Add export buttons (optional)
    addExportButtons();

    // Add theme switcher (optional)
    addThemeSwitcher();

    // Show keyboard shortcuts help
    addKeyboardShortcutsHelp();
});

function addExportButtons() {
    // Add export button to search results
    const searchSection = document.querySelector('.section h2');
    if (searchSection && searchSection.textContent.includes('Search')) {
        const exportBtn = document.createElement('button');
        exportBtn.className = 'btn btn-primary';
        exportBtn.style.cssText = 'float: right; font-size: 12px; padding: 5px 10px;';
        exportBtn.textContent = 'Export Results';
        exportBtn.onclick = () => {
            if (window.lastSearchResults) {
                ExportManager.exportSearchResults(window.lastSearchResults);
                window.notifications.success('Search results exported successfully!');
            } else {
                window.notifications.error('No search results to export');
            }
        };
        searchSection.appendChild(exportBtn);
    }
}

function addThemeSwitcher() {
    // Add theme switcher to header
    const header = document.querySelector('.header');
    if (header) {
        const themeSelector = document.createElement('select');
        themeSelector.style.cssText = `
            position: absolute;
            top: 20px;
            right: 20px;
            padding: 5px 10px;
            border: 1px solid rgba(255,255,255,0.3);
            background: rgba(255,255,255,0.1);
            color: white;
            border-radius: 5px;
        `;

        window.themeManager.getAvailableThemes().forEach(theme => {
            const option = document.createElement('option');
            option.value = theme;
            option.textContent = theme.charAt(0).toUpperCase() + theme.slice(1);
            option.selected = theme === window.themeManager.currentTheme;
            themeSelector.appendChild(option);
        });

        themeSelector.onchange = (e) => {
            window.themeManager.switchTheme(e.target.value);
            window.notifications.info(`Theme changed to ${e.target.value}`);
        };

        header.style.position = 'relative';
        header.appendChild(themeSelector);
    }
}

function addKeyboardShortcutsHelp() {
    // Add help button
    const statusBar = document.querySelector('.status-bar');
    if (statusBar) {
        const helpBtn = document.createElement('button');
        helpBtn.textContent = '⌨️ Shortcuts';
        helpBtn.className = 'btn btn-primary';
        helpBtn.style.cssText = 'font-size: 12px; padding: 5px 10px; margin-left: 10px;';
        helpBtn.onclick = showKeyboardShortcuts;
        statusBar.appendChild(helpBtn);
    }
}

function showKeyboardShortcuts() {
    const shortcuts = [
        'Ctrl/Cmd + Enter: Search',
        'Ctrl/Cmd + U: Upload files',
        'Ctrl/Cmd + Shift + R: Refresh documents',
        '/: Focus search box',
        'Arrow keys: Navigate suggestions',
        'Escape: Close suggestions'
    ];

    const message = 'Keyboard Shortcuts:\n\n' + shortcuts.join('\n');
    alert(message);
}

// Enhance the existing search function to save history
const originalPerformSearch = window.performSearch;
if (typeof originalPerformSearch === 'function') {
    window.performSearch = async function() {
        const result = await originalPerformSearch.apply(this, arguments);
        
        // Save to history if search was successful
        if (window.lastSearchResults && window.queryHistory) {
            const query = document.getElementById('searchQuery').value.trim();
            window.queryHistory.saveQuery(query, window.lastSearchResults);
        }
        
        return result;
    };
}

// Add CSS for additional features
const additionalStyles = `
    <style>
        .theme-dark {
            filter: invert(1) hue-rotate(180deg);
        }
        
        .theme-dark img, 
        .theme-dark video {
            filter: invert(1) hue-rotate(180deg);
        }
        
        .suggestions-dropdown {
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .suggestion-item:hover {
            background: #f8f9fa !important;
        }
        
        mark {
            background: #ffeb3b;
            padding: 0 2px;
            border-radius: 2px;
        }
        
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    </style>
`;

document.head.insertAdjacentHTML('beforeend', additionalStyles);