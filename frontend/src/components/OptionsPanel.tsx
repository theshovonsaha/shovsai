import React, { useState, useEffect, useCallback } from 'react';
import './OptionsPanel.css';

interface Memory {
    id: number;
    subject: string;
    predicate: string;
    object: string;
    created_at: string;
    similarity?: number;
}

interface OptionsPanelProps {
    sessionId: string | null;
    contextLines: number;
}

export const OptionsPanel: React.FC<OptionsPanelProps> = ({ sessionId, contextLines }) => {
    const [memories, setMemories] = useState<Memory[]>([]);
    const [total, setTotal] = useState(0);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState<Memory[] | null>(null);
    const [isSearching, setIsSearching] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [sessionContext, setSessionContext] = useState<string[]>([]);
    const [activeTab, setActiveTab] = useState<'memory' | 'context'>('memory');
    const [confirmClear, setConfirmClear] = useState(false);

    const loadMemories = useCallback(async () => {
        setIsLoading(true);
        try {
            const data = await fetch('/api/memory').then(r => r.json());
            setMemories(data.memories || []);
            setTotal(data.total || 0);
        } catch (e) {
            console.error('Failed to load memories', e);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const loadContext = useCallback(async () => {
        if (!sessionId) return;
        try {
            const data = await fetch(`/api/sessions/${sessionId}/context`).then(r => r.json());
            setSessionContext(data.context || []);
        } catch (e) {
            console.error('Failed to load context', e);
        }
    }, [sessionId]);

    useEffect(() => {
        loadMemories();
    }, [loadMemories]);

    useEffect(() => {
        if (activeTab === 'context') loadContext();
    }, [activeTab, loadContext]);

    const deleteMemory = async (id: number) => {
        try {
            await fetch(`/api/memory/${id}`, { method: 'DELETE' });
            setMemories(prev => prev.filter(m => m.id !== id));
            setSearchResults(prev => prev ? prev.filter(m => m.id !== id) : null);
            setTotal(prev => Math.max(0, prev - 1));
        } catch (e) {
            console.error('Failed to delete memory', e);
        }
    };

    const clearAllMemories = async () => {
        if (!confirmClear) {
            setConfirmClear(true);
            setTimeout(() => setConfirmClear(false), 3000);
            return;
        }
        try {
            await fetch('/api/memory', { method: 'DELETE' });
            setMemories([]);
            setTotal(0);
            setSearchResults(null);
            setConfirmClear(false);
        } catch (e) {
            console.error('Failed to clear memories', e);
        }
    };

    const searchMemory = async () => {
        if (!searchQuery.trim()) {
            setSearchResults(null);
            return;
        }
        setIsSearching(true);
        try {
            const data = await fetch('/api/memory/search', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: searchQuery, top_k: 10 }),
            }).then(r => r.json());
            setSearchResults(data.results || []);
        } catch (e) {
            console.error('Failed to search memories', e);
        } finally {
            setIsSearching(false);
        }
    };

    const displayMemories = searchResults || memories;

    return (
        <div className="options-panel">
            <div className="options-tabs">
                <button
                    className={`options-tab ${activeTab === 'memory' ? 'active' : ''}`}
                    onClick={() => setActiveTab('memory')}
                >
                    🧠 Memory
                    {total > 0 && <span className="tab-badge">{total}</span>}
                </button>
                <button
                    className={`options-tab ${activeTab === 'context' ? 'active' : ''}`}
                    onClick={() => setActiveTab('context')}
                >
                    📋 Context
                    {contextLines > 0 && <span className="tab-badge">{contextLines}</span>}
                </button>
            </div>

            {activeTab === 'memory' && (
                <div className="options-section">
                    <div className="memory-search-row">
                        <input
                            className="memory-search-input"
                            type="text"
                            placeholder="Semantic search…"
                            value={searchQuery}
                            onChange={e => setSearchQuery(e.target.value)}
                            onKeyDown={e => e.key === 'Enter' && searchMemory()}
                        />
                        <button className="memory-search-btn" onClick={searchMemory} disabled={isSearching}>
                            {isSearching ? '…' : '⌕'}
                        </button>
                        {searchResults !== null && (
                            <button className="memory-clear-search" onClick={() => { setSearchResults(null); setSearchQuery(''); }}>✕</button>
                        )}
                    </div>

                    {searchResults !== null && (
                        <div className="memory-search-label">
                            {searchResults.length} semantic match{searchResults.length !== 1 ? 'es' : ''} for "{searchQuery}"
                        </div>
                    )}

                    <div className="memory-list">
                        {isLoading ? (
                            <div className="memory-empty">Loading…</div>
                        ) : displayMemories.length === 0 ? (
                            <div className="memory-empty">
                                {searchResults !== null ? 'No matches found.' : 'No memories stored yet.'}
                                {searchResults === null && <><br /><span style={{ opacity: 0.5, fontSize: '11px' }}>Ask the agent to remember something!</span></>}
                            </div>
                        ) : (
                            displayMemories.map(memory => (
                                <div key={memory.id} className="memory-card">
                                    <div className="memory-triplet">
                                        <span className="memory-subject">{memory.subject}</span>
                                        <span className="memory-predicate">{memory.predicate}</span>
                                        <span className="memory-object">{memory.object}</span>
                                    </div>
                                    <div className="memory-footer">
                                        <span className="memory-date">
                                            {memory.similarity !== undefined
                                                ? `${Math.round(memory.similarity * 100)}% match`
                                                : new Date(memory.created_at).toLocaleDateString()}
                                        </span>
                                        <button className="memory-delete-btn" onClick={() => deleteMemory(memory.id)} title="Delete">✕</button>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>

                    {total > 0 && (
                        <div className="memory-actions">
                            <button onClick={loadMemories} className="memory-refresh-btn">↺ Refresh</button>
                            <button
                                onClick={clearAllMemories}
                                className={`memory-clear-btn ${confirmClear ? 'confirm' : ''}`}
                            >
                                {confirmClear ? '⚠ Click again to confirm' : '🗑 Clear All'}
                            </button>
                        </div>
                    )}
                </div>
            )}

            {activeTab === 'context' && (
                <div className="options-section">
                    {!sessionId ? (
                        <div className="memory-empty">No active session. Start a chat first.</div>
                    ) : sessionContext.length === 0 ? (
                        <div className="memory-empty">Context is empty for this session.</div>
                    ) : (
                        <>
                            <div className="context-meta">{sessionContext.length} context lines</div>
                            <div className="context-list">
                                {sessionContext.map((line, i) => (
                                    <div key={i} className="context-line">{line}</div>
                                ))}
                            </div>
                        </>
                    )}
                    {sessionId && (
                        <button className="memory-refresh-btn" style={{ marginTop: '8px' }} onClick={loadContext}>↺ Refresh</button>
                    )}
                </div>
            )}
        </div>
    );
};
