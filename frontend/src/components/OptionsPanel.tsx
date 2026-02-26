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
    currentSearchEngine: string;
    setCurrentSearchEngine: (engine: string) => void;
    models: Record<string, string[]>;
    usePlanner: boolean;
    setUsePlanner: (val: boolean) => void;
    plannerModel: string;
    setPlannerModel: (val: string) => void;
    contextModel: string;
    setContextModel: (val: string) => void;
    clearSessionContext: () => void;
}

export const OptionsPanel: React.FC<OptionsPanelProps> = ({
    sessionId,
    contextLines,
    currentSearchEngine,
    setCurrentSearchEngine,
    models,
    usePlanner,
    setUsePlanner,
    plannerModel,
    setPlannerModel,
    contextModel,
    setContextModel,
    clearSessionContext
}) => {
    const [memories, setMemories] = useState<Memory[]>([]);
    const [total, setTotal] = useState(0);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState<Memory[] | null>(null);
    const [isSearching, setIsSearching] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [sessionContext, setSessionContext] = useState<string[]>([]);
    const [activeTab, setActiveTab] = useState<'settings' | 'memory' | 'context'>('settings');
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
                    className={`options-tab ${activeTab === 'settings' ? 'active' : ''}`}
                    onClick={() => setActiveTab('settings')}
                >
                    ⚙ Settings
                </button>
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

            {activeTab === 'settings' && (
                <div className="options-section">
                    <div className="settings-card" style={{ marginBottom: '20px', padding: '12px', background: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                        <label className="settings-label" style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 600 }}>Web Search Engine</label>
                        <select
                            className="settings-select"
                            value={currentSearchEngine}
                            onChange={(e) => setCurrentSearchEngine(e.target.value)}
                            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid var(--border-color)', background: 'var(--bg-pane)', color: 'var(--text-main)', outline: 'none' }}
                        >
                            <option value="duckduckgo">DuckDuckGo (Free)</option>
                            <option value="tavily">Tavily Search</option>
                            <option value="brave">Brave Search</option>
                            <option value="searxng">SearxNG</option>
                        </select>
                        <p className="settings-help" style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-dim)', lineHeight: 1.4 }}>Select the engine used by the web_search tool. Ensure API keys for Tavily or Brave are set in your backend .env if selected.</p>
                    </div>

                    <div className="settings-card" style={{ marginBottom: '20px', padding: '12px', background: 'var(--surface2)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <label className="settings-label" style={{ fontSize: '13px', fontWeight: 600 }}>Manager Agent (Orchestration)</label>
                            <input
                                type="checkbox"
                                checked={usePlanner}
                                onChange={e => setUsePlanner(e.target.checked)}
                                style={{ transform: 'scale(1.2)', cursor: 'pointer' }}
                            />
                        </div>
                        <p className="settings-help" style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-dim)', lineHeight: 1.4 }}>When enabled, the agent uses a specialized "Planner" layer to proactively select tools before execution.</p>

                        {usePlanner && (
                            <div style={{ marginTop: '12px' }}>
                                <label className="settings-label" style={{ display: 'block', marginBottom: '6px', fontSize: '12px' }}>Planner Model</label>
                                <select
                                    className="settings-select"
                                    value={plannerModel}
                                    onChange={e => setPlannerModel(e.target.value)}
                                    style={{ width: '100%', padding: '6px', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--text)', fontSize: '12px' }}
                                >
                                    <option value="">Matches Session Model</option>
                                    {Object.entries(models).map(([provider, names]) => (
                                        <optgroup key={provider} label={provider.toUpperCase()}>
                                            {names.map(name => (
                                                <option key={name} value={`${provider}:${name}`}>{name}</option>
                                            ))}
                                        </optgroup>
                                    ))}
                                </select>
                            </div>
                        )}
                    </div>

                    <div className="settings-card" style={{ marginBottom: '20px', padding: '12px', background: 'var(--bg-card)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                        <label className="settings-label" style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 600 }}>Context Engine Intelligence</label>
                        <select
                            className="settings-select"
                            value={contextModel}
                            onChange={e => setContextModel(e.target.value)}
                            style={{ width: '100%', padding: '8px', borderRadius: '4px', border: '1px solid var(--border)', background: 'var(--bg)', color: 'var(--text)' }}
                        >
                            {Object.entries(models).map(([provider, names]) => (
                                <optgroup key={provider} label={provider.toUpperCase()}>
                                    {names.map(name => (
                                        <option key={name} value={`${provider}:${name}`}>{name}</option>
                                    ))}
                                </optgroup>
                            ))}
                        </select>
                        <p className="settings-help" style={{ marginTop: '8px', fontSize: '11px', color: 'var(--text-dim)', lineHeight: 1.4 }}>The model used for background memory compression and fact extraction (Default: DeepSeek R1).</p>
                    </div>

                    <div className="settings-card" style={{ padding: '12px', background: 'var(--surface2)', borderRadius: '8px', border: '1px solid var(--border)' }}>
                        <label className="settings-label" style={{ display: 'block', marginBottom: '8px', fontSize: '13px', fontWeight: 600 }}>Memory Controls</label>
                        <button
                            className="memory-clear-btn"
                            style={{ width: '100%', padding: '10px' }}
                            onClick={() => {
                                if (window.confirm("Are you sure you want to purge the compressed memory of this session? Conversation history will remain.")) {
                                    clearSessionContext();
                                }
                            }}
                        >
                            🗑 Purge Session Context
                        </button>
                    </div>
                </div>
            )}

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

                    <h4 className="section-title" style={{ fontSize: '13px', borderBottom: '1px solid var(--border-color)', paddingBottom: '4px', marginBottom: '10px' }}>Context Window</h4>
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
