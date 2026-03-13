import React, { useState, useEffect } from 'react';
import { PremiumSelect } from './components/PremiumSelect';

interface AgentProfile {
    id: string;
    name: string;
    description: string;
    model: string;
    tools: string[];
}

interface DashboardProps {
    onSelectAgent: (id: string) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onSelectAgent }) => {
    const [agents, setAgents] = useState<AgentProfile[]>([]);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [agentToDelete, setAgentToDelete] = useState<AgentProfile | null>(null);
    const [loading, setLoading] = useState(true);
    const [availableModels, setAvailableModels] = useState<Record<string, string[]>>({});
    const [savingModel, setSavingModel] = useState<string | null>(null); // agent id being saved

    useEffect(() => {
        fetchAgents();
        // Load available models for the inline selectors
        fetch('/api/models').then(r => r.json()).then(d => {
            if (d.models) setAvailableModels(d.models);
        }).catch(() => { });
    }, []);

    const fetchAgents = async () => {
        try {
            const data = await fetch('/api/agents').then(r => r.json());
            setAgents(data.agents || []);
        } catch (e) { console.error('Failed to fetch agents:', e); }
        finally { setLoading(false); }
    };

    const handleModelChange = async (agentId: string, newModel: string) => {
        // Optimistically update UI
        setAgents(prev => prev.map(a => a.id === agentId ? { ...a, model: newModel } : a));
        setSavingModel(agentId);
        try {
            await fetch(`/api/agents/${agentId}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model: newModel }),
            });
        } catch (e) {
            console.error('Failed to update agent model:', e);
            fetchAgents(); // revert on error
        } finally {
            setSavingModel(null);
        }
    };

    const confirmDelete = async () => {
        if (!agentToDelete) return;
        try {
            const res = await fetch(`/api/agents/${agentToDelete.id}`, { method: 'DELETE' });
            if (res.ok) fetchAgents();
            else {
                const data = await res.json();
                alert(data.detail || 'Failed to delete agent');
            }
        } catch (e) { console.error('Delete failed:', e); }
        finally { setAgentToDelete(null); }
    };

    if (loading) {
        return (
            <div className="dashboard-container">
                <div style={{ color: 'var(--text-dim)', fontSize: '10px', letterSpacing: '.2em' }}>
                    initializing platform...
                </div>
            </div>
        );
    }

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <div className="branding">
                    <span className="logo-a">SHOVS</span>
                    <span className="logo-sep">//</span>
                    <span className="logo-b">PLATFORM</span>
                </div>
                <button
                    className="btn-create-agent"
                    onClick={() => setShowCreateModal(true)}
                >
                    + Create Agent
                </button>
            </header>

            <div className="agent-grid">
                {agents.length === 0 ? (
                    <div className="dashboard-empty">
                        <div className="glyph">◈</div>
                        <div>no agents configured</div>
                    </div>
                ) : (
                    agents.map(agent => (
                        <div key={agent.id} className="agent-card">
                            <div className="agent-avatar">{agent.name.charAt(0).toUpperCase()}</div>
                            <h3>{agent.name}</h3>
                            <p>{agent.description || 'No description.'}</p>

                            {/* ── Per-Agent Model Selector ── */}
                            <div className="agent-model-selector" style={{ margin: '10px 0' }}>
                                <label style={{
                                    display: 'block',
                                    fontSize: '10px',
                                    letterSpacing: '.12em',
                                    color: 'var(--text-dim)',
                                    marginBottom: '5px',
                                    textTransform: 'uppercase',
                                }}>
                                    {savingModel === agent.id ? '⟳ saving…' : 'Model'}
                                </label>
                                <select
                                    value={agent.model}
                                    onChange={e => handleModelChange(agent.id, e.target.value)}
                                    disabled={savingModel === agent.id}
                                    style={{
                                        width: '100%',
                                        padding: '6px 8px',
                                        borderRadius: '6px',
                                        border: '1px solid var(--border)',
                                        background: 'var(--bg-pane, #111)',
                                        color: 'var(--text)',
                                        fontSize: '11px',
                                        outline: 'none',
                                        cursor: 'pointer',
                                        opacity: savingModel === agent.id ? 0.6 : 1,
                                        transition: 'border-color 0.15s',
                                    }}
                                    onFocus={e => (e.target.style.borderColor = 'var(--primary, #00d1ff)')}
                                    onBlur={e => (e.target.style.borderColor = 'var(--border)')}
                                >
                                    {/* Current model always visible even if not in list yet */}
                                    {!Object.values(availableModels).flat().includes(agent.model) && (
                                        <option value={agent.model}>{agent.model}</option>
                                    )}
                                    {Object.entries(availableModels).map(([provider, names]) => (
                                        names.length > 0 && (
                                            <optgroup key={provider} label={provider.toUpperCase()}>
                                                {names.map(name => (
                                                    <option key={name} value={`${provider}:${name}`}>
                                                        {name}
                                                    </option>
                                                ))}
                                            </optgroup>
                                        )
                                    ))}
                                </select>
                            </div>

                            <div className="agent-meta">
                                <span>{agent.tools.length} tools</span>
                            </div>
                            <div className="agent-actions">
                                <button className="launch-btn" onClick={() => onSelectAgent(agent.id)}>
                                    launch →
                                </button>
                                {agent.id !== 'default' && (
                                    <button className="danger-btn" onClick={(e) => { e.stopPropagation(); setAgentToDelete(agent); }}>
                                        delete
                                    </button>
                                )}
                            </div>
                        </div>
                    ))
                )}
            </div>

            {showCreateModal && (
                <CreateAgentModal
                    onClose={() => setShowCreateModal(false)}
                    onCreated={() => { setShowCreateModal(false); fetchAgents(); }}
                />
            )}
            {agentToDelete && (
                <DeleteConfirmationModal
                    agentName={agentToDelete.name}
                    onClose={() => setAgentToDelete(null)}
                    onConfirm={confirmDelete}
                />
            )}
        </div>
    );
};


const CreateAgentModal: React.FC<{ onClose: () => void; onCreated: () => void }> = ({ onClose, onCreated }) => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [model, setModel] = useState('llama3.2');
    const [embedModel, setEmbedModel] = useState('nomic-embed-text');
    const [selectedTools, setSelectedTools] = useState<string[]>([]);
    const [availableTools, setAvailableTools] = useState<any[]>([]);
    const [availableModels, setAvailableModels] = useState<Record<string, string[]>>({ 'ollama': ['llama3.2'] });
    const [availableEmbedModels] = useState<string[]>(['nomic-embed-text', 'text-embedding-3-small']);
    const [creating, setCreating] = useState(false);

    useEffect(() => {
        fetch('/api/tools').then(r => r.json()).then(d => setAvailableTools(d.tools || []));
        fetch('/api/models').then(r => r.json()).then(d => {
            if (d.models) setAvailableModels(d.models);
        });
    }, []);

    const toggleTool = (name: string) => setSelectedTools(prev =>
        prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    );

    const handleCreate = async () => {
        if (!name.trim()) return;
        setCreating(true);
        try {
            await fetch('/api/agents', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name.trim(),
                    description,
                    model,
                    embed_model: embedModel,
                    tools: selectedTools
                }),
            });
            onCreated();
        } catch (e) { console.error('Create failed:', e); }
        finally { setCreating(false); }
    };

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={e => e.stopPropagation()}>
                <h2>Initialize Agent</h2>

                <div className="input-group">
                    <label>Name</label>
                    <input
                        value={name}
                        onChange={e => setName(e.target.value)}
                        placeholder="e.g. research-assistant"
                        autoFocus
                    />
                </div>

                <div className="input-group">
                    <label>Description</label>
                    <input
                        value={description}
                        onChange={e => setDescription(e.target.value)}
                        placeholder="What is this agent's purpose?"
                    />
                </div>

                <div className="input-group">
                    <PremiumSelect
                        label="Chat Model"
                        value={model}
                        options={availableModels}
                        onChange={setModel}
                    />
                </div>

                <div className="input-group">
                    <PremiumSelect
                        label="Embedding Model"
                        value={embedModel}
                        options={availableEmbedModels}
                        onChange={setEmbedModel}
                    />
                </div>

                <div className="input-group">
                    <label>Tools</label>
                    <div className="tool-chips">
                        {availableTools.map(t => (
                            <span
                                key={t.name}
                                className={`chip ${selectedTools.includes(t.name) ? 'active' : ''}`}
                                onClick={() => toggleTool(t.name)}
                                title={t.description}
                            >
                                {t.name}
                            </span>
                        ))}
                    </div>
                </div>

                <div className="modal-actions">
                    <button onClick={onClose}>cancel</button>
                    <button
                        className="primary-btn"
                        onClick={handleCreate}
                        disabled={!name.trim() || creating}
                    >
                        {creating ? 'initializing...' : 'initialize'}
                    </button>
                </div>
            </div>
        </div>
    );
};

const DeleteConfirmationModal: React.FC<{ agentName: string; onClose: () => void; onConfirm: () => void }> = ({ agentName, onClose, onConfirm }) => {
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content danger" onClick={e => e.stopPropagation()}>
                <h2 style={{ color: 'var(--error)' }}>Confirm Deletion</h2>
                <p style={{ margin: '15px 0', color: 'var(--text-mid)', fontSize: '12px' }}>
                    Are you sure you want to delete <strong style={{ color: 'var(--text)' }}>{agentName}</strong>?
                    This action cannot be undone and will remove all associated configurations.
                </p>
                <div className="modal-actions">
                    <button onClick={onClose}>cancel</button>
                    <button className="danger-btn" onClick={onConfirm}>delete agent</button>
                </div>
            </div>
        </div>
    );
};