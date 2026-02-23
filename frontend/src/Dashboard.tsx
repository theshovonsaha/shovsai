import React, { useState, useEffect } from 'react';

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
    const [loading, setLoading] = useState(true);

    useEffect(() => { fetchAgents(); }, []);

    const fetchAgents = async () => {
        try {
            const data = await fetch('/api/agents').then(r => r.json());
            setAgents(data.agents || []);
        } catch (e) { console.error('Failed to fetch agents:', e); }
        finally { setLoading(false); }
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
            <div className="dashboard-header">
                <h1><span>AGENT</span> // PLATFORM</h1>
                <button className="primary-btn" onClick={() => setShowCreateModal(true)}>
                    + new agent
                </button>
            </div>

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
                            <div className="agent-meta">
                                <span>{agent.model}</span>
                                <span>{agent.tools.length} tools</span>
                            </div>
                            <button className="launch-btn" onClick={() => onSelectAgent(agent.id)}>
                                launch →
                            </button>
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
        </div>
    );
};


const CreateAgentModal: React.FC<{ onClose: () => void; onCreated: () => void }> = ({ onClose, onCreated }) => {
    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [model, setModel] = useState('llama3.2');
    const [selectedTools, setSelectedTools] = useState<string[]>([]);
    const [availableTools, setAvailableTools] = useState<any[]>([]);
    const [availableModels, setAvailableModels] = useState<string[]>(['llama3.2']);
    const [creating, setCreating] = useState(false);

    useEffect(() => {
        fetch('/api/tools').then(r => r.json()).then(d => setAvailableTools(d.tools || []));
        fetch('/api/models').then(r => r.json()).then(d => { if (d.models?.length) setAvailableModels(d.models); });
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
                body: JSON.stringify({ name: name.trim(), description, model, tools: selectedTools }),
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
                    <label>Model</label>
                    <select
                        value={model}
                        onChange={e => setModel(e.target.value)}
                        style={{ width: '100%', padding: '10px 12px', borderRadius: '2px' }}
                    >
                        {availableModels.map(m => <option key={m} value={m}>{m}</option>)}
                    </select>
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