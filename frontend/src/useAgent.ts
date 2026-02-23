import { useState, useEffect, useRef, useCallback } from 'react';

export interface Session {
    id: string;
    title: string;
    model: string;
    created_at: string;
    updated_at: string;
    message_count: number;
}

export interface Attachment {
    id: string;
    file: File;
    dataURL: string | null;
}

export interface MessageBlock {
    type: 'text' | 'tool_call' | 'tool_result' | 'tool_error' | 'attachment_badge' | 'compressing';
    content: string;
    tool?: string;
    id: string;
}

export interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    files?: Attachment[];
    blocks: MessageBlock[];
}

export function useAgent() {
    const [health, setHealth] = useState<{ status: string; ollama: boolean }>({ status: 'connecting...', ollama: false });
    const [models, setModels] = useState<string[]>(['llama3.2']);
    const [tools, setTools] = useState<any[]>([]);
    const [sessions, setSessions] = useState<Session[]>([]);
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
    const [activeAgentId, setActiveAgentId] = useState<string | null>(null);
    const [currentModel, setCurrentModel] = useState<string>('llama3.2');
    const [currentSearchBackend, setCurrentSearchBackend] = useState<string>('auto');
    const [messages, setMessages] = useState<Message[]>([]);
    const [contextLines, setContextLines] = useState(0);
    const [isStreaming, setIsStreaming] = useState(false);
    const [pendingFiles, setPendingFiles] = useState<Attachment[]>([]);
    const [forceMemory, setForceMemory] = useState(false);

    const isSendingRef = useRef(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => { fetchHealth(); fetchModels(); fetchTools(); }, []);
    useEffect(() => { fetchSessions(); }, [activeAgentId]);
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isStreaming]);

    const fetchHealth = async () => {
        try {
            const data = await fetch('/api/health').then(r => r.json());
            setHealth(data);
        } catch { setHealth({ status: 'error', ollama: false }); }
    };

    const fetchModels = async () => {
        try {
            const data = await fetch('/api/models').then(r => r.json());
            if (data.models?.length) setModels(data.models);
        } catch { }
    };

    const fetchTools = async () => {
        try {
            const data = await fetch('/api/tools').then(r => r.json());
            if (data.tools?.length) setTools(data.tools);
        } catch { }
    };

    const fetchSessions = useCallback(async () => {
        try {
            const url = activeAgentId ? `/api/sessions?agent_id=${activeAgentId}` : '/api/sessions';
            const data = await fetch(url).then(r => r.json());
            setSessions(data.sessions || []);
        } catch { }
    }, [activeAgentId]);

    const loadSession = async (id: string) => {
        if (isSendingRef.current) return;
        try {
            const data = await fetch(`/api/sessions/${id}`).then(r => r.json());
            setCurrentSessionId(id);
            if (data.model) setCurrentModel(data.model);
            const loaded: Message[] = (data.history || []).map((m: any, i: number) => ({
                id: `hist-${i}`,
                role: m.role,
                content: m.content,
                blocks: [{ id: `b-${i}`, type: 'text' as const, content: m.content }],
            }));
            setMessages(loaded);
            setContextLines(data.context_lines || 0);
            fetchSessions();
        } catch (e) { console.error(e); }
    };

    const newSession = () => {
        if (isSendingRef.current) return;
        setCurrentSessionId(null);
        setMessages([]);
        setContextLines(0);
        fetchSessions();
    };

    const deleteSession = async (id: string) => {
        try {
            await fetch(`/api/sessions/${id}`, { method: 'DELETE' });
            if (id === currentSessionId) newSession();
            else fetchSessions();
        } catch { }
    };

    const addFiles = (filesList: File[]) => {
        const newAttachments = filesList.map(file => {
            const id = Math.random().toString(36).slice(2);
            const attachment: Attachment = { id, file, dataURL: null };
            if (file.type.startsWith('image/')) {
                const reader = new FileReader();
                reader.onload = e => setPendingFiles(prev =>
                    prev.map(p => p.id === id ? { ...p, dataURL: e.target?.result as string } : p)
                );
                reader.readAsDataURL(file);
            }
            return attachment;
        });
        setPendingFiles(prev => [...prev, ...newAttachments]);
    };

    const removeFile = (id: string) => setPendingFiles(prev => prev.filter(f => f.id !== id));

    const sendMessage = async (text: string) => {
        if (isSendingRef.current || (!text.trim() && !pendingFiles.length)) return;

        isSendingRef.current = true;
        setIsStreaming(true);

        const filesToSend = [...pendingFiles];
        setPendingFiles([]);

        const userMsgId = Date.now().toString();
        const assistantMsgId = (Date.now() + 1).toString();

        setMessages(prev => [
            ...prev,
            { id: userMsgId, role: 'user', content: text, files: filesToSend, blocks: [] },
            { id: assistantMsgId, role: 'assistant', content: '', blocks: [] },
        ]);

        try {
            const fd = new FormData();
            fd.append('message', text || '(see attached files)');
            if (currentSessionId) fd.append('session_id', currentSessionId);
            if (activeAgentId) fd.append('agent_id', activeAgentId);
            fd.append('model', currentModel);
            fd.append('search_backend', currentSearchBackend);
            fd.append('force_memory', forceMemory ? 'true' : 'false');
            filesToSend.forEach(f => fd.append('files', f.file));

            const res = await fetch('/api/chat/stream', { method: 'POST', body: fd });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const reader = res.body?.getReader();
            if (!reader) throw new Error('No reader');

            const decoder = new TextDecoder();
            let buf = '';

            // Track where the last text block started so we can retract safely
            let lastTextBlockStart = 0;

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });

                const lines = buf.split('\n');
                buf = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    let ev: any;
                    try { ev = JSON.parse(line.slice(6)); } catch { continue; }

                    setMessages(prev => {
                        const next = [...prev];
                        const msg = next[next.length - 1];
                        if (msg.role !== 'assistant') return prev;

                        const mkId = () => Math.random().toString(36).slice(2);
                        const addBlock = (block: Omit<MessageBlock, 'id'>) => {
                            msg.blocks = [...msg.blocks, { ...block, id: mkId() }];
                        };

                        const lastBlock = msg.blocks[msg.blocks.length - 1];

                        switch (ev.type) {
                            case 'session':
                                setCurrentSessionId(ev.session_id);
                                break;

                            case 'attachment':
                                addBlock({
                                    type: 'attachment_badge',
                                    content: ev.ok
                                        ? `✓ ${ev.filename} (${ev.file_type})`
                                        : `✗ ${ev.filename}: ${ev.error}`,
                                });
                                break;

                            case 'token':
                                msg.content += ev.content;
                                if (lastBlock?.type === 'text') {
                                    lastBlock.content += ev.content;
                                } else {
                                    // Record start position in msg.content for retract
                                    lastTextBlockStart = msg.content.length - ev.content.length;
                                    addBlock({ type: 'text', content: ev.content });
                                }
                                break;

                            case 'tool_call':
                                // FIX: show tool name + first arg value, not raw JSON blob
                                const argSummary = Object.entries(ev.arguments || {})
                                    .map(([k, v]) => `${k}: ${String(v).slice(0, 60)}`)
                                    .join(', ');
                                addBlock({
                                    type: 'tool_call',
                                    tool: ev.tool_name,
                                    content: argSummary,
                                });
                                break;

                            case 'tool_running':
                                // Optional: could update the last tool_call block to show spinner
                                break;

                            case 'tool_result':
                                // FIX: backend sends tool_name + success, not tool + content
                                addBlock({
                                    type: ev.success ? 'tool_result' : 'tool_error',
                                    tool: ev.tool_name,
                                    content: ev.success ? 'completed' : 'failed',
                                });
                                break;

                            case 'retract_last_tokens':
                                // FIX: only strip from the last text block, not the whole message
                                // Find and strip the trailing JSON object from the last text block
                                if (lastBlock?.type === 'text') {
                                    // Remove trailing JSON: find last '{' that starts a complete object
                                    const content = lastBlock.content;
                                    const jsonStart = content.lastIndexOf('\n{');
                                    if (jsonStart !== -1) {
                                        lastBlock.content = content.slice(0, jsonStart).trim();
                                        msg.content = msg.content.slice(0, msg.content.lastIndexOf('\n{')).trim();
                                    } else if (content.trimStart().startsWith('{')) {
                                        // Entire block is a JSON call — remove the block
                                        msg.blocks = msg.blocks.slice(0, -1);
                                        msg.content = msg.content.slice(0, lastTextBlockStart).trim();
                                    }
                                }
                                break;

                            case 'compressing':
                                addBlock({ type: 'compressing', content: 'compressing context…' });
                                break;

                            case 'context_updated':
                                msg.blocks = msg.blocks.filter(b => b.type !== 'compressing');
                                setContextLines(ev.lines);
                                fetchSessions();
                                break;

                            case 'error':
                                addBlock({ type: 'text', content: `\n\n⚠ ${ev.message}` });
                                break;
                        }

                        return next;
                    });
                }
            }
        } catch (e: any) {
            setMessages(prev => {
                const next = [...prev];
                const msg = next[next.length - 1];
                if (msg.role === 'assistant') {
                    msg.blocks = [...msg.blocks, {
                        id: 'err', type: 'text',
                        content: `\n\n⚠ connection error: ${e.message}`,
                    }];
                }
                return next;
            });
        } finally {
            setIsStreaming(false);
            isSendingRef.current = false;
        }
    };

    return {
        health, models, tools, sessions, currentSessionId,
        activeAgentId, setActiveAgentId,
        currentModel, setCurrentModel,
        currentSearchBackend, setCurrentSearchBackend,
        messages, contextLines,
        isStreaming, pendingFiles, forceMemory, setForceMemory,
        loadSession, newSession, deleteSession,
        addFiles, removeFile, sendMessage, bottomRef,
    };
}