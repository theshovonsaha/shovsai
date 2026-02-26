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
    type: 'text' | 'thought' | 'plan' | 'tool_call' | 'tool_result' | 'tool_error' | 'attachment_badge' | 'compressing';
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
    const [models, setModels] = useState<Record<string, string[]>>({ ollama: ['llama3.2'] });
    const [tools, setTools] = useState<any[]>([]);
    const [sessions, setSessions] = useState<Session[]>([]);
    const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
    const [activeAgentId, setActiveAgentId] = useState<string | null>(null);
    const [currentModel, setCurrentModel] = useState<string>(localStorage.getItem('shovs_model') || '');
    const [currentSearchBackend, setCurrentSearchBackend] = useState<string>(localStorage.getItem('shovs_search_backend') || 'auto');
    const [currentSearchEngine, setCurrentSearchEngine] = useState<string>(localStorage.getItem('shovs_search_engine') || 'duckduckgo');
    const [messages, setMessages] = useState<Message[]>([]);
    const [contextLines, setContextLines] = useState(0);
    const [isStreaming, setIsStreaming] = useState(false);
    const [pendingFiles, setPendingFiles] = useState<Attachment[]>([]);
    const [forcedTools, setForcedTools] = useState<string[]>([]);

    // V10 Layer Controls
    const [usePlanner, setUsePlanner] = useState<boolean>(localStorage.getItem('shovs_use_planner') !== 'false');
    const [plannerModel, setPlannerModel] = useState<string>(localStorage.getItem('shovs_planner_model') || '');
    const [contextModel, setContextModel] = useState<string>(localStorage.getItem('shovs_context_model') || 'deepseek-r1:8b');

    // Voice / Jarvis States
    const [isListening, setIsListening] = useState(false);
    const [speaking, setSpeaking] = useState(false);
    const [lastUserText, setLastUserText] = useState('');
    const [currentToken, setCurrentToken] = useState('');
    const [lastAgentResponse, setLastAgentResponse] = useState('');
    const [voiceStatus, setVoiceStatus] = useState<'idle' | 'recording' | 'processing' | 'speaking'>('idle');

    const wsRef = useRef<WebSocket | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const ttsChunksRef = useRef<ArrayBuffer[]>([]);

    const isSendingRef = useRef(false);
    const bottomRef = useRef<HTMLDivElement>(null);

    useEffect(() => { fetchHealth(); fetchModels(); fetchTools(); }, []);
    useEffect(() => { fetchSessions(); }, [activeAgentId]);

    useEffect(() => {
        if (currentModel) localStorage.setItem('shovs_model', currentModel);
    }, [currentModel]);

    useEffect(() => {
        localStorage.setItem('shovs_search_backend', currentSearchBackend);
    }, [currentSearchBackend]);

    useEffect(() => {
        localStorage.setItem('shovs_search_engine', currentSearchEngine);
    }, [currentSearchEngine]);

    useEffect(() => {
        localStorage.setItem('shovs_use_planner', usePlanner.toString());
    }, [usePlanner]);

    useEffect(() => {
        localStorage.setItem('shovs_planner_model', plannerModel);
    }, [plannerModel]);

    useEffect(() => {
        localStorage.setItem('shovs_context_model', contextModel);
    }, [contextModel]);

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
            if (data.models) {
                setModels(data.models);
                // If no model is current, pick the first available one
                if (!currentModel) {
                    const providers = Object.keys(data.models);
                    for (const p of providers) {
                        if (data.models[p].length > 0) {
                            setCurrentModel(`${p}:${data.models[p][0]}`);
                            break;
                        }
                    }
                }
            }
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

    const clearSessionContext = async () => {
        if (!currentSessionId) return;
        try {
            await fetch(`/api/sessions/${currentSessionId}/clear_context`, { method: 'POST' });
            setContextLines(0);
        } catch (e) {
            console.error('Failed to clear context', e);
        }
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
            fd.append('search_engine', currentSearchEngine); // PASS TO BACKEND!
            fd.append('planner_model', plannerModel);
            fd.append('context_model', contextModel);
            fd.append('use_planner', usePlanner.toString());

            fd.append('forced_tools_json', JSON.stringify(forcedTools));
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

                            case 'plan':
                                addBlock({
                                    type: 'plan',
                                    content: ev.strategy || 'Planning strategy...'
                                });
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
                                let token = ev.content;
                                msg.content += token;

                                // Thinking Tag Detection Logic
                                if (token.includes('<think>')) {
                                    const parts = token.split('<think>');
                                    if (parts[0]) {
                                        if (lastBlock?.type === 'text') lastBlock.content += parts[0];
                                        else addBlock({ type: 'text', content: parts[0] });
                                    }
                                    addBlock({ type: 'thought', content: parts[1] || '' });
                                } else if (token.includes('</think>')) {
                                    const parts = token.split('</think>');
                                    if (lastBlock?.type === 'thought') {
                                        lastBlock.content += parts[0];
                                    }
                                    addBlock({ type: 'text', content: parts[1] || '' });
                                } else {
                                    if (lastBlock?.type === 'text' || lastBlock?.type === 'thought') {
                                        lastBlock.content += token;
                                    } else {
                                        lastTextBlockStart = msg.content.length - token.length;
                                        addBlock({ type: 'text', content: token });
                                    }
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
                                addBlock({
                                    type: ev.success ? 'tool_result' : 'tool_error',
                                    tool: ev.tool_name,
                                    content: ev.content || (ev.success ? 'completed' : 'failed'),
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

    const startRecording = async () => {
        if (!activeAgentId) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            mediaRecorderRef.current = recorder;

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
                    wsRef.current.send(e.data);
                }
            };

            recorder.onstop = () => {
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({ type: 'stt_end' }));
                }
                stream.getTracks().forEach(track => track.stop());
            };

            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/ws/voice`;
            const ws = new WebSocket(wsUrl);
            ws.binaryType = 'arraybuffer';
            wsRef.current = ws;

            ws.onopen = () => {
                ws.send(JSON.stringify({
                    type: 'config',
                    session_id: currentSessionId,
                    agent_id: activeAgentId,
                    model: currentModel
                }));
            };

            ws.onmessage = async (e) => {
                if (typeof e.data === 'string') {
                    const msg = JSON.parse(e.data);
                    switch (msg.type) {
                        case 'config_ack':
                            recorder.start(250);
                            setIsListening(true);
                            setVoiceStatus('recording');
                            setLastUserText('');
                            break;
                        case 'stt_result':
                            if (msg.text) {
                                setLastUserText(msg.text);

                                if (msg.is_final) {
                                    setVoiceStatus('processing');
                                } else {
                                    setVoiceStatus('recording');
                                }

                                // Barge-in: If we are speaking and user starts talking, stop!
                                if (speaking) {
                                    stopSpeaking();
                                }

                                if (msg.is_final) {
                                    // Mirror to chat history only when final
                                    setMessages(prev => [...prev, {
                                        id: 'u-' + Date.now(),
                                        role: 'user',
                                        content: msg.text,
                                        blocks: [{ id: 'b-' + Date.now(), type: 'text', content: msg.text }]
                                    }]);
                                    setLastAgentResponse('');
                                    setCurrentToken('');
                                }
                            }
                            break;
                        case 'agent_token':
                            setCurrentToken(prev => prev + msg.content);
                            break;
                        case 'agent_done':
                            setLastAgentResponse(msg.full_response);
                            setCurrentToken('');
                            // Mirror agent response to chat history
                            setMessages(prev => [...prev, {
                                id: 'a-' + Date.now(),
                                role: 'assistant',
                                content: msg.full_response,
                                blocks: [{ id: 'ba-' + Date.now(), type: 'text', content: msg.full_response }]
                            }]);
                            break;
                        case 'tts_start':
                            setSpeaking(true);
                            setVoiceStatus('speaking');
                            ttsChunksRef.current = [];
                            break;
                        case 'tts_end':
                            setSpeaking(false);
                            setVoiceStatus('idle');
                            if (ttsChunksRef.current.length > 0) {
                                playBufferedAudio();
                            }
                            if (wsRef.current) wsRef.current.close();
                            break;
                        case 'error':
                            console.error('Voice Error:', msg.message);
                            stopRecording();
                            setSpeaking(false);
                            setVoiceStatus('idle');
                            break;
                    }
                } else {
                    // Binary audio data (TTS)
                    if (e.data instanceof ArrayBuffer) {
                        ttsChunksRef.current.push(e.data);
                    }
                }
            };

            ws.onclose = () => {
                setIsListening(false);
                setVoiceStatus('idle');
            };

        } catch (err: any) {
            console.error('Mic error:', err);
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        setIsListening(false);
    };

    const stopSpeaking = () => {
        if (wsRef.current) wsRef.current.close();
        setSpeaking(false);
        setVoiceStatus('idle');
    };

    const playBufferedAudio = async () => {
        if (ttsChunksRef.current.length === 0) return;

        // Merge chunks
        const totalLen = ttsChunksRef.current.reduce((acc, c) => acc + c.byteLength, 0);
        const merged = new Uint8Array(totalLen);
        let offset = 0;
        for (const chunk of ttsChunksRef.current) {
            merged.set(new Uint8Array(chunk), offset);
            offset += chunk.byteLength;
        }
        ttsChunksRef.current = [];

        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        const ctx = audioContextRef.current;
        try {
            const buffer = await ctx.decodeAudioData(merged.buffer);
            const source = ctx.createBufferSource();
            source.buffer = buffer;
            source.connect(ctx.destination);
            source.start();
        } catch (e) {
            console.error('Failed to decode/play buffered audio:', e);
        }
    };

    return {
        health, models, tools, sessions, currentSessionId,
        activeAgentId, setActiveAgentId,
        currentModel, setCurrentModel,
        currentSearchBackend, setCurrentSearchBackend,
        currentSearchEngine, setCurrentSearchEngine,
        messages, contextLines,
        isStreaming, pendingFiles,
        forcedTools, setForcedTools,
        isListening, speaking, voiceStatus,
        lastUserText, currentToken, lastAgentResponse,
        startRecording, stopRecording, stopSpeaking,
        usePlanner, setUsePlanner,
        plannerModel, setPlannerModel,
        contextModel, setContextModel,
        clearSessionContext,
        loadSession, newSession, deleteSession,
        addFiles, removeFile, sendMessage, bottomRef,
    };
}