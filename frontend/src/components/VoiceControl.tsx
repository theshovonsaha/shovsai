import React, { useState, useRef } from 'react';

interface VoiceControlProps {
    sessionId: string | null;
    agentId: string;
    model: string;
}

export const VoiceControl: React.FC<VoiceControlProps> = ({ sessionId, agentId, model }) => {
    const [isRecording, setIsRecording] = useState(false);
    const [status, setStatus] = useState<'idle' | 'recording' | 'processing' | 'speaking'>('idle');
    const [error, setError] = useState<string | null>(null);

    const wsRef = useRef<WebSocket | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const audioContextRef = useRef<AudioContext | null>(null);

    const toggleRecording = async () => {
        if (isRecording) {
            stopRecording();
        } else {
            startRecording();
        }
    };

    const startRecording = async () => {
        try {
            setError(null);
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const recorder = new MediaRecorder(stream);
            mediaRecorderRef.current = recorder;
            audioChunksRef.current = [];

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunksRef.current.push(e.data);
                    if (wsRef.current?.readyState === WebSocket.OPEN) {
                        wsRef.current.send(e.data);
                    }
                }
            };

            recorder.onstop = () => {
                if (wsRef.current?.readyState === WebSocket.OPEN) {
                    wsRef.current.send(JSON.stringify({ type: 'stt_end' }));
                }
                stream.getTracks().forEach(track => track.stop());
            };

            // Connect WebSocket
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/api/ws/voice`;
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                ws.send(JSON.stringify({
                    type: 'config',
                    session_id: sessionId,
                    agent_id: agentId,
                    model: model
                }));
            };

            ws.onmessage = async (e) => {
                if (typeof e.data === 'string') {
                    const msg = JSON.parse(e.data);
                    switch (msg.type) {
                        case 'config_ack':
                            recorder.start(250); // Send chunks every 250ms
                            setIsRecording(true);
                            setStatus('recording');
                            break;
                        case 'stt_result':
                            if (msg.text) setStatus('processing');
                            break;
                        case 'tts_start':
                            setStatus('speaking');
                            break;
                        case 'tts_end':
                            setStatus('idle');
                            break;
                        case 'error':
                            setError(msg.message);
                            stopRecording();
                            break;
                    }
                } else {
                    // Binary audio data (TTS)
                    playAudioChunk(e.data);
                }
            };

            ws.onerror = () => {
                setError('WebSocket connection failed');
                stopRecording();
            };

            ws.onclose = () => {
                stopRecording();
            };

        } catch (err: any) {
            setError(err.message || 'Microphone access denied');
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop();
        }
        if (wsRef.current) {
            wsRef.current.close();
        }
        setIsRecording(false);
        setStatus('idle');
    };

    const playAudioChunk = async (data: ArrayBuffer) => {
        if (!audioContextRef.current) {
            audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
        }
        const ctx = audioContextRef.current;
        try {
            const buffer = await ctx.decodeAudioData(data);
            const source = ctx.createBufferSource();
            source.buffer = buffer;
            source.connect(ctx.destination);
            source.start();
        } catch (e) {
            console.error('Failed to decode/play audio chunk:', e);
        }
    };

    return (
        <div className="voice-control">
            <button
                className={`voice-btn ${status}`}
                onClick={toggleRecording}
                title={isRecording ? 'Stop Recording' : 'Start Voice Conversation'}
            >
                {status === 'idle' && <span>🎤</span>}
                {status === 'recording' && <span className="pulse">🔴</span>}
                {status === 'processing' && <span className="spin">⚙</span>}
                {status === 'speaking' && <span className="wave">🔊</span>}
            </button>
            {error && <div className="voice-error">{error}</div>}
        </div>
    );
};
