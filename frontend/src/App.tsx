import React, { useState, useRef, useEffect } from 'react';
import { useAgent } from './useAgent';
import { RichContentViewer } from './components/RichContentViewer';
import { Dashboard } from './Dashboard';
import { LogPanel } from './LogPanel';
import { VoiceControl } from './components/VoiceControl';
import { PremiumSelect } from './components/PremiumSelect';
import { ShovsView } from './components/ShovsView';
import { OptionsPanel } from './components/OptionsPanel';
import { GuardrailConfirmationModal } from './components/GuardrailConfirmationModal';

const AppViewer = ({ title, path }: { title: string, path: string }) => {
  return (
    <div className="v8-app-viewer" style={{
      width: '100%', border: '1px solid var(--border)', borderRadius: '12px', overflow: 'hidden', marginTop: '12px', background: '#000'
    }}>
      <div className="v8-app-header" style={{
        padding: '8px 16px', background: '#111', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-dim)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: 'var(--primary)' }}>◈</span>
          <span style={{ fontWeight: 600, color: 'var(--text)' }}>{title}</span>
        </div>
        <div>
          <a href={path} target="_blank" rel="noreferrer" style={{ color: 'var(--primary)', textDecoration: 'none' }}>open full ↗</a>
        </div>
      </div>
      <iframe src={path} style={{ width: '100%', height: '500px', border: 'none', display: 'block' }} title={title} />
    </div>
  );
};

const ToolEvent = ({ type, tool, content }: { type: 'call' | 'result' | 'error'; tool: string; content?: string }) => {
  const [expanded, setExpanded] = React.useState(false);

  let label = type === 'call' ? 'Calling' : type === 'result' ? 'Returned' : 'Failed';
  let summary = '';

  if (content && type === 'result') {
    try {
      const data = JSON.parse(content);
      if (data.type === 'web_search_results' && data.results) {
        label = `Found ${data.results.length} results`;
      } else if (data.type === 'web_fetch_result') {
        label = data.error ? 'Fetch Failed' : `Fetched ${data.total_length || 0} chars`;
      } else if (data.status === 'success' && data.type === 'app_view') {
        // App view special rendering
        return <AppViewer title={data.title} path={data.path} />;
      }
    } catch (e) {
      // Fallback
    }
  } else if (content && type === 'call') {
    summary = content.length > 50 ? content.substring(0, 50) + '...' : content;
  }

  const icon = type === 'call' ? '⚙' : type === 'result' ? '✓' : '✗';

  return (
    <div className={`tool-event ${type} ${expanded ? 'expanded' : ''}`}>
      <div className="tool-header" onClick={() => setExpanded(!expanded)}>
        <span className="tool-icon">{icon}</span>
        <span className="tool-text">
          <span className="tool-label">{label}</span>
          <span className="tool-name">{tool}</span>
          {summary && !expanded && <span className="tool-summary">· {summary}</span>}
        </span>
        <span className="tool-toggle">{expanded ? '▴' : '▾'}</span>
      </div>
      {expanded && content && (
        <div className="tool-details">
          <RichContentViewer content={content} />
        </div>
      )}
    </div>
  );
};

const ThoughtBlock = ({ content }: { content: string }) => {
  const [expanded, setExpanded] = React.useState(false);
  return (
    <div className={`thought-block ${expanded ? 'expanded' : ''}`}>
      <div className="thought-header" onClick={() => setExpanded(!expanded)}>
        <span className="thought-icon">⬡</span>
        <span className="thought-label">REASONING</span>
        <span className="thought-toggle">{expanded ? '▴' : '▾'}</span>
      </div>
      {expanded && (
        <div className="thought-content">
          <RichContentViewer content={content} />
        </div>
      )}
    </div>
  );
};

const PlanBlock = ({ content }: { content: string }) => {
  const [expanded, setExpanded] = React.useState(true);
  return (
    <div className={`plan-block ${expanded ? 'expanded' : ''}`}>
      <div className="plan-header" onClick={() => setExpanded(!expanded)}>
        <span className="plan-icon">◈</span>
        <span className="plan-label">STRATEGY</span>
        <span className="plan-toggle">{expanded ? '▴' : '▾'}</span>
      </div>
      {expanded && (
        <div className="plan-content">
          <RichContentViewer content={content} />
        </div>
      )}
    </div>
  );
};

function App() {
  const agent = useAgent();
  const [inputText, setInputText] = useState('');
  const [logOpen, setLogOpen] = useState(false);
  const [shovsMode, setShovsMode] = useState(false);
  const [toolMenuOpen, setToolMenuOpen] = useState(false);
  const [sidebarTab, setSidebarTab] = useState<'sessions' | 'options'>('sessions');
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const toolBtnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, [inputText]);

  const handleSend = () => {
    if (!inputText.trim() && agent.pendingFiles.length === 0) return;
    agent.sendMessage(inputText);
    setInputText('');
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  const StatusDot = () => {
    const state = agent.health.status === 'ok' ? 'ok' : 'error';
    return (
      <div className="status-pill">
        <div className={`dot ${agent.isStreaming ? 'busy' : state}`} />
        <span>{agent.isStreaming ? 'answering...' : agent.health.status === 'ok' ? 'connected' : 'connecting...'}</span>
      </div>
    );
  };

  if (!agent.activeAgentId) {
    return <Dashboard onSelectAgent={(id) => agent.setActiveAgentId(id)} />;
  }

  return (
    <div className={`layout ${logOpen ? 'log-open' : ''}`}>

      {/* Topbar */}
      <header className="topbar">
        <button className="home-btn" onClick={() => agent.setActiveAgentId(null)}>← Agents</button>
        <div className="branding">
          <span className="logo-a">SHOVS</span>
          <span className="logo-sep">//</span>
          <span className="logo-b">PLATFORM</span>
        </div>
        <StatusDot />
        <div className="topbar-right">
          <button
            className={`log-toggle-btn ${logOpen ? 'open' : ''}`}
            onClick={() => setLogOpen(o => !o)}
            title="Toggle internal log panel"
          >
            {logOpen && <span className="dot-live" />}
            ⬡ logs
          </button>
          <button
            className={`jarvis-toggle-btn ${shovsMode ? 'active' : ''}`}
            onClick={() => setShovsMode(o => !o)}
            title="Toggle immersive shovs Voice Mode"
          >
            {shovsMode ? '◈ HUD' : '◈ SHOVS'}
          </button>
          <span className="model-label">MODEL</span>
          <PremiumSelect
            value={agent.currentModel}
            options={agent.models}
            onChange={m => agent.setCurrentModel(m)}
          />
        </div>
      </header>

      {/* Sidebar */}
      <aside className="sidebar">
        {/* Sidebar Tab Switcher */}
        <div className="sidebar-tabs">
          <button
            className={`sidebar-tab ${sidebarTab === 'sessions' ? 'active' : ''}`}
            onClick={() => setSidebarTab('sessions')}
          >Sessions</button>
          <button
            className={`sidebar-tab ${sidebarTab === 'options' ? 'active' : ''}`}
            onClick={() => setSidebarTab('options')}
          >⚙ Options</button>
        </div>

        {sidebarTab === 'sessions' && (
          <>
            <div className="sidebar-head">
              <span className="sidebar-title">Sessions</span>
              <button className="btn-new" onClick={agent.newSession}>+ new</button>
            </div>
            <div className="session-list">
              {agent.sessions.length === 0 ? (
                <div style={{ padding: '12px 14px', fontSize: '10px', color: 'var(--text-dim)' }}>no sessions</div>
              ) : agent.sessions.map(s => (
                <div
                  key={s.id}
                  className={`s-item ${s.id === agent.currentSessionId ? 'active' : ''}`}
                  onClick={() => agent.loadSession(s.id)}
                >
                  <div className="s-title">{s.title || 'New Chat'}</div>
                  <div className="s-meta">{s.message_count} msg · {(s.model || '').split(':')[0]}</div>
                  <button className="s-delete" onClick={e => { e.stopPropagation(); agent.deleteSession(s.id); }}>✕</button>
                </div>
              ))}
            </div>
            <div className="ctx-panel">
              <div className="ctx-head">
                <span className="ctx-label">Context Engine</span>
                <span className={`ctx-state ${agent.contextLines > 0 ? 'warm' : 'cold'}`}>
                  ● {agent.contextLines > 0 ? 'warm' : 'cold'}
                </span>
              </div>
              <div className="ctx-bar-track">
                <div className="ctx-bar-fill" style={{ width: `${Math.min(100, (agent.contextLines / 80) * 100)}%` }} />
              </div>
              <div className="ctx-detail">
                {agent.contextLines === 0 ? 'no context loaded' : `${agent.contextLines} items`}
              </div>
              <div className="ctx-head" style={{ marginTop: '18px' }}>
                <span className="ctx-label">Tools</span>
                <span className="ctx-state warm">{agent.tools.length}</span>
              </div>
              <div className="ctx-detail">
                {agent.tools.length === 0 ? 'none' : agent.tools.map((t, i) => (
                  <React.Fragment key={t.name}>
                    <span title={t.description} style={{ cursor: 'help' }}>{t.name}</span>
                    {i < agent.tools.length - 1 && ' · '}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </>
        )}

        {sidebarTab === 'options' && (
          <OptionsPanel
            sessionId={agent.currentSessionId}
            contextLines={agent.contextLines}
            currentSearchEngine={agent.currentSearchEngine}
            setCurrentSearchEngine={agent.setCurrentSearchEngine}
            models={agent.models}
            usePlanner={agent.usePlanner}
            setUsePlanner={agent.setUsePlanner}
            plannerModel={agent.plannerModel}
            setPlannerModel={agent.setPlannerModel}
            contextModel={agent.contextModel}
            setContextModel={agent.setContextModel}
            contextMode={agent.contextMode}
            setSessionContextMode={agent.setSessionContextMode}
            clearSessionContext={agent.clearSessionContext}
            showPlannerLog={agent.showPlannerLog}
            setShowPlannerLog={agent.setShowPlannerLog}
            showActorThought={agent.showActorThought}
            setShowActorThought={agent.setShowActorThought}
            showObserverActivity={agent.showObserverActivity}
            setShowObserverActivity={agent.setShowObserverActivity}
          />
        )}
      </aside>

      {/* Main chat */}
      <main className="main">
        <div className="messages">
          {!agent.currentSessionId && agent.messages.length === 0 ? (
            <div className="cold-state">
              <div className="cold-glyph">◈</div>
              <div className="cold-text">Context engine cold</div>
              <div className="cold-hint">send a message to begin</div>
            </div>
          ) : agent.messages.map((m, idx) => (
            <div key={m.id || idx} className={`msg ${m.role}`}>
              <div className="msg-role">{m.role === 'user' ? 'YOU' : 'AGENT'}</div>
              <div className="msg-body">
                {m.files?.filter(f => f.dataURL).length ? (
                  <div className="msg-images">
                    {m.files!.filter(f => f.dataURL).map(f => (
                      <img key={f.id} className="msg-img" src={f.dataURL!} title={f.file.name} alt="attachment" />
                    ))}
                  </div>
                ) : null}
                {m.files?.filter(f => !f.dataURL).length ? (
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px', marginBottom: '6px' }}>
                    {m.files!.filter(f => !f.dataURL).map(f => (
                      <span key={f.id} className="attach-badge ok">📄 {f.file.name}</span>
                    ))}
                  </div>
                ) : null}
                {m.role === 'user' && !m.blocks?.length && (
                  <RichContentViewer content={m.content} />
                )}
                {m.blocks?.map(block => {
                  switch (block.type) {
                    case 'text':
                      return <RichContentViewer key={block.id} content={block.content} />;
                    case 'thought':
                      return agent.showActorThought ? <ThoughtBlock key={block.id} content={block.content} /> : null;
                    case 'plan':
                      return agent.showPlannerLog ? <PlanBlock key={block.id} content={block.content} /> : null;
                    case 'tool_call':
                      return <ToolEvent key={block.id} type="call" tool={block.tool || 'unknown'} content={block.content || ''} />;
                    case 'tool_result':
                      return <ToolEvent key={block.id} type="result" tool={block.tool || 'unknown'} content={block.content || ''} />;
                    case 'tool_error':
                      return <ToolEvent key={block.id} type="error" tool={block.tool || 'unknown'} content={block.content || ''} />;
                    case 'attachment_badge':
                      return <div key={block.id} className={`attach-badge ${block.content.startsWith('✓') ? 'ok' : 'err'}`}>{block.content}</div>;
                    case 'compressing':
                      return (
                        <div key={block.id} className="compressing-indicator">
                          <div className="spinner" /><span>compressing context…</span>
                        </div>
                      );
                    default: return null;
                  }
                })}
                {agent.isStreaming && m.role === 'assistant' && idx === agent.messages.length - 1 && <span className="cursor" />}
              </div>
            </div>
          ))}
          <div ref={agent.bottomRef} />
        </div>

        <div
          className="input-area"
          onDragOver={e => e.preventDefault()}
          onDrop={e => { e.preventDefault(); if (e.dataTransfer.files) agent.addFiles(Array.from(e.dataTransfer.files)); }}
        >
          <div className={`attach-preview ${(agent.pendingFiles.length > 0 || agent.forcedTools.length > 0) ? 'has-files' : ''}`}>
            {agent.forcedTools.map(tName => (
              <div key={`forced-${tName}`} className="attach-chip tool">
                <span style={{ marginRight: 4 }}>⚙</span>
                <span className="attach-chip-name">{tName}</span>
                <span className="attach-chip-remove" onClick={() => agent.setForcedTools(prev => prev.filter(n => n !== tName))}>✕</span>
              </div>
            ))}
            {agent.pendingFiles.map(f => (
              <div key={f.id} className={`attach-chip ${f.dataURL ? 'image' : ''}`}>
                {f.dataURL
                  ? <img src={f.dataURL} style={{ width: 18, height: 18, objectFit: 'cover', borderRadius: 1, marginRight: 2 }} alt="thumb" />
                  : <span style={{ color: 'var(--text-dim)', marginRight: 3 }}>📄</span>}
                <span className="attach-chip-name">{f.file.name}</span>
                <span className="attach-chip-remove" onClick={() => agent.removeFile(f.id)}>✕</span>
              </div>
            ))}
          </div>
          <div className="input-row">
            <VoiceControl
              isRecording={agent.isListening}
              status={agent.voiceStatus}
              onToggle={() => {
                if (agent.isListening) agent.stopRecording();
                else if (agent.speaking) agent.stopSpeaking();
                else agent.startRecording();
              }}
            />
            <div className="input-wrap">
              <span className="input-prefix">›</span>
              <textarea ref={textareaRef} placeholder="message…" rows={1}
                value={inputText} onChange={e => setInputText(e.target.value)} onKeyDown={handleKeyDown} />
            </div>

            <div className="tool-selector-wrap">
              <button
                ref={toolBtnRef}
                className={`btn-tool-trigger ${toolMenuOpen ? 'active' : ''}`}
                onClick={() => setToolMenuOpen(!toolMenuOpen)}
                title="Force specific tools"
              >
                ⚙
              </button>
              {toolMenuOpen && (
                <div className="tool-menu">
                  <div className="tool-menu-head">Force Tool</div>
                  {agent.tools.map(t => (
                    <div
                      key={t.name}
                      className={`tool-menu-item ${agent.forcedTools.includes(t.name) ? 'selected' : ''}`}
                      onClick={() => {
                        agent.setForcedTools(prev =>
                          prev.includes(t.name) ? prev.filter(n => n !== t.name) : [...prev, t.name]
                        );
                        setToolMenuOpen(false);
                      }}
                    >
                      <span className="t-name">{t.name}</span>
                      <span className="t-desc">{t.description.split('.')[0]}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <label className="btn-attach">⊕<input type="file" multiple style={{ display: 'none' }} onChange={e => e.target.files && agent.addFiles(Array.from(e.target.files))} /></label>
            <button className="btn-send" onClick={handleSend} disabled={agent.isStreaming || (!inputText.trim() && agent.pendingFiles.length === 0)}>Send</button>
          </div>
          <div className="input-footer">
            <span className="input-hint">Enter · Shift+Enter for newline</span>
            <div className="input-controls">
              <span className={`ctx-inline ${agent.contextLines > 0 ? 'warm' : 'cold'}`}>● context {agent.contextLines > 0 ? 'warm' : 'cold'}</span>
            </div>
          </div>
        </div>
      </main>

      {/* Log Panel — slides in as 3rd column */}
      <LogPanel
        sessionId={agent.currentSessionId}
        isOpen={logOpen}
        onClose={() => setLogOpen(false)}
      />

      {/* shovs Overlay */}
      {
        shovsMode && (
          <ShovsView
            onClose={() => setShovsMode(false)}
            isListening={agent.isListening}
            isThinking={agent.isStreaming && !agent.speaking}
            isSpeaking={agent.speaking}
            lastUserText={agent.lastUserText}
            currentAgentToken={agent.currentToken}
            lastAgentResponse={agent.lastAgentResponse}
            voiceSensitivity={agent.voiceSensitivity}
            setVoiceSensitivity={agent.setVoiceSensitivity}
            voiceModel={agent.voiceModel}
            setVoiceModel={agent.setVoiceModel}
            onToggleMic={() => {
              if (agent.isListening) {
                agent.stopRecording();
              } else if (agent.speaking) {
                agent.stopSpeaking();
              } else {
                agent.startRecording();
              }
            }}
          />
        )
      }

      {/* Guardrail Confirmation Modal */}
      {agent.pendingConfirmation && (
        <GuardrailConfirmationModal
          confirmation={agent.pendingConfirmation}
          onApprove={agent.approveConfirmation}
          onDeny={agent.denyConfirmation}
        />
      )}
    </div >
  );
}

export default App;
