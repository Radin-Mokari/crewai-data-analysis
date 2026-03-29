"use client";
import { useEffect, useRef, useState } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import Markdown from 'react-markdown';

const AGENT_CHIPS = [
  { name: 'visualization', label: 'Visualization' },
  { name: 'eda', label: 'EDA' },
  { name: 'feature_engineering', label: 'Feature Engineering' },
  { name: 'class_imbalance', label: 'Imbalance' },
  { name: 'report', label: 'Report' },
  { name: 'statistics', label: 'Statistics' },
  { name: 'cleaning', label: 'Cleaning' },
];

function Chip({ active, label, onClick }) {
  return (
    <button
      onClick={onClick}
      className="transition-colors"
      style={{
        padding: '6px 12px',
        borderRadius: '8px',
        fontSize: '12px',
        fontWeight: 500,
        background: active ? 'var(--surface-hover)' : 'transparent',
        border: `1px solid ${active ? 'var(--accent-blue)' : 'var(--border-light)'}`,
        color: active ? 'var(--text-primary)' : 'var(--text-secondary)',
      }}
    >
      {label}
    </button>
  );
}

export default function ChatSection() {
  const activeSession = useSessionStore((s) => s.activeSession);
  const logs = useSessionStore((s) => s.logs);
  const isRunning = useSessionStore((s) => s.isRunning);
  const setRunning = useSessionStore((s) => s.setRunning);
  const clearCurrent = useSessionStore((s) => s.clearCurrent);
  const setSessions = useSessionStore((s) => s.setSessions);
  const setCompletedSessionId = useSessionStore((s) => s.setCompletedSessionId);
  const bottomRef = useRef(null);

  const [prompt, setPrompt] = useState('');
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [showTools, setShowTools] = useState(false);
  const [uploading, setUploading] = useState(false);

  const [datasetPath, setDatasetPath] = useState('');
  const [datasetFileName, setDatasetFileName] = useState('');

  const agentThoughts = logs.filter((l) => l.type === 'agent_thought');
  const reportText = activeSession?.result?.task_summaries?.report;

  const resolvedDatasetPath = datasetPath || activeSession?.dataset_path || '';

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs, activeSession]);

  const refreshSessions = () => {
    fetch('/sessions')
      .then((r) => r.json())
      .then((data) => Array.isArray(data) && setSessions(data))
      .catch(() => {});
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file || uploading) return;
    setUploading(true);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await fetch('/upload', { method: 'POST', body: form });
      const data = await res.json();
      if (data?.file_path) {
        setDatasetPath(data.file_path);
        setDatasetFileName(data.filename);
      }
    } finally {
      setUploading(false);
    }
  };

  const start = async () => {
    if (isRunning) return;

    const ds = resolvedDatasetPath;
    if (!ds) return;

    const effectivePrompt = prompt.trim() || 'Analyze this dataset completely';

    // DO NOT clear current if using same active session (to keep history visible before refresh)
    if (!activeSession || activeSession.dataset_path !== ds) {
        clearCurrent();
    }
    
    setCompletedSessionId(null);
    setRunning(true);
    setPrompt(''); // clear input

    const params = new URLSearchParams({
      dataset_path: ds,
      prompt: effectivePrompt,
    });
    
    if (activeSession && activeSession.dataset_path === ds) {
        params.append('session_id', activeSession.id);
    }

    let endpoint = `/analyze?${params.toString()}`;
    if (selectedAgent) {
      const agentParams = new URLSearchParams({
        dataset_path: ds,
        agent_name: selectedAgent,
        prompt: effectivePrompt,
      });
      if (activeSession && activeSession.dataset_path === ds) {
          agentParams.append('session_id', activeSession.id);
      }
      endpoint = `/agent/run?${agentParams.toString()}`;
    }

    await fetch(endpoint, { method: 'POST' }).catch(() => {});
    refreshSessions();
  };

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', height: '100%', overflow: 'hidden', background: 'var(--bg-chat)', position: 'relative' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '16px 24px', flexShrink: 0, borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: 0 }}>
          <h1 style={{ fontSize: '18px', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-primary)', margin: 0 }}>
            {activeSession?.prompt || 'New Analysis'}
          </h1>
          <span style={{ padding: '2px 8px', borderRadius: '9999px', fontSize: '11px', fontWeight: 500, background: 'var(--surface-hover)', color: 'var(--text-secondary)' }}>
            Python
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)' }}>
          <button style={{ padding: '8px', background: 'transparent', transition: 'colors' }} className="hover:text-[var(--text-primary)]" aria-label="Share">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"></path>
              <polyline points="16 6 12 2 8 6"></polyline>
              <line x1="12" y1="2" x2="12" y2="15"></line>
            </svg>
          </button>
          <button style={{ padding: '8px', background: 'transparent', transition: 'colors' }} className="hover:text-[var(--text-primary)]" aria-label="Star">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon>
            </svg>
          </button>
        </div>
      </div>

      {/* Chat Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '24px 24px 160px 24px' }}>
        
        {/* Render History Messages from DB */}
        {activeSession?.messages?.map((msg, i) => {
          if (msg.role === 'user') {
            return (
              <div key={i} className="mb-8 flex justify-end">
                <div className="max-w-[80%] p-4 rounded-2xl rounded-tr-sm" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                  <p className="text-[15px] leading-relaxed" style={{ color: 'var(--text-primary)' }}>
                    {msg.content}
                  </p>
                  <div className="text-[11px] mt-2 text-right" style={{ color: 'var(--text-muted)' }}>
                    {new Date(msg.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </div>
            );
          } else {
            return (
              <div key={i} className="mb-8 flex gap-4">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-1" style={{ background: 'var(--accent-blue)', color: '#fff' }}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                    <circle cx="12" cy="5" r="2"></circle>
                    <path d="M12 7v4"></path>
                    <line x1="8" y1="16" x2="8" y2="16"></line>
                    <line x1="16" y1="16" x2="16" y2="16"></line>
                  </svg>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium mb-1" style={{ color: 'var(--text-primary)' }}>Dev Agent</div>
                  <div className="prose-dark mt-2 text-[15px]">
                    <Markdown>{msg.content}</Markdown>
                  </div>
                </div>
              </div>
            );
          }
        })}

        {/* Live execution / Streaming thoughts */}
        {(isRunning || agentThoughts.length > 0) && (
          <div className="mb-8 flex gap-4">
            <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-1" style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <rect x="3" y="11" width="18" height="10" rx="2"></rect>
                <circle cx="12" cy="5" r="2"></circle>
                <path d="M12 7v4"></path>
              </svg>
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-sm font-medium mb-2" style={{ color: 'var(--text-primary)' }}>Dev Agent Thinking...</div>

              <div className="flex flex-col gap-2 mb-4">
                {agentThoughts.map((t, idx) => {
                  let title = `AGENT THOUGHT STEP ${idx + 1}`;
                  if (typeof t.content === 'string') {
                    if (t.content.includes('OBSERVE')) title = 'Manager Analyzing Data State';
                    else if (t.content.includes('DECIDE: DELEGATE')) {
                      const match = t.content.match(/DELEGATE:\s*(\w+)/i);
                      title = match ? `Manager Delegating to ${match[1].toUpperCase()}` : 'Manager Delegating Task';
                    }
                    else if (t.content.includes('DECIDE: COMPLETE')) title = 'Manager Marked Analysis Complete';
                    else if (t.content.includes('DECIDE:')) title = 'Manager Making Decision';
                  }
                  
                  // Keep the latest thought open by default
                  const isOpen = idx === agentThoughts.length - 1;

                  return (
                    <details key={idx} className="rounded-xl border border-[var(--border)] overflow-hidden bg-[var(--surface)] group" open={isOpen}>
                      <summary className="px-3 py-2 cursor-pointer flex items-center justify-between text-[11.5px] font-semibold tracking-wide text-[var(--text-secondary)] hover:bg-[var(--surface-hover)] outline-none" style={{ color: t.content.includes('DECIDE') ? 'var(--accent-blue)' : 'var(--text-secondary)' }}>
                        <div className="flex items-center gap-2">
                          <svg className="transform transition-transform group-open:rotate-90" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <polyline points="9 18 15 12 9 6"></polyline>
                          </svg>
                          {title}
                        </div>
                      </summary>
                      <div className="px-4 pb-3 pt-1 text-[13px] leading-relaxed text-[var(--text-secondary)] border-t border-[var(--border-light)]">
                        <pre className="whitespace-pre-wrap font-sans m-0">{t.content}</pre>
                      </div>
                    </details>
                  );
                })}
              </div>

              {reportText && !isRunning ? (
                <div className="prose-dark mt-4">
                  <Markdown>{reportText}</Markdown>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-sm mt-3" style={{ color: 'var(--text-primary)' }}>
                  {isRunning ? (
                    <>
                      <div className="w-4 h-4 rounded-full border-2 border-[var(--accent-blue)] border-t-transparent animate-spin" />
                      Agent is working...
                    </>
                  ) : (
                    <>
                      <span>✨</span> Execution Finished
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Prompt Island */}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, padding: '48px 24px 24px 24px', background: 'linear-gradient(to top, var(--bg-chat) 70%, transparent)' }}>
        <div style={{ maxWidth: '48rem', margin: '0 auto', position: 'relative' }}>
          {/* Tools popover */}
          {showTools && (
            <div
              className="absolute bottom-full left-0 mb-3 rounded-xl p-4"
              style={{ background: 'var(--surface)', border: '1px solid var(--border)', width: '320px', zIndex: 30 }}
            >
              <div className="text-xs font-semibold" style={{ color: 'var(--text-muted)', marginBottom: '10px' }}>
                Tools
              </div>
              <div className="flex items-center gap-3" style={{ marginBottom: '12px' }}>
                <label
                  className="px-3 py-2 rounded-lg text-xs font-medium transition-colors"
                  style={{ background: 'var(--surface-hover)', color: 'var(--text-secondary)', border: '1px solid var(--border-light)' }}
                >
                  Upload CSV
                  <input type="file" accept=".csv" className="hidden" onChange={handleUpload} disabled={uploading} />
                </label>
                <div className="min-w-0 truncate text-[11px]" style={{ color: 'var(--text-muted)' }}>
                  {datasetFileName || (resolvedDatasetPath ? 'Using session dataset' : 'No dataset')}
                </div>
              </div>
              <div className="flex items-center justify-between">
                <button
                  onClick={() => setShowTools(false)}
                  className="px-3 py-2 rounded-lg text-xs font-medium transition-colors"
                  style={{ background: 'transparent', color: 'var(--text-secondary)', border: '1px solid var(--border-light)' }}
                >
                  Close
                </button>
              </div>
            </div>
          )}

          {/* Main Input Container */}
          <div style={{ display: 'flex', flexDirection: 'column', background: 'var(--surface)', borderRadius: '16px', border: '1px solid var(--border)', boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.25)' }}>
            
            {/* Top row: Agent Chips */}
            <div style={{ padding: '12px 16px 4px 16px', display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              <Chip active={!selectedAgent} label="Full Analysis" onClick={() => setSelectedAgent(null)} />
              {AGENT_CHIPS.map((c) => (
                <Chip key={c.name} active={selectedAgent === c.name} label={c.label} onClick={() => setSelectedAgent(c.name)} />
              ))}
            </div>

            {/* Middle: Textarea */}
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Ask the agent to edit code, explain concepts, or run commands..."
              className="resize-none outline-none"
              style={{ width: '100%', background: 'transparent', padding: '12px 16px', fontSize: '14px', color: 'var(--text-primary)', minHeight: '60px', border: 'none' }}
              rows={2}
              disabled={isRunning}
              onKeyDown={(e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') start();
              }}
            />

            {/* Bottom row: Tools & Send */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '4px 12px 12px 12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <button
                  onClick={() => setShowTools((v) => !v)}
                  className="transition-colors"
                  style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', borderRadius: '8px', fontSize: '12px', fontWeight: 500, color: 'var(--text-secondary)', background: 'transparent' }}
                  disabled={isRunning}
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <rect x="3" y="3" width="7" height="7"></rect>
                    <rect x="14" y="3" width="7" height="7"></rect>
                    <rect x="14" y="14" width="7" height="7"></rect>
                    <rect x="3" y="14" width="7" height="7"></rect>
                  </svg>
                  Tools
                </button>
                <button
                  className="p-1.5 rounded-lg transition-colors hover:bg-[var(--surface-hover)]"
                  style={{ color: 'var(--text-muted)' }}
                  onClick={() => setPrompt('')}
                  disabled={isRunning}
                  aria-label="Clear prompt"
                  title="Clear"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                  </svg>
                </button>
                <button
                  className="p-1.5 rounded-lg transition-colors hover:bg-[var(--surface-hover)]"
                  style={{ color: 'var(--text-muted)' }}
                  aria-label="Attach file"
                  title="Attach"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                    <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48" />
                  </svg>
                </button>
              </div>

              <button
                onClick={start}
                className="w-8 h-8 rounded-lg flex items-center justify-center transition-colors shadow-sm"
                style={{
                  background: !resolvedDatasetPath || isRunning ? 'var(--surface-hover)' : 'var(--accent-blue)',
                  color: !resolvedDatasetPath || isRunning ? 'var(--text-muted)' : '#fff',
                  cursor: !resolvedDatasetPath || isRunning ? 'not-allowed' : 'pointer',
                }}
                disabled={!resolvedDatasetPath || isRunning}
              >
                {isRunning ? (
                  <div className="w-4 h-4 rounded-full border-2 border-current border-t-transparent animate-spin" />
                ) : (
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="19" x2="12" y2="5" />
                    <polyline points="5 12 12 5 19 12" />
                  </svg>
                )}
              </button>
            </div>
          </div>
          
          <div className="text-center mt-3 text-[11px]" style={{ color: 'var(--text-muted)' }}>
            AI may produce inaccurate information.
          </div>
        </div>
      </div>
    </div>
  );
}
