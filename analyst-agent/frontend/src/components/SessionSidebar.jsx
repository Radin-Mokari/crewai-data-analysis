"use client";
import { useEffect, useState } from 'react';
import { useSessionStore } from '../stores/sessionStore';

function groupByDate(sessions) {
  const today = new Date().toDateString();
  const yesterday = new Date(Date.now() - 86400000).toDateString();
  const groups = { TODAY: [], YESTERDAY: [], EARLIER: [] };
  sessions.forEach((s) => {
    const d = new Date(s.created_at).toDateString();
    if (d === today) groups.TODAY.push(s);
    else if (d === yesterday) groups.YESTERDAY.push(s);
    else groups.EARLIER.push(s);
  });
  return groups;
}

export default function SessionSidebar() {
  const sessions = useSessionStore((s) => s.sessions);
  const setSessions = useSessionStore((s) => s.setSessions);
  const activeSession = useSessionStore((s) => s.activeSession);
  const setActiveSession = useSessionStore((s) => s.setActiveSession);
  const clearCurrent = useSessionStore((s) => s.clearCurrent);
  const connected = useSessionStore((s) => s.connected);
  const cells = useSessionStore((s) => s.cells);
  const logs = useSessionStore((s) => s.logs);

  const [debugInfo, setDebugInfo] = useState(null);

  useEffect(() => {
    fetch('/sessions')
      .then((r) => r.json())
      .then((data) => Array.isArray(data) ? setSessions(data) : null)
      .catch(() => {});
  }, []);

  const testWebSocket = async () => {
    try {
      const res = await fetch('/debug/test-ws', { method: 'POST' });
      const data = await res.json();
      setDebugInfo(data);
      console.log('[Debug] Test WS response:', data);
    } catch (e) {
      console.error('[Debug] Test WS failed:', e);
      setDebugInfo({ error: String(e) });
    }
  };

  const fetchDebugState = async () => {
    try {
      const res = await fetch('/debug/ws');
      const data = await res.json();
      setDebugInfo(data);
      console.log('[Debug] WS state:', data);
    } catch (e) {
      console.error('[Debug] Fetch state failed:', e);
      setDebugInfo({ error: String(e) });
    }
  };

  const loadSession = async (id) => {
    const res = await fetch(`/sessions/${id}`);
    const data = await res.json();
    setActiveSession(data);
  };

  const groups = groupByDate(sessions);

  return (
    <div
      className="flex flex-col h-full overflow-hidden flex-shrink-0"
      style={{
        width: '260px',
        background: 'var(--bg-sidebar)',
        borderRight: '1px solid var(--border)',
      }}
    >
      {/* New session button */}
      <div style={{ padding: '16px' }}>
        <button
          onClick={() => { clearCurrent(); setActiveSession(null); }}
          className="transition-opacity"
          style={{ 
            width: '100%', 
            padding: '10px 16px', 
            borderRadius: '6px', 
            fontSize: '14px', 
            fontWeight: 500, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center', 
            gap: '8px',
            background: 'var(--accent-blue)', 
            color: '#fff' 
          }}
          onMouseEnter={(e) => { e.currentTarget.style.opacity = '0.9'; }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = '1'; }}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
          </svg>
          New Session
        </button>
      </div>

      {/* List */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '0 8px 12px 8px' }}>
        {Object.entries(groups).map(([label, items]) =>
          items.length > 0 ? (
            <div key={label} style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '12px', fontWeight: 'bold', letterSpacing: '0.05em', padding: '6px 12px', color: 'var(--text-muted)' }}>
                {label}
              </div>
              {items.map((s) => {
                const isActive = activeSession?.id === s.id;
                return (
                  <button
                    key={s.id}
                    onClick={() => loadSession(s.id)}
                    className="transition-colors"
                    style={{ 
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 12px',
                      borderRadius: '6px',
                      fontSize: '13px',
                      marginBottom: '4px',
                      display: 'block',
                      background: isActive ? 'var(--surface)' : 'transparent',
                      color: isActive ? 'var(--text-primary)' : 'var(--text-secondary)'
                    }}
                    onMouseEnter={(e) => { 
                      if (!isActive) { e.currentTarget.style.background = 'var(--surface)'; e.currentTarget.style.color = 'var(--text-primary)'; } 
                    }}
                    onMouseLeave={(e) => { 
                      if (!isActive) { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-secondary)'; } 
                    }}
                  >
                    <span style={{ display: 'block', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {s.prompt?.slice(0, 30) || s.dataset_path?.split(/[\\/]/).pop() || 'Untitled'}
                    </span>
                  </button>
                );
              })}
            </div>
          ) : null,
        )}
      </div>

      {/* Debug Panel */}
      <div style={{ padding: '12px', borderTop: '1px solid var(--border)', background: 'var(--surface)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: connected ? 'var(--accent-green)' : 'var(--accent-red)'
          }} />
          <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
            WS: {connected ? 'Connected' : 'Disconnected'}
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: 'auto' }}>
            {cells.length} cells | {logs.length} logs
          </span>
        </div>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button
            onClick={testWebSocket}
            style={{
              flex: 1,
              padding: '6px',
              fontSize: '10px',
              borderRadius: '4px',
              background: 'var(--surface-hover)',
              color: 'var(--text-secondary)',
              border: '1px solid var(--border-light)'
            }}
          >
            Test WS
          </button>
          <button
            onClick={fetchDebugState}
            style={{
              flex: 1,
              padding: '6px',
              fontSize: '10px',
              borderRadius: '4px',
              background: 'var(--surface-hover)',
              color: 'var(--text-secondary)',
              border: '1px solid var(--border-light)'
            }}
          >
            Debug Info
          </button>
        </div>
        {debugInfo && (
          <pre style={{
            marginTop: '8px',
            fontSize: '9px',
            color: 'var(--text-muted)',
            overflow: 'auto',
            maxHeight: '80px',
            background: 'var(--bg-chat)',
            padding: '6px',
            borderRadius: '4px'
          }}>
            {JSON.stringify(debugInfo, null, 2)}
          </pre>
        )}
      </div>
    </div>
  );
}
