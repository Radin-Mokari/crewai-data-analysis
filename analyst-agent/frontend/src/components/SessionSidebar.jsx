"use client";
import { useEffect } from 'react';
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

  useEffect(() => {
    fetch('/sessions')
      .then((r) => r.json())
      .then((data) => Array.isArray(data) ? setSessions(data) : null)
      .catch(() => {});
  }, []);

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

      {/* User Profile Footer */}
      <div style={{ padding: '16px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderTop: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ width: '32px', height: '32px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '12px', fontWeight: 'bold', background: '#d946ef', color: '#fff' }}>
            JD
          </div>
          <div style={{ display: 'flex', flexDirection: 'col' }}>
            <span style={{ fontSize: '14px', fontWeight: 500, color: 'var(--text-primary)', display: 'block' }}>John Doe</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block' }}>Pro Plan</span>
          </div>
        </div>
        <button style={{ padding: '6px', borderRadius: '6px', color: 'var(--text-muted)' }}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
          </svg>
        </button>
      </div>
    </div>
  );
}
