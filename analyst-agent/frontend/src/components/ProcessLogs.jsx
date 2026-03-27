"use client";
import { useEffect, useRef } from 'react';
import { useSessionStore } from '../stores/sessionStore';

export default function ProcessLogs() {
  const logs = useSessionStore((s) => s.logs);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const getLogStyle = (log) => {
    if (log.type === 'quality_event') {
      if (log.content?.includes('FAIL')) return { label: 'WARN', color: 'var(--accent-orange)' };
      if (log.content?.includes('PASS')) return { label: 'SUCCESS', color: 'var(--accent-green)' };
    }
    if (log.type === 'agent_thought') return { label: 'INFO', color: 'var(--accent-blue)' };
    return { label: 'INFO', color: 'var(--text-muted)' };
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flexShrink: 0, background: 'var(--bg-right)', borderTop: '1px solid var(--border)', height: '220px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', flexShrink: 0, borderBottom: '1px solid var(--border)' }}>
        <span style={{ fontSize: '11px', fontWeight: 600, letterSpacing: '0.1em', color: 'var(--text-secondary)' }}>
          PROCESS LOGS
        </span>
        <button className="hover:text-[var(--text-primary)] transition-colors" style={{ color: 'var(--text-muted)', background: 'transparent' }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <line x1="9" y1="3" x2="9" y2="21"></line>
          </svg>
        </button>
      </div>

      {/* Logs Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', fontFamily: 'monospace', fontSize: '12px', lineHeight: '1.6' }}>
        {logs.length === 0 ? (
          <div style={{ color: 'var(--text-muted)' }}>Waiting for process...</div>
        ) : (
          logs.map((log, i) => {
            const style = getLogStyle(log);
            return (
              <div key={i} style={{ display: 'flex', gap: '12px', marginBottom: '6px' }}>
                <span style={{ flexShrink: 0, color: 'var(--text-muted)' }}>
                  [{log.timestamp?.slice(11, 19) || '00:00:00'}]
                </span>
                <span style={{ flexShrink: 0, fontWeight: 600, color: style.color, width: '60px' }}>
                  {style.label}:
                </span>
                <span style={{ wordBreak: 'break-word', color: 'var(--text-secondary)' }}>
                  {typeof log.content === 'string' ? log.content.slice(0, 300) : JSON.stringify(log.content).slice(0, 300)}
                </span>
              </div>
            );
          })
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
