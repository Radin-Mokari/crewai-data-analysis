"use client";
import { useEffect, useRef } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';

export default function KernelView() {
  const cells = useSessionStore((s) => s.cells);
  const isRunning = useSessionStore((s) => s.isRunning);
  const connected = useSessionStore((s) => s.connected);
  const bottomRef = useRef(null);

  useEffect(() => { Prism.highlightAll(); }, [cells]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [cells]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden', background: 'var(--bg-right)', position: 'relative' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', flexShrink: 0, borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0, background: connected ? 'var(--accent-green)' : 'var(--accent-red)' }} />
          <span style={{ fontSize: '14px', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-primary)' }}>Python 3 (ipykernel)</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {isRunning && (
            <span
              style={{ fontSize: '10px', fontWeight: 'bold', letterSpacing: '0.1em', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-blue)', textTransform: 'uppercase' }}
            >
              <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
              </svg>
              AGENT: DEV AGENT IS RUNNING...
            </span>
          )}

          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-muted)' }}>
            <button style={{ padding: '6px', background: 'transparent' }} className="hover:text-[var(--text-primary)] transition-colors" aria-label="Kernel">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              </svg>
            </button>
            <button style={{ padding: '6px', background: 'transparent' }} className="hover:text-[var(--text-primary)] transition-colors" aria-label="Kernel">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.59-9.21l-5.44-5.44"></path>
              </svg>
            </button>
          </div>

          {isRunning ? (
            <button
              className="transition-colors hover:opacity-80"
              style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-red)', background: 'transparent' }}
              title="Stop Execution (UI only)"
              onClick={() => {}}
            >
              Stop Execution
            </button>
          ) : (
            <span style={{ fontSize: '10px', fontWeight: 500, padding: '4px 8px', color: 'var(--text-muted)' }}>
            </span>
          )}
        </div>
      </div>

      {/* Cells Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {cells.map((cell) => (
          <div key={cell.cell_id} style={{ display: 'flex', gap: '12px' }}>
            {/* Left sidebar of cell (Play button & execution count) */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: '8px', width: '32px', flexShrink: 0 }}>
              <button className="transition-colors" style={{ width: '28px', height: '28px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--text-primary)' }}>
                <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
              </button>
              <span style={{ fontSize: '10px', marginTop: '4px', fontFamily: 'monospace', color: 'var(--text-muted)' }}>[{cell.execution_count}]</span>
            </div>

            {/* Cell Content */}
            <div style={{ flex: 1, minWidth: 0 }}>
              {/* Code Box */}
              <div className="rounded-xl overflow-hidden" style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}>
                <div className="flex">
                  <div className="py-3 px-3 text-right select-none" style={{ color: 'var(--text-muted)', fontSize: '12px', fontFamily: 'monospace', minWidth: '36px', borderRight: '1px solid var(--border-light)' }}>
                    {cell.code.split('\n').map((_, i) => <div key={i}>{i + 1}</div>)}
                  </div>
                  <pre className="flex-1 m-0 p-3 overflow-x-auto" style={{ fontSize: '13px', lineHeight: '1.5' }}>
                    <code className="language-python">{cell.code}</code>
                  </pre>
                </div>
              </div>

              {/* Output Box */}
              {(cell.stdout || cell.stderr || cell.images?.length > 0) && (
                <div style={{ marginTop: '12px', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)', background: 'transparent' }}>
                  <div style={{ padding: '6px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '10px', fontWeight: 600, borderBottom: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                    OUTPUT
                    <button className="hover:text-[var(--text-primary)]" style={{ background: 'transparent' }}><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg></button>
                  </div>
                  <div className="p-3 text-[13px] font-mono" style={{ color: 'var(--text-primary)' }}>
                    {cell.stdout && <pre className="m-0 whitespace-pre-wrap">{cell.stdout}</pre>}
                    {cell.stderr && <pre className="m-0 whitespace-pre-wrap text-[var(--accent-red)]">{cell.stderr}</pre>}
                    {cell.images?.map((img, i) => (
                      <img key={i} src={`/files/charts/${img.split(/[\\/]/).pop()}`} alt="output" className="mt-2 rounded border border-[var(--border)] max-w-full" />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
        
        {/* Add Cell Button */}
        <div className="flex justify-center mt-4">
          <button className="px-4 py-2 rounded-full text-xs font-medium transition-colors flex items-center gap-2" style={{ border: '1px dashed var(--border-light)', color: 'var(--text-muted)' }} onMouseEnter={e => { e.currentTarget.style.borderColor = 'var(--text-secondary)'; e.currentTarget.style.color = 'var(--text-primary)'; }} onMouseLeave={e => { e.currentTarget.style.borderColor = 'var(--border-light)'; e.currentTarget.style.color = 'var(--text-muted)'; }}>
            + Code + Text
          </button>
        </div>
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
