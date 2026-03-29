"use client";
import { useEffect, useRef, useState } from 'react';
import { useSessionStore } from '../stores/sessionStore';
import Prism from 'prismjs';
import 'prismjs/components/prism-python';
import ReactMarkdown from 'react-markdown';

function EditableCodeBlock({ code, cellType, cellId, onRun, onChange, onDelete }) {
  const [isEditing, setIsEditing] = useState(false);
  const [value, setValue] = useState(code || '');

  useEffect(() => {
    if (!isEditing) setValue(code || '');
  }, [code, isEditing]);

  const handleBlur = () => {
    setIsEditing(false);
    if (value !== code) {
      onChange(cellId, value);
    }
  };

  const handleKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      setIsEditing(false);
      onRun(cellId, value);
    }
  };

  const isMarkdown = cellType === 'markdown';

  return (
    <div className="rounded-xl overflow-hidden relative group" style={{ background: isMarkdown ? 'transparent' : 'var(--surface)', border: isMarkdown ? '1px solid transparent' : '1px solid var(--border)' }}>
      {/* Absolute action bar top right */}
      <div className="absolute top-2 right-2 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity z-10">
        {!isEditing && (
          <>
            <button onClick={() => setIsEditing(true)} className="p-1.5 rounded bg-[var(--surface-hover)] border border-[var(--border)] text-[var(--text-secondary)] hover:text-white" title="Edit">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
            </button>
            <button onClick={() => onDelete(cellId)} className="p-1.5 rounded bg-[rgba(248,81,73,0.1)] border border-[rgba(248,81,73,0.3)] text-[var(--accent-red)] hover:bg-[var(--accent-red)] hover:text-white" title="Delete">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
            </button>
          </>
        )}
      </div>

      {isMarkdown ? (
        <div className="p-2" onDoubleClick={() => setIsEditing(true)}>
          {isEditing ? (
            <textarea
              autoFocus
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onBlur={handleBlur}
              onKeyDown={handleKeyDown}
              className="w-full m-0 p-3 outline-none resize-none rounded-md"
              style={{ fontSize: '14px', lineHeight: '1.6', background: 'var(--surface)', color: 'var(--text-primary)', border: '1px solid var(--border-light)', minHeight: `${Math.max(((value || '').split('\n').length)*24, 60)}px` }}
            />
          ) : (
            <div className="prose prose-sm dark:prose-invert max-w-none min-h-[40px] p-2 hover:bg-[var(--surface-hover)] rounded-md cursor-pointer transition-colors">
              <ReactMarkdown>{code || '*Double click to edit text*'}</ReactMarkdown>
            </div>
          )}
        </div>
      ) : (
        <div className="flex">
          <div className="py-3 px-3 text-right select-none" style={{ color: 'var(--text-muted)', fontSize: '12px', fontFamily: 'monospace', minWidth: '36px', borderRight: '1px solid var(--border-light)' }}>
            {(isEditing ? value : (code || '')).split('\n').map((_, i) => <div key={i}>{i + 1}</div>)}
          </div>
          
          {isEditing ? (
            <textarea
              autoFocus
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onBlur={handleBlur}
              onKeyDown={handleKeyDown}
              className="flex-1 m-0 p-3 outline-none resize-none"
              style={{ fontSize: '13px', lineHeight: '1.5', background: 'transparent', color: 'var(--text-primary)', fontFamily: 'monospace', minHeight: `${Math.max(((value || '').split('\n').length)*24, 60)}px` }}
            />
          ) : (
            <pre className="flex-1 m-0 p-3 overflow-x-auto" style={{ fontSize: '13px', lineHeight: '1.5' }} onDoubleClick={() => setIsEditing(true)}>
              <code className="language-python">{code || ''}</code>
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// Agent badge colors
const AGENT_COLORS = {
  inspection: '#6366f1',  // indigo
  cleaning: '#22c55e',    // green
  eda: '#3b82f6',         // blue
  visualization: '#a855f7', // purple
  statistics: '#f97316',  // orange
  feature_engineering: '#14b8a6', // teal
  class_imbalance: '#ec4899', // pink
  report: '#6b7280',      // gray
  validation: '#64748b',  // slate
  system: '#475569',      // dark slate
  user: '#0ea5e9',        // sky blue
};

export default function KernelView() {
  const activeSession = useSessionStore((s) => s.activeSession);
  const cells = useSessionStore((s) => s.cells);
  const isRunning = useSessionStore((s) => s.isRunning);
  const connected = useSessionStore((s) => s.connected);
  const setRunning = useSessionStore((s) => s.setRunning);
  const bottomRef = useRef(null);
  const [kernelStats, setKernelStats] = useState(null);

  useEffect(() => { Prism.highlightAll(); }, [cells]);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [cells]);

  // Fetch kernel stats periodically
  useEffect(() => {
    if (activeSession?.id && !isRunning) {
      console.log('[KernelView] Loading cells for session:', activeSession.id);
      const ds = activeSession?.dataset_path ? `?dataset_path=${encodeURIComponent(activeSession.dataset_path)}` : '';
      fetch(`/kernel/cells${ds}`)
        .then(r => r.json())
        .then(data => {
          if (Array.isArray(data)) {
            useSessionStore.setState({ cells: data });
          }
        })
        .catch(e => console.error('[KernelView] Failed to load previous cells:', e));
    }
  }, [activeSession?.id, isRunning]);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const res = await fetch('/kernel/stats');
        if (res.ok) {
          const data = await res.json();
          setKernelStats(data);
        }
      } catch (e) {
        console.error('[KernelView] Failed to fetch stats:', e);
      }
    };
    fetchStats();
    const interval = setInterval(fetchStats, 10000); // Every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const handleInterrupt = async () => {
    try {
      const res = await fetch('/kernel/interrupt', { method: 'POST' });
      const data = await res.json();
      console.log('[KernelView] Interrupt result:', data);
      if (data.success) {
        setRunning(false);
      }
    } catch (e) {
      console.error('[KernelView] Interrupt failed:', e);
    }
  };

  const handleRestart = async () => {
    if (!confirm('Restart kernel? This will clear all variables.')) return;
    try {
      const ds = activeSession?.dataset_path ? `?dataset_path=${encodeURIComponent(activeSession.dataset_path)}` : '';
      const res = await fetch(`/kernel/restart${ds}`, { method: 'POST' });
      const data = await res.json();
      console.log('[KernelView] Restart result:', data);
    } catch (e) {
      console.error('[KernelView] Restart failed:', e);
    }
  };

  const handleRunAllCells = async () => {
    try {
      setRunning(true);
      const ds = activeSession?.dataset_path ? `?dataset_path=${encodeURIComponent(activeSession.dataset_path)}` : '';
      const res = await fetch(`/kernel/run-all${ds}`, { method: 'POST' });
      const data = await res.json();
      console.log('[KernelView] Run all result:', data);
    } catch (e) {
      console.error('[KernelView] Run all failed:', e);
    } finally {
      setRunning(false);
    }
  };

  const handleRerunCell = async (cellId, newCode = null) => {
    try {
      const ds = activeSession?.dataset_path ? `?dataset_path=${encodeURIComponent(activeSession.dataset_path)}` : '';
      if (newCode !== null) {
        await fetch(`/kernel/cells/${cellId}/edit-and-rerun${ds}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ code: newCode })
        });
      } else {
        await fetch(`/kernel/cells/${cellId}/rerun${ds}`, { method: 'POST' });
      }
    } catch (e) {
      console.error('[KernelView] Rerun cell failed:', e);
    }
  };

  const handleEditCell = async (cellId, newCode) => {
    try {
      const ds = activeSession?.dataset_path ? `?dataset_path=${encodeURIComponent(activeSession.dataset_path)}` : '';
      await fetch(`/kernel/cells/${cellId}${ds}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: newCode })
      });
    } catch (e) {
      console.error('[KernelView] Edit failed:', e);
    }
  };

  const handleDeleteCell = async (cellId) => {
    if (!confirm('Delete this cell?')) return;
    try {
      const ds = activeSession?.dataset_path ? `?dataset_path=${encodeURIComponent(activeSession.dataset_path)}` : '';
      await fetch(`/kernel/cells/${cellId}${ds}`, { method: 'DELETE' });
    } catch (e) {
      console.error('[KernelView] Delete failed:', e);
    }
  };

  const handleAddCell = async () => {
    try {
      await fetch(`/kernel/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          code: "# New code cell\n", 
          agent_name: "user", 
          cell_type: "code",
          dataset_path: activeSession?.dataset_path || null
        })
      });
    } catch (e) {
      console.error('[KernelView] Add code cell failed:', e);
    }
  };

  const handleAddTextCell = async () => {
    try {
      await fetch(`/kernel/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          code: "Enter your markdown text here...", 
          agent_name: "user", 
          cell_type: "markdown",
          dataset_path: activeSession?.dataset_path || null
        })
      });
    } catch (e) {
      console.error('[KernelView] Add text cell failed:', e);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden', background: 'var(--bg-right)', position: 'relative' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', flexShrink: 0, borderBottom: '1px solid var(--border)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: 0 }}>
          <div style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            flexShrink: 0,
            background: kernelStats?.is_alive ? 'var(--accent-green)' : (connected ? 'var(--accent-orange)' : 'var(--accent-red)')
          }} />
          <span style={{ fontSize: '14px', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-primary)' }}>
            Python 3 (ipykernel)
          </span>
          {kernelStats?.uptime > 0 && (
            <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
              | {Math.floor(kernelStats.uptime / 60)}m {Math.floor(kernelStats.uptime % 60)}s
            </span>
          )}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          {isRunning && (
            <span
              style={{ fontSize: '10px', fontWeight: 'bold', letterSpacing: '0.1em', display: 'flex', alignItems: 'center', gap: '6px', color: 'var(--accent-blue)', textTransform: 'uppercase' }}
            >
              <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
              </svg>
              Running...
            </span>
          )}

          <div style={{ display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-muted)' }}>
            {/* Run All Cells */}
            <button
              onClick={handleRunAllCells}
              disabled={isRunning}
              style={{ padding: '6px', background: 'transparent', cursor: isRunning ? 'not-allowed' : 'pointer', opacity: isRunning ? 0.5 : 1 }}
              className="hover:text-[var(--text-primary)] transition-colors"
              aria-label="Run All Cells"
              title="Run All Cells"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
            </button>
            {/* Restart Kernel */}
            <button
              onClick={handleRestart}
              style={{ padding: '6px', background: 'transparent' }}
              className="hover:text-[var(--text-primary)] transition-colors"
              aria-label="Restart Kernel"
              title="Restart Kernel"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.59-9.21l-5.44-5.44"></path>
              </svg>
            </button>
          </div>

          {isRunning ? (
            <button
              onClick={handleInterrupt}
              className="transition-colors hover:opacity-80"
              style={{ fontSize: '12px', fontWeight: 600, color: 'var(--accent-red)', background: 'transparent' }}
              title="Interrupt Kernel"
            >
              Interrupt
            </button>
          ) : (
            <span style={{ fontSize: '10px', fontWeight: 500, padding: '4px 8px', color: 'var(--text-muted)' }}>
              {cells.length} cells
            </span>
          )}
        </div>
      </div>

      {/* Cells Area */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        {cells.length === 0 && (
          <div style={{ textAlign: 'center', padding: '40px', color: 'var(--text-muted)' }}>
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" style={{ margin: '0 auto 16px', opacity: 0.5 }}>
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <line x1="3" y1="9" x2="21" y2="9"></line>
              <line x1="9" y1="21" x2="9" y2="9"></line>
            </svg>
            <p style={{ fontSize: '14px', marginBottom: '8px' }}>No cells yet</p>
            <p style={{ fontSize: '12px' }}>Upload a dataset and start an analysis to see code cells here</p>
          </div>
        )}

        {cells.map((cell) => (
          <div key={cell.cell_id} style={{ display: 'flex', gap: '12px' }}>
            {/* Left sidebar of cell (Play button & execution count) */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', paddingTop: '8px', width: '32px', flexShrink: 0 }}>
              <button
                onClick={() => handleRerunCell(cell.cell_id)}
                className="transition-colors"
                style={{
                  width: '28px',
                  height: '28px',
                  borderRadius: '50%',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'var(--surface)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-primary)',
                  cursor: 'pointer'
                }}
                title="Re-run cell"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
              </button>
              <span style={{ fontSize: '10px', marginTop: '4px', fontFamily: 'monospace', color: 'var(--text-muted)' }}>[{cell.execution_count}]</span>
            </div>

            {/* Cell Content */}
            <div style={{ flex: 1, minWidth: 0 }}>
              {/* Agent Badge */}
              <div style={{ marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{
                  fontSize: '10px',
                  fontWeight: 600,
                  padding: '2px 8px',
                  borderRadius: '9999px',
                  background: AGENT_COLORS[cell.agent] || '#6b7280',
                  color: '#fff',
                  textTransform: 'uppercase',
                  letterSpacing: '0.05em'
                }}>
                  {cell.agent}
                </span>
                {!cell.success && (
                  <span style={{ fontSize: '10px', color: 'var(--accent-red)', fontWeight: 500 }}>
                    Error
                  </span>
                )}
              </div>

              {/* Editable Code Box */}
              <EditableCodeBlock
                code={cell.code}
                cellType={cell.cell_type}
                cellId={cell.cell_id}
                onRun={handleRerunCell}
                onChange={handleEditCell}
                onDelete={handleDeleteCell}
              />

              {/* Output Box */}
              {(cell.stdout || cell.stderr || cell.html || cell.svg || cell.latex || cell.markdown || cell.images?.length > 0) && (
                <div style={{ marginTop: '12px', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border)', background: 'transparent' }}>
                  <div style={{ padding: '6px 12px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '10px', fontWeight: 600, borderBottom: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                    OUTPUT
                    <div style={{ display: 'flex', gap: '4px' }}>
                      {cell.images?.length > 0 && (
                        <span style={{ background: 'var(--surface)', padding: '2px 6px', borderRadius: '4px', fontSize: '9px' }}>
                          {cell.images.length} image{cell.images.length > 1 ? 's' : ''}
                        </span>
                      )}
                    </div>
                  </div>
                  <div className="p-3 text-[13px] font-mono" style={{ color: 'var(--text-primary)' }}>
                    {/* LaTeX output */}
                    {cell.latex && (
                      <div
                        className="latex-output mb-3 p-3 rounded"
                        style={{ background: 'var(--surface)', fontFamily: 'serif', fontSize: '14px' }}
                      >
                        <code>{cell.latex}</code>
                      </div>
                    )}

                    {/* Markdown output */}
                    {cell.markdown && (
                      <div className="markdown-output mb-3 prose prose-sm prose-invert max-w-none">
                        <ReactMarkdown>{cell.markdown}</ReactMarkdown>
                      </div>
                    )}

                    {/* HTML output (DataFrames, tables, etc.) */}
                    {cell.html && (
                      <div
                        className="dataframe-output overflow-x-auto mb-3"
                        dangerouslySetInnerHTML={{ __html: cell.html }}
                        style={{
                          maxWidth: '100%',
                          fontSize: '12px',
                        }}
                      />
                    )}

                    {/* SVG output */}
                    {cell.svg && (
                      <div
                        className="svg-output mb-3"
                        dangerouslySetInnerHTML={{ __html: cell.svg }}
                        style={{ maxWidth: '100%' }}
                      />
                    )}

                    {/* Plain text output */}
                    {cell.stdout && <pre className="m-0 mb-2 whitespace-pre-wrap">{cell.stdout}</pre>}

                    {/* Error output */}
                    {cell.stderr && (
                      <pre className="m-0 whitespace-pre-wrap p-2 rounded" style={{ background: 'rgba(248, 81, 73, 0.1)', color: 'var(--accent-red)' }}>
                        {cell.stderr}
                      </pre>
                    )}

                    {/* PNG images */}
                    {cell.images?.map((img, i) => {
                      const filename = img.split(/[\\/]/).pop();
                      return (
                        <div key={i} style={{ marginTop: '12px' }}>
                          <img
                            src={`/files/charts/${filename}`}
                            alt={`Output ${i + 1}`}
                            className="rounded border border-[var(--border)] max-w-full"
                            style={{ maxHeight: '500px', objectFit: 'contain' }}
                          />
                          <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                            {filename}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}

        {/* Add Cell Buttons */}
        <div className="flex justify-center gap-4 mt-6 mb-8">
          <button
            onClick={handleAddCell}
            className="px-4 py-2 rounded-full text-xs font-medium transition-colors flex items-center gap-2 cursor-pointer"
            style={{ border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text-primary)' }}
            onMouseEnter={e => { e.currentTarget.style.background = 'var(--surface-hover)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'var(--surface)'; }}
          >
            + Code
          </button>
          <button
            onClick={handleAddTextCell}
            className="px-4 py-2 rounded-full text-xs font-medium transition-colors flex items-center gap-2 cursor-pointer"
            style={{ border: '1px solid var(--border)', background: 'var(--surface)', color: 'var(--text-primary)' }}
            onMouseEnter={e => { e.currentTarget.style.background = 'var(--surface-hover)'; }}
            onMouseLeave={e => { e.currentTarget.style.background = 'var(--surface)'; }}
          >
            + Text
          </button>
        </div>
        <div ref={bottomRef} />
      </div>

      {/* DataFrame Styling */}
      <style jsx global>{`
        .dataframe-output table {
          border-collapse: collapse;
          font-size: 12px;
          font-family: monospace;
        }
        .dataframe-output th,
        .dataframe-output td {
          padding: 6px 10px;
          border: 1px solid var(--border);
          text-align: left;
        }
        .dataframe-output th {
          background: var(--surface);
          font-weight: 600;
        }
        .dataframe-output tr:nth-child(even) {
          background: var(--surface);
        }
        .dataframe-output tr:hover {
          background: var(--surface-hover);
        }

        /* SVG styling */
        .svg-output svg {
          max-width: 100%;
          height: auto;
        }

        /* Markdown styling */
        .markdown-output h1,
        .markdown-output h2,
        .markdown-output h3 {
          color: var(--text-primary);
          margin-top: 1em;
          margin-bottom: 0.5em;
        }
        .markdown-output p {
          color: var(--text-secondary);
          margin-bottom: 0.5em;
        }
        .markdown-output code {
          background: var(--surface);
          padding: 2px 6px;
          border-radius: 4px;
          font-size: 12px;
        }
        .markdown-output pre code {
          display: block;
          padding: 12px;
        }
        .markdown-output ul,
        .markdown-output ol {
          margin-left: 1.5em;
          color: var(--text-secondary);
        }
        .markdown-output blockquote {
          border-left: 3px solid var(--accent-blue);
          padding-left: 1em;
          color: var(--text-muted);
        }
      `}</style>
    </div>
  );
}
