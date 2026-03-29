"use client";
import { useEffect, useRef } from 'react';
import { useSessionStore } from '../stores/sessionStore';

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 15000, 30000];
const MAX_RECONNECT = 10;

// Debug flag - set to true to enable console logging
const DEBUG_WS = true;

function wsLog(...args: unknown[]) {
  if (DEBUG_WS) {
    console.log('[WS]', ...args);
  }
}

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  const eventCount = useRef(0);

  const addLog = useSessionStore((s: any) => s.addLog);
  const addCell = useSessionStore((s: any) => s.addCell);
  const updateCell = useSessionStore((s: any) => s.updateCell);
  const removeCell = useSessionStore((s: any) => s.removeCell);
  const setProgress = useSessionStore((s: any) => s.setProgress);
  const setRunning = useSessionStore((s: any) => s.setRunning);
  const setConnected = useSessionStore((s: any) => s.setConnected);
  const setCompletedSessionId = useSessionStore((s: any) => s.setCompletedSessionId);

  function connect() {
    if (ws.current?.readyState === WebSocket.OPEN) {
      wsLog('Already connected, skipping...');
      return;
    }

    // Connect directly to FastAPI backend — Next.js rewrites() cannot proxy WebSocket traffic
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//localhost:8000/ws`;
    wsLog('Connecting to:', wsUrl);

    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      wsLog('Connected successfully!');
      reconnectAttempt.current = 0;
      setConnected(true);
    };

    ws.current.onmessage = (event) => {
      eventCount.current += 1;
      const data = JSON.parse(event.data);
      wsLog(`Event #${eventCount.current} [${data.type}]:`, data.content);

      switch (data.type) {
        case 'agent_thought':
        case 'quality_event':
          wsLog('  -> Adding to logs');
          addLog(data);
          break;

        case 'cell_update':
          wsLog('  -> Adding/updating cell:', data.content?.cell_id);
          addCell(data.content);
          break;

        case 'cell_edited':
          wsLog('  -> Editing cell:', data.content?.cell_id);
          updateCell(data.content.cell_id, { code: data.content.code });
          break;

        case 'cell_delete':
          wsLog('  -> Deleting cell:', data.content?.cell_id);
          removeCell(data.content.cell_id);
          break;

        case 'progress':
          wsLog('  -> Setting progress:', data.content);
          setProgress(data.content);
          break;

        case 'done': {
          const sid = data.content;
          wsLog('  -> Analysis done, session:', sid);
          setRunning(false);
          if (sid && !sid.startsWith('error:')) {
            setCompletedSessionId(sid);
          }
          break;
        }

        case 'results_saved':
          wsLog('  -> Results saved:', data.content);
          break;

        default:
          wsLog('  -> Unknown event type:', data.type);
      }
    };

    ws.current.onclose = (event) => {
      wsLog('Connection closed:', event.code, event.reason);
      setConnected(false);

      if (reconnectAttempt.current >= MAX_RECONNECT) {
        wsLog('Max reconnect attempts reached, giving up');
        return;
      }

      const delay = RECONNECT_DELAYS[
        Math.min(reconnectAttempt.current, RECONNECT_DELAYS.length - 1)
      ];
      wsLog(`Reconnecting in ${delay}ms (attempt ${reconnectAttempt.current + 1})`);
      reconnectAttempt.current += 1;

      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      reconnectTimer.current = setTimeout(connect, delay);
    };

    ws.current.onerror = (error) => {
      wsLog('Error:', error);
      ws.current?.close();
    };
  }

  useEffect(() => {
    wsLog('useWebSocket hook mounted, initiating connection...');
    connect();

    return () => {
      wsLog('useWebSocket hook unmounting, cleaning up...');
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return ws;
}
