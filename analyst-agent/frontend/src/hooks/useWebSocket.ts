"use client";
import { useEffect, useRef } from 'react';
import { useSessionStore } from '../stores/sessionStore';

const RECONNECT_DELAYS = [1000, 2000, 4000, 8000, 15000, 30000];
const MAX_RECONNECT = 10;

export function useWebSocket() {
  const ws = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const reconnectTimer = useRef<NodeJS.Timeout | null>(null);
  const addLog = useSessionStore((s: any) => s.addLog);
  const addCell = useSessionStore((s: any) => s.addCell);
  const updateCell = useSessionStore((s: any) => s.updateCell);
  const removeCell = useSessionStore((s: any) => s.removeCell);
  const setProgress = useSessionStore((s: any) => s.setProgress);
  const setRunning = useSessionStore((s: any) => s.setRunning);
  const setConnected = useSessionStore((s: any) => s.setConnected);
  const setCompletedSessionId = useSessionStore((s: any) => s.setCompletedSessionId);

  function connect() {
    if (ws.current?.readyState === WebSocket.OPEN) return;

    // Connect directly to FastAPI backend — Next.js rewrites() cannot proxy WebSocket traffic
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws.current = new WebSocket(`${protocol}//localhost:8000/ws`);

    ws.current.onopen = () => {
      reconnectAttempt.current = 0;
      setConnected(true);
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      switch (data.type) {
        case 'agent_thought':
        case 'quality_event':
          addLog(data);
          break;
        case 'cell_update':
          addCell(data.content);
          break;
        case 'cell_edited':
          updateCell(data.content.cell_id, { code: data.content.code });
          break;
        case 'cell_delete':
          removeCell(data.content.cell_id);
          break;
        case 'progress':
          setProgress(data.content);
          break;
        case 'done': {
          setRunning(false);
          const sid = data.content;
          if (sid && !sid.startsWith('error:')) {
            setCompletedSessionId(sid);
          }
          break;
        }
      }
    };

    ws.current.onclose = () => {
      setConnected(false);
      if (reconnectAttempt.current >= MAX_RECONNECT) return; // stop flooding
      const delay = RECONNECT_DELAYS[
        Math.min(reconnectAttempt.current, RECONNECT_DELAYS.length - 1)
      ];
      reconnectAttempt.current += 1;
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      reconnectTimer.current = setTimeout(connect, delay);
    };

    ws.current.onerror = () => {
      ws.current?.close();
    };
  }

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      ws.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return ws;
}
