"use client";

import { useEffect, useState } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import SessionSidebar from '../components/SessionSidebar';
import ChatSection from '../components/ChatSection';
import KernelView from '../components/KernelView';
import ProcessLogs from '../components/ProcessLogs';

export default function Home() {
  useWebSocket();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div suppressHydrationWarning style={{ height: '100vh', width: '100vw', background: 'var(--bg-chat)' }} />
    );
  }

  return (
    <div suppressHydrationWarning style={{ height: '100vh', width: '100vw', display: 'flex', overflow: 'hidden', background: 'var(--bg-chat)', color: 'var(--text-primary)' }}>
      {/* Left: Sidebar */}
      <SessionSidebar />

      {/* Middle: Chat Interface */}
      <ChatSection />

      {/* Right: Kernel and Logs */}
      <div 
        style={{ display: 'flex', flexDirection: 'column', width: '480px', minWidth: '400px', borderLeft: '1px solid var(--border)', zIndex: 10, background: 'var(--bg-right)' }}
      >
        <KernelView />
        <ProcessLogs />
      </div>
    </div>
  );
}
