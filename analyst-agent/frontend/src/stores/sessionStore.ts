"use client";
import { create } from 'zustand';

export const useSessionStore = create((set: any) => ({
  logs: [],
  cells: [],
  progress: '',
  sessions: [],
  activeSession: null,
  isRunning: false,
  connected: false,
  completedSessionId: null,
  savedPath: null,

  addLog: (entry: any) => set((state: any) => ({ logs: [...state.logs, entry] })),

  addCell: (cell: any) =>
    set((state: any) => {
      const existing = state.cells.findIndex((c: any) => c.cell_id === cell.cell_id);
      if (existing >= 0) {
        const updated = [...state.cells];
        updated[existing] = cell;
        return { cells: updated };
      }
      return { cells: [...state.cells, cell] };
    }),

  updateCell: (cell_id: string, updates: any) =>
    set((state: any) => {
      const idx = state.cells.findIndex((c: any) => c.cell_id === cell_id);
      if (idx < 0) return state;
      const updated = [...state.cells];
      updated[idx] = { ...updated[idx], ...updates };
      return { cells: updated };
    }),

  removeCell: (cell_id: string) =>
    set((state: any) => ({
      cells: state.cells.filter((c: any) => c.cell_id !== cell_id),
    })),

  setProgress: (msg: string) => set({ progress: msg }),
  setSessions: (list: any[]) => set({ sessions: list }),
  setActiveSession: (session: any) => set({ activeSession: session }),
  setRunning: (bool: boolean) => set({ isRunning: bool }),
  setConnected: (bool: boolean) => set({ connected: bool }),
  setCompletedSessionId: (id: string | null) => set({ completedSessionId: id }),
  setSavedPath: (path: string | null) => set({ savedPath: path }),

  clearCurrent: () =>
    set({ logs: [], cells: [], progress: '', completedSessionId: null, savedPath: null }),
}));
