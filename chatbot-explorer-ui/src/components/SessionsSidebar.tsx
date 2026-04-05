import { useState } from "react";
import { Plus, MessageSquare, Trash2, MoreHorizontal, PanelLeftClose, PanelLeft } from "lucide-react";

export interface Session {
  id: string;
  title: string;
  date: string;
}

interface SessionsSidebarProps {
  sessions: Session[];
  activeId: string;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

const SessionsSidebar = ({ sessions, activeId, onSelect, onNew, onDelete }: SessionsSidebarProps) => {
  const [collapsed, setCollapsed] = useState(false);
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  if (collapsed) {
    return (
      <div className="flex flex-col items-center gap-2 border-r border-border bg-[hsl(var(--sidebar-background))] py-3 px-2 w-12">
        <button
          onClick={() => setCollapsed(false)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          aria-label="Expand sidebar"
        >
          <PanelLeft className="h-4 w-4" />
        </button>
        <button
          onClick={onNew}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          aria-label="New chat"
        >
          <Plus className="h-4 w-4" />
        </button>
      </div>
    );
  }

  return (
    <div className="flex w-64 flex-col border-r border-border bg-[hsl(var(--sidebar-background))]">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-border">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Sessions</span>
        <div className="flex gap-1">
          <button
            onClick={onNew}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
            aria-label="New chat"
          >
            <Plus className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => setCollapsed(true)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
            aria-label="Collapse sidebar"
          >
            <PanelLeftClose className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Session list */}
      <div className="flex-1 overflow-y-auto scrollbar-thin py-2 px-2 space-y-0.5">
        {sessions.map((s) => (
          <button
            key={s.id}
            onClick={() => onSelect(s.id)}
            onMouseEnter={() => setHoveredId(s.id)}
            onMouseLeave={() => setHoveredId(null)}
            className={`group flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-left transition-colors ${
              s.id === activeId
                ? "bg-accent text-foreground"
                : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
            }`}
          >
            <MessageSquare className="h-3.5 w-3.5 shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="truncate text-[13px]">{s.title}</p>
              <p className="text-[11px] text-muted-foreground/70">{s.date}</p>
            </div>
            {hoveredId === s.id && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(s.id);
                }}
                className="flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-colors"
                aria-label="Delete session"
              >
                <Trash2 className="h-3 w-3" />
              </button>
            )}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SessionsSidebar;
