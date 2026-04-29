import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertOctagon,
  AlertTriangle,
  Bot,
  Circle,
  Eraser,
  Info,
  PanelLeftClose,
  PanelLeft,
  PauseCircle,
  PlayCircle,
  Terminal,
  Wrench,
  ShieldAlert,
  Cog,
  Workflow,
} from "lucide-react";
import type { LogCategory, LogEvent, LogLevel } from "@/lib/api";

interface LogsSidebarProps {
  logs: LogEvent[];
  onClear: () => void;
}

const LEVELS: Array<{ id: LogLevel; label: string }> = [
  { id: "info", label: "Info" },
  { id: "warn", label: "Warn" },
  { id: "error", label: "Error" },
];

const LEVEL_STYLE: Record<LogLevel, string> = {
  info: "text-foreground/80",
  warn: "text-amber-600 dark:text-amber-400",
  error: "text-red-600 dark:text-red-400",
};

const LEVEL_DOT: Record<LogLevel, string> = {
  info: "bg-blue-500/70",
  warn: "bg-amber-500",
  error: "bg-red-500",
};

function CategoryIcon({ category }: { category: LogCategory }) {
  const cls = "h-3 w-3 shrink-0";
  switch (category) {
    case "workflow":
      return <Workflow className={cls} />;
    case "manager":
      return <Bot className={cls} />;
    case "specialist":
      return <Cog className={cls} />;
    case "task":
      return <Circle className={cls} />;
    case "tool":
      return <Wrench className={cls} />;
    case "guardrail":
      return <ShieldAlert className={cls} />;
    default:
      return <Circle className={cls} />;
  }
}

function LevelIcon({ level }: { level: LogLevel }) {
  const cls = "h-3 w-3 shrink-0";
  if (level === "error") return <AlertOctagon className={cls} />;
  if (level === "warn") return <AlertTriangle className={cls} />;
  return <Info className={cls} />;
}

function formatTimestamp(iso: string): string {
  try {
    const d = new Date(iso);
    const hh = String(d.getHours()).padStart(2, "0");
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    const ms = String(d.getMilliseconds()).padStart(3, "0");
    return `${hh}:${mm}:${ss}.${ms}`;
  } catch {
    return iso;
  }
}

const LogsSidebar = ({ logs, onClear }: LogsSidebarProps) => {
  const [collapsed, setCollapsed] = useState(false);
  const [activeLevels, setActiveLevels] = useState<Set<LogLevel>>(
    new Set<LogLevel>(["info", "warn", "error"]),
  );
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(
    () => logs.filter((l) => activeLevels.has(l.level)),
    [logs, activeLevels],
  );

  useEffect(() => {
    if (!autoScroll) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [filtered, autoScroll]);

  const toggleLevel = (level: LogLevel) => {
    setActiveLevels((prev) => {
      const next = new Set(prev);
      if (next.has(level)) next.delete(level);
      else next.add(level);
      if (next.size === 0) return prev;
      return next;
    });
  };

  if (collapsed) {
    return (
      <div className="flex flex-col items-center gap-2 border-r border-border bg-[hsl(var(--sidebar-background))] py-3 px-2 w-12">
        <button
          onClick={() => setCollapsed(false)}
          className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          aria-label="Expand logs"
          title="Expand logs"
        >
          <PanelLeft className="h-4 w-4" />
        </button>
        <div className="flex h-8 w-8 items-center justify-center rounded-lg text-muted-foreground" title="Workflow logs">
          <Terminal className="h-4 w-4" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex w-72 flex-col border-r border-border bg-[hsl(var(--sidebar-background))]">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-3 border-b border-border">
        <div className="flex items-center gap-1.5">
          <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Workflow Logs</span>
        </div>
        <div className="flex gap-1">
          <button
            onClick={() => setAutoScroll((v) => !v)}
            className={`flex h-7 w-7 items-center justify-center rounded-lg transition-colors ${
              autoScroll
                ? "text-foreground hover:bg-accent"
                : "text-muted-foreground hover:bg-accent hover:text-foreground"
            }`}
            aria-label={autoScroll ? "Pause auto-scroll" : "Resume auto-scroll"}
            title={autoScroll ? "Pause auto-scroll" : "Resume auto-scroll"}
          >
            {autoScroll ? <PauseCircle className="h-3.5 w-3.5" /> : <PlayCircle className="h-3.5 w-3.5" />}
          </button>
          <button
            onClick={onClear}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
            aria-label="Clear logs"
            title="Clear logs"
          >
            <Eraser className="h-3.5 w-3.5" />
          </button>
          <button
            onClick={() => setCollapsed(true)}
            className="flex h-7 w-7 items-center justify-center rounded-lg text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
            aria-label="Collapse sidebar"
            title="Collapse sidebar"
          >
            <PanelLeftClose className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Level filters */}
      <div className="flex items-center gap-1 border-b border-border px-3 py-2">
        {LEVELS.map((lv) => {
          const active = activeLevels.has(lv.id);
          return (
            <button
              key={lv.id}
              onClick={() => toggleLevel(lv.id)}
              className={`flex items-center gap-1 rounded-md px-2 py-0.5 text-[11px] transition-colors ${
                active
                  ? "bg-accent text-foreground"
                  : "text-muted-foreground hover:bg-accent/40 hover:text-foreground"
              }`}
              aria-pressed={active}
            >
              <span className={`h-1.5 w-1.5 rounded-full ${LEVEL_DOT[lv.id]}`} aria-hidden />
              {lv.label}
            </button>
          );
        })}
        <div className="ml-auto text-[10px] text-muted-foreground/70" aria-live="polite">
          {filtered.length}/{logs.length}
        </div>
      </div>

      {/* Log list */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin font-mono text-[11px] leading-snug">
        {filtered.length === 0 ? (
          <div className="px-3 py-4 text-muted-foreground italic">
            No events yet — logs will appear here in real-time when a workflow runs.
          </div>
        ) : (
          <ol className="divide-y divide-border/40">
            {filtered.map((l, i) => (
              <li
                key={i}
                className="flex gap-2 px-3 py-1.5 hover:bg-accent/30 transition-colors"
                title={`${l.category}.${l.event}`}
              >
                <span className="text-muted-foreground/70 shrink-0">{formatTimestamp(l.ts)}</span>
                <span className={`flex items-center gap-1 ${LEVEL_STYLE[l.level]}`}>
                  <LevelIcon level={l.level} />
                </span>
                <span className="flex items-center gap-1 text-muted-foreground/80 shrink-0">
                  <CategoryIcon category={l.category} />
                  <span className="text-[10px] uppercase tracking-wide">{l.category}</span>
                </span>
                <span className={`min-w-0 flex-1 break-words ${LEVEL_STYLE[l.level]}`}>
                  {l.message || l.event}
                  {l.tokens != null && (
                    <span className="ml-2 inline-flex items-center rounded-full bg-accent/50 px-1.5 py-0.5 text-[9px] font-medium text-muted-foreground">
                      {l.tokens} tokens
                    </span>
                  )}
                </span>
              </li>
            ))}
          </ol>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-border px-3 py-1.5 text-[10px] text-muted-foreground/80">
        Infrastructure-only — agent decisions/outputs appear in the main chain-of-thoughts panel.
      </div>
    </div>
  );
};

export default LogsSidebar;
