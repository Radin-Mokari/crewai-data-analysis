import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  ChevronDown,
  Circle,
  Loader2,
  User,
  Code2,
} from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import type { SupervisorStreamEvent } from "@/lib/api";

type Props = {
  events: SupervisorStreamEvent[];
  /** Show spinner / "Working" when the stream has not finished */
  loading?: boolean;
  /** Start expanded while loading; collapsed when done unless user opens */
  defaultOpen?: boolean;
  onLoadCode?: (code: string) => void;
};

function eventLabel(e: SupervisorStreamEvent): string {
  switch (e.type) {
    case "manager_decision":
      return `Manager → ${e.next_agent}`;
    case "specialist_start":
      return `Running ${e.agent} (step ${e.step})`;
    case "specialist_complete":
      return `${e.agent} finished`;
    case "manager_message":
      return "Manager message";
    case "manager_summary":
      return "Generating summary";
    case "guardrail":
      return "Guardrail";
    default:
      return "Event";
  }
}

export default function AgentChainPanel({ events, loading = false, defaultOpen, onLoadCode }: Props) {
  const open = defaultOpen !== undefined ? defaultOpen : loading;
  if (events.length === 0 && !loading) {
    return null;
  }

  return (
    <Collapsible defaultOpen={open} className="rounded-xl border border-border/60 bg-muted/30">
      <CollapsibleTrigger className="group flex w-full items-center gap-2 px-3 py-2 text-left text-[12px] font-medium text-foreground hover:bg-muted/50 rounded-t-xl">
        {loading ? (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" aria-hidden />
        ) : (
          <CheckCircle2 className="h-3.5 w-3.5 shrink-0 text-emerald-600 dark:text-emerald-400" aria-hidden />
        )}
        <span className="flex-1">Supervisor run {loading ? "(live)" : ""}</span>
        <ChevronDown className="h-4 w-4 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ol className="space-y-2 border-t border-border/40 px-3 py-2 text-[11px] text-muted-foreground">
          {loading && events.length === 0 && (
            <li className="flex items-center gap-2 text-foreground/80">
              <Loader2 className="h-3 w-3 animate-spin" />
              Waiting for supervisor…
            </li>
          )}
          {events.map((e, i) => (
            <li key={i} className="rounded-lg bg-background/50 px-2 py-1.5 border border-border/40">
              <div className="flex items-start gap-2">
                {e.type === "manager_decision" ? (
                  <Bot className="h-3.5 w-3.5 mt-0.5 shrink-0 text-primary" />
                ) : e.type === "manager_summary" ? (
                  <Bot className="h-3.5 w-3.5 mt-0.5 shrink-0 text-emerald-600" />
                ) : e.type === "guardrail" ? (
                  <AlertTriangle className="h-3.5 w-3.5 mt-0.5 shrink-0 text-amber-600" />
                ) : e.type === "specialist_complete" ? (
                  <CheckCircle2 className="h-3.5 w-3.5 mt-0.5 shrink-0 text-emerald-600" />
                ) : e.type === "specialist_start" ? (
                  <Circle className="h-3.5 w-3.5 mt-0.5 shrink-0 text-blue-500 fill-blue-500/30" />
                ) : (
                  <User className="h-3.5 w-3.5 mt-0.5 shrink-0 text-muted-foreground" />
                )}
                <div className="min-w-0 flex-1 space-y-1">
                  <div className="flex items-center justify-between">
                    <div className="font-medium text-foreground/90">{eventLabel(e)}</div>
                    {e.type === "specialist_complete" && e.agent_code && onLoadCode && (
                      <Button
                        variant="secondary"
                        size="sm"
                        className="h-5 px-2 text-[9px] gap-1 bg-accent/50 hover:bg-accent"
                        onClick={(ev) => {
                          ev.stopPropagation();
                          onLoadCode(e.agent_code!);
                        }}
                      >
                        <Code2 className="h-3 w-3" />
                        Load Code
                      </Button>
                    )}
                  </div>
                  {e.type === "manager_decision" && (
                    <pre className="whitespace-pre-wrap break-words font-mono text-[10px] leading-snug text-foreground/80">
                      {e.instruction ? `Instruction: ${e.instruction}` : ""}
                      {e.rationale ? `\nRationale: ${e.rationale}` : ""}
                    </pre>
                  )}
                  {(e.type === "manager_message" || e.type === "manager_summary") && (
                    <pre className="whitespace-pre-wrap break-words font-sans text-[11px] text-foreground/85">
                      {e.text}
                    </pre>
                  )}
                  {e.type === "guardrail" && <p className="text-amber-900/90 dark:text-amber-100/90">{e.message}</p>}
                  {e.type === "specialist_complete" && e.excerpt && (
                    <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words font-mono text-[10px] leading-snug text-foreground/75">
                      {e.excerpt.length > 1200 ? `${e.excerpt.slice(0, 1200)}…` : e.excerpt}
                    </pre>
                  )}
                </div>
              </div>
            </li>
          ))}
        </ol>
      </CollapsibleContent>
    </Collapsible>
  );
}
