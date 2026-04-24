import { useState, useRef, useEffect, useCallback } from "react";
import { AlertCircle, Bot, Loader2, PlayCircle, Code2 } from "lucide-react";
import { toast } from "sonner";
import ChatMessage from "@/components/ChatMessage";
import PromptIsland from "@/components/PromptIsland";
import LogsSidebar from "@/components/LogsSidebar";
import CodeEditorPanel from "@/components/CodeEditorPanel";
import { Button } from "@/components/ui/button";
import {
  ApiError,
  getHealth,
  postChatStream,
  postPipelineStream,
  postReport,
  resolveArtifactUrl,
  type HealthResponse,
  type SupervisorStreamEvent,
  type LogEvent,
} from "@/lib/api";
import AgentChainPanel from "@/components/AgentChainPanel";
import { formatChatReply } from "@/lib/formatChatReply";
import { prepareMarkdownForChat, runIdFromRunDir } from "@/lib/markdownDisplay";

interface Message {
  id: number;
  content: string;
  role: "user" | "bot";
  chipLabel?: string;
  chain?: SupervisorStreamEvent[];
}

const WELCOME: Message = {
  id: 0,
  role: "bot",
  content:
    "Hello! I'm connected to your **local analysis server** when it is running. " +
    "Start `uvicorn server:app` from the project root (with `DATASET_PATH` set), then run this UI with `npm run dev`. " +
    "Use **+** for specialist shortcuts, type a goal, or ask follow-ups. First message in a session sets the **analysis goal** for the supervisor. " +
    "Use **Run full batch** in the header to run the **entire dynamic supervisor loop** in one go (until DONE or step limit), like `POST /pipeline` — this can take a long time.",
};

/** Maps action chips to supervisor-oriented instructions (dataset is already on the server). */
const CHIP_TO_INSTRUCTION: Record<string, string> = {
  Cleaning:
    "Focus on data cleaning and preparation only for this turn. The CSV is already loaded in the server session (df_raw / df_clean).",
  Visualization:
    "Focus on data visualization only: produce charts using the session data (df_clean or df_raw as appropriate).",
  EDA: "Focus on exploratory data analysis only: summaries, distributions, correlations, and data quality notes.",
  Statistics: "Focus on statistical analysis and tests only.",
  "Feature Engineering": "Focus on feature engineering from df_clean only.",
  "Class Imbalance": "Focus on class imbalance assessment and recommendations only.",
};

const INITIAL_SESSION_ID = "1";

const Index = () => {
  const [logs, setLogs] = useState<LogEvent[]>([]);
  const [activeSessionId] = useState(INITIAL_SESSION_ID);
  const [messagesMap, setMessagesMap] = useState<Record<string, Message[]>>({
    [INITIAL_SESSION_ID]: [WELCOME],
  });
  /** Locked supervisor goal for this session (first substantive turn). */
  const [sessionGoals, setSessionGoals] = useState<Record<string, string>>({});
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoadError, setHealthLoadError] = useState<string | null>(null);
  const [chatLoading, setChatLoading] = useState(false);
  const [pipelineLoading, setPipelineLoading] = useState(false);
  /** Live SSE events for the in-flight supervisor run (chat or full batch). */
  const [liveChain, setLiveChain] = useState<SupervisorStreamEvent[]>([]);
  const [supervisorStreamOpen, setSupervisorStreamOpen] = useState(false);
  const [isCodeEditorOpen, setIsCodeEditorOpen] = useState(false);
  const [editorCode, setEditorCode] = useState<string | undefined>(undefined);
  const chainAccRef = useRef<SupervisorStreamEvent[]>([]);

  const scrollRef = useRef<HTMLDivElement>(null);
  const nextId = useRef(1);

  const messages = messagesMap[activeSessionId] || [WELCOME];
  const backendOk = health?.status === "ok";
  const busy = chatLoading || pipelineLoading;

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((h) => {
        if (!cancelled) {
          setHealth(h);
          setHealthLoadError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setHealth(null);
          setHealthLoadError(e instanceof Error ? e.message : String(e));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, liveChain, supervisorStreamOpen]);

  const appendMessages = useCallback((sessionId: string, newMsgs: Message[]) => {
    setMessagesMap((prev) => ({
      ...prev,
      [sessionId]: [...(prev[sessionId] || []), ...newMsgs],
    }));
  }, []);

  const handleLoadCode = useCallback((code: string) => {
    setEditorCode(code);
    setIsCodeEditorOpen(true);
  }, []);

  const handleLog = useCallback((log: LogEvent) => {
    setLogs((prev) => {
      const next = [...prev, log];
      if (next.length > 1000) {
        return next.slice(next.length - 1000);
      }
      return next;
    });
  }, []);

  const clearLogs = useCallback(() => {
    setLogs([]);
  }, []);

  const handleSend = useCallback(
    async (content: string, chip?: string) => {
      const sid = activeSessionId;
      const displayText = content.trim() || (chip ? `Run ${chip}` : "");
      if (!displayText) return;

      if (!backendOk) {
        toast.error("Backend not ready. Start the API with DATASET_PATH set (see README).");
        return;
      }

      const userMsg: Message = {
        id: nextId.current++,
        content: displayText,
        role: "user",
        chipLabel: chip,
      };
      appendMessages(sid, [userMsg]);

      if (chip === "Reporter") {
        setChatLoading(true);
        try {
          const rep = await postReport();
          const md = rep.report_markdown?.trim();
          const rid =
            health?.status === "ok" ? health.run_id : runIdFromRunDir(rep.run_dir) ?? "";
          const content =
            md && md.length > 0
              ? prepareMarkdownForChat(
                (rep.truncated ? "_Report truncated for display._\n\n" : "") + md,
                rid,
              )
              : `**Report generated.** Markdown saved under the run folder.\n\n\`${rep.run_dir}\``;
          appendMessages(sid, [
            {
              id: nextId.current++,
              role: "bot",
              content,
            },
          ]);
        } catch (e) {
          const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
          toast.error(msg);
          appendMessages(sid, [
            {
              id: nextId.current++,
              role: "bot",
              content: `**Report failed:** ${msg}`,
            },
          ]);
        } finally {
          setChatLoading(false);
        }
        return;
      }

      let apiMessage = displayText;
      if (chip && CHIP_TO_INSTRUCTION[chip]) {
        apiMessage = `${CHIP_TO_INSTRUCTION[chip]}\n\nUser message: ${displayText}`;
      }

      const lockedGoal = sessionGoals[sid] ?? apiMessage;
      setSessionGoals((prev) => ({
        ...prev,
        [sid]: prev[sid] ?? apiMessage,
      }));

      setChatLoading(true);
      setSupervisorStreamOpen(true);
      chainAccRef.current = [];
      setLiveChain([]);
      try {
        const data = await postChatStream(
          {
            message: apiMessage,
            user_prompt: lockedGoal,
          },
          (evt) => {
            if (evt.type === "heartbeat") return;
            chainAccRef.current.push(evt);
            setLiveChain([...chainAccRef.current]);
          },
          handleLog,
        );
        appendMessages(sid, [
          {
            id: nextId.current++,
            role: "bot",
            content: formatChatReply(data, resolveArtifactUrl),
            chain: [...chainAccRef.current],
          },
        ]);
      } catch (e) {
        const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
        toast.error(msg);
        appendMessages(sid, [
          {
            id: nextId.current++,
            role: "bot",
            content: `**Request failed:** ${msg}`,
          },
        ]);
      } finally {
        setChatLoading(false);
        setSupervisorStreamOpen(false);
        setLiveChain([]);
      }
    },
    [activeSessionId, appendMessages, backendOk, health, sessionGoals],
  );

  const handleRunFullBatch = useCallback(async () => {
    if (!backendOk) {
      toast.error("Backend not ready.");
      return;
    }
    const sid = activeSessionId;
    const sessionGoal = sessionGoals[sid]?.trim();
    const ok = window.confirm(
      "Run the full dynamic supervisor batch now? The manager will keep routing specialists until DONE, a guardrail triggers, or DYNAMIC_MAX_STEPS is reached. " +
      "This often takes many minutes. HTTP interactive state resets afterward (your next chat message starts a fresh supervisor loop; same Python session and data).\n\n" +
      (sessionGoal
        ? `Goal for this run: ${sessionGoal.slice(0, 200)}${sessionGoal.length > 200 ? "…" : ""}`
        : "No session goal in this chat yet — the server will use USER_ANALYSIS_PROMPT from .env if set, or the request will fail."),
    );
    if (!ok) return;

    appendMessages(sid, [
      {
        id: nextId.current++,
        role: "user",
        content: "[ Run full dynamic batch ]",
      },
    ]);
    setPipelineLoading(true);
    setSupervisorStreamOpen(true);
    chainAccRef.current = [];
    setLiveChain([]);
    try {
      const data = await postPipelineStream(
        {
          user_prompt: sessionGoal || undefined,
        },
        (evt) => {
          if (evt.type === "heartbeat") return;
          chainAccRef.current.push(evt);
          setLiveChain([...chainAccRef.current]);
        },
        handleLog,
      );
      appendMessages(sid, [
        {
          id: nextId.current++,
          role: "bot",
          content:
            `**Full batch finished.**\n\n` +
            `- Specialist steps executed: **${data.specialist_steps}**\n` +
            `- run_id: \`${data.run_id}\`\n\n` +
            `_HTTP chat state was reset on the server. Your next normal message starts a new supervisor loop on the same kernel._`,
          chain: [...chainAccRef.current],
        },
      ]);
      toast.success("Full batch complete");
      getHealth()
        .then((h) => {
          setHealth(h);
          setHealthLoadError(null);
        })
        .catch(() => { });
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);
      toast.error(msg);
      appendMessages(sid, [
        {
          id: nextId.current++,
          role: "bot",
          content: `**Full batch failed:** ${msg}`,
        },
      ]);
    } finally {
      setPipelineLoading(false);
      setSupervisorStreamOpen(false);
      setLiveChain([]);
    }
  }, [activeSessionId, appendMessages, backendOk, sessionGoals]);


  return (
    <div className="flex h-screen overflow-hidden">
      <LogsSidebar logs={logs} onClear={clearLogs} />

      <div className="flex flex-1 flex-col min-w-0">
        <header className="flex items-center gap-3 border-b border-border bg-card/50 px-5 py-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-foreground">
            <Bot className="h-4 w-4" />
          </div>
          <div className="min-w-0 flex-1">
            <h1 className="text-sm font-semibold text-foreground">DataBot</h1>
            <p className="text-[11px] text-muted-foreground truncate">
              {health?.status === "ok"
                ? `Run ${health.run_id} · server session active`
                : health?.status === "no_workflow"
                  ? "Server up — workflow not initialized"
                  : healthLoadError
                    ? `Cannot reach API (${healthLoadError})`
                    : "Checking API…"}
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <Button
              type="button"
              variant="secondary"
              size="sm"
              className="h-8 gap-1.5 text-[11px]"
              disabled={busy || !backendOk}
              onClick={() => void handleRunFullBatch()}
              title="POST /pipeline — full supervisor batch until DONE or step limit"
            >
              {pipelineLoading ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
              ) : (
                <PlayCircle className="h-3.5 w-3.5" aria-hidden />
              )}
              Full batch
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              className={`h-8 gap-1.5 text-[11px] ${isCodeEditorOpen ? "bg-accent text-accent-foreground" : ""}`}
              onClick={() => setIsCodeEditorOpen(!isCodeEditorOpen)}
              title="Toggle interactive code editor"
            >
              <Code2 className="h-3.5 w-3.5" aria-hidden />
              Code
            </Button>
            {chatLoading && !pipelineLoading && (
              <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" aria-label="Chat loading" />
            )}
          </div>
        </header>

        {(healthLoadError || health?.status === "no_workflow") && (
          <div
            className="flex items-start gap-2 border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-[12px] text-amber-950 dark:text-amber-100"
            role="status"
          >
            <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
            <div>
              {healthLoadError ? (
                <p>
                  <strong>API unreachable.</strong> Start FastAPI:{" "}
                  <code className="rounded bg-background/60 px-1">uvicorn server:app --host 127.0.0.1 --port 8765</code> from
                  the repo root. This UI proxies <code className="rounded bg-background/60 px-1">/api</code> to that port in
                  dev.
                </p>
              ) : (
                <p>
                  <strong>No workflow.</strong> {health.hint} Set <code className="rounded bg-background/60 px-1">DATASET_PATH</code>{" "}
                  in <code className="rounded bg-background/60 px-1">.env</code> and restart the server.
                </p>
              )}
            </div>
          </div>
        )}

        <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin px-4 py-6">
          <div className="mx-auto max-w-[min(96vw,52rem)] space-y-5">
            {messages.map((m) => (
              <ChatMessage
                key={m.id}
                content={m.content}
                role={m.role}
                chipLabel={m.chipLabel}
                wide={m.role === "bot"}
                chain={m.chain}
                onLoadCode={handleLoadCode}
              />
            ))}
            {/* Live supervisor steps belong after the latest user turn (same as Cursor-style streaming). */}
            {supervisorStreamOpen && (
              <AgentChainPanel events={liveChain} loading={chatLoading || pipelineLoading} defaultOpen onLoadCode={handleLoadCode} />
            )}
          </div>
        </div>

        <div className="border-t border-border bg-background px-4 pb-5 pt-3">
          <PromptIsland onSend={handleSend} disabled={busy || !backendOk} />
        </div>
      </div>

      {isCodeEditorOpen && (
        <div className="w-1/3 min-w-[350px] max-w-[600px] shrink-0 border-l border-border h-full bg-background z-10 transition-all">
          <CodeEditorPanel
            code={editorCode ?? `# Interactive Python Session
# The kernel retains state between runs.
# Existing variables: df_raw, df_clean, df_features

print("Current shape of df_raw:", df_raw.shape)
`}
            onChange={setEditorCode}
            onClose={() => setIsCodeEditorOpen(false)}
          />
        </div>
      )}
    </div>
  );
};

export default Index;
