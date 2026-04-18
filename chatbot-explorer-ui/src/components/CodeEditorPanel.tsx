import { useState } from "react";
import Editor from "@monaco-editor/react";
import { Play, Loader2, XCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { postRunCode, resolveArtifactUrl } from "@/lib/api";

interface CodeEditorPanelProps {
  code: string;
  onChange: (code: string) => void;
  onClose?: () => void;
}

const DEFAULT_CODE = `# Interactive Python Session
# The kernel retains state between runs.
# Existing variables: df_raw, df_clean, df_features

print("Current shape of df_raw:", df_raw.shape)
`;

export default function CodeEditorPanel({ code, onChange, onClose }: CodeEditorPanelProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [output, setOutput] = useState<{ stdout: string; error: string | null; charts: string[] } | null>(null);

  const handleRun = async () => {
    setIsRunning(true);
    setOutput(null);
    try {
      const res = await postRunCode(code);
      setOutput({ stdout: res.stdout, error: res.error, charts: res.charts });
    } catch (err) {
      setOutput({ stdout: "", error: err instanceof Error ? err.message : String(err), charts: [] });
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="flex h-full flex-col border-l border-border bg-card">
      <div className="flex items-center justify-between border-b border-border p-3">
        <h2 className="text-sm font-semibold tracking-tight text-foreground">Interactive Editor</h2>
        <div className="flex items-center space-x-2">
          <Button
            size="sm"
            variant="default"
            onClick={handleRun}
            disabled={isRunning}
            className="h-7 gap-1 px-3"
          >
            {isRunning ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3" />}
            Run
          </Button>
          {onClose && (
            <Button size="icon" variant="ghost" onClick={onClose} className="h-7 w-7 text-muted-foreground">
              <XCircle className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      <div className="flex-1 overflow-hidden bg-[#1e1e1e]">
        <Editor
          height="100%"
          defaultLanguage="python"
          theme="vs-dark"
          value={code}
          onChange={(val) => onChange(val || "")}
          options={{
            minimap: { enabled: false },
            fontSize: 13,
            lineHeight: 1.5,
            padding: { top: 16, bottom: 16 },
            wordWrap: "on",
          }}
        />
      </div>

      <div className="flex h-1/3 min-h-[200px] flex-col border-t border-border bg-background">
        <div className="border-b border-border bg-muted/30 px-3 py-1.5">
          <span className="text-xs font-medium uppercase text-muted-foreground">Output Terminal</span>
        </div>
        <div className="flex-1 overflow-y-auto p-3 font-mono text-sm">
          {output ? (
            <div className="space-y-4">
              {output.error && (
                <div className="text-destructive whitespace-pre-wrap">
                  {output.error}
                </div>
              )}
              {output.stdout && (
                <div className="text-foreground whitespace-pre-wrap">
                  {output.stdout}
                </div>
              )}
              {!output.error && !output.stdout && output.charts.length === 0 && (
                <div className="text-muted-foreground italic">Code executed successfully with no output.</div>
              )}
              {output.charts.length > 0 && (
                <div className="space-y-2">
                  <div className="text-xs text-muted-foreground">Generated Charts:</div>
                  {output.charts.map((url, i) => (
                    <img key={i} src={resolveArtifactUrl(url)} alt={`Generated ${i}`} className="max-w-full rounded-md border border-border" />
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="text-muted-foreground italic">Run code to see output...</div>
          )}
        </div>
      </div>
    </div>
  );
}
