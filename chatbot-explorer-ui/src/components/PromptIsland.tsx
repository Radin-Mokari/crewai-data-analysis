import { useState } from "react";
import { SendHorizonal, Plus, X, Sparkles, BarChart3, Search, PieChart, Wrench, Scale, FileText } from "lucide-react";
import ActionChip from "./ActionChip";

const actions = [
  { label: "Cleaning", icon: Sparkles },
  { label: "Visualization", icon: BarChart3 },
  { label: "EDA", icon: Search },
  { label: "Statistics", icon: PieChart },
  { label: "Feature Engineering", icon: Wrench },
  { label: "Class Imbalance", icon: Scale },
  { label: "Reporter", icon: FileText },
];

interface PromptIslandProps {
  onSend: (message: string, chip?: string) => void;
  disabled?: boolean;
}

const PromptIsland = ({ onSend, disabled = false }: PromptIslandProps) => {
  const [input, setInput] = useState("");
  const [showActions, setShowActions] = useState(false);
  const [selectedChip, setSelectedChip] = useState<string | null>(null);

  const handleSend = () => {
    if (disabled || (!input.trim() && !selectedChip)) return;
    onSend(input.trim() || `Run ${selectedChip}`, selectedChip || undefined);
    setInput("");
    setSelectedChip(null);
  };

  const handleChipClick = (label: string) => {
    setSelectedChip(label);
    setShowActions(false);
  };

  const selectedAction = selectedChip ? actions.find((a) => a.label === selectedChip) : null;

  return (
    <div className="relative w-full max-w-3xl mx-auto">
      {showActions && (
        <div className="absolute bottom-full left-0 right-0 mb-3 flex flex-wrap justify-center gap-2 animate-in fade-in slide-in-from-bottom-2 duration-200">
          {actions.map((a) => (
            <ActionChip key={a.label} label={a.label} icon={a.icon} onClick={() => handleChipClick(a.label)} />
          ))}
        </div>
      )}

      <div className="flex flex-col gap-0 rounded-2xl border border-border bg-card">
        {/* Input row */}
        <div className="flex-1 px-4 pt-3 pb-1">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !disabled && handleSend()}
            placeholder="Ask me anything about your data…"
            disabled={disabled}
            className="w-full bg-transparent text-[13px] outline-none placeholder:text-muted-foreground disabled:opacity-50"
          />
        </div>

        {/* Bottom bar with chip + actions */}
        <div className="flex items-center justify-between px-2 pb-2 pt-1">
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              disabled={disabled}
              onClick={() => setShowActions((v) => !v)}
              className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-accent hover:text-foreground disabled:opacity-40"
              aria-label="Toggle actions"
            >
              {showActions ? <X className="h-4 w-4" /> : <Plus className="h-4 w-4" />}
            </button>

            {selectedChip && selectedAction && (
              <div className="flex items-center gap-1.5 rounded-lg border border-primary/20 bg-primary/5 px-2.5 py-1 text-[12px] font-medium text-primary animate-in fade-in zoom-in-95 duration-150">
                <selectedAction.icon className="h-3 w-3" />
                <span>{selectedChip}</span>
                <button
                  onClick={() => setSelectedChip(null)}
                  className="ml-0.5 rounded-sm text-primary/60 hover:text-primary transition-colors"
                  aria-label="Remove action"
                >
                  <X className="h-3 w-3" />
                </button>
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={handleSend}
            disabled={disabled || (!input.trim() && !selectedChip)}
            className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/80 text-primary-foreground transition-all hover:bg-primary disabled:opacity-30"
            aria-label="Send message"
          >
            <SendHorizonal className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default PromptIsland;
