import type { LucideIcon } from "lucide-react";

interface ActionChipProps {
  label: string;
  icon: LucideIcon;
  onClick: () => void;
}

const ActionChip = ({ label, icon: Icon, onClick }: ActionChipProps) => (
  <button
    onClick={onClick}
    className="group flex items-center gap-2 rounded-xl border border-border bg-card px-3.5 py-2 text-[13px] font-medium text-muted-foreground shadow-sm transition-all duration-150 hover:border-primary/30 hover:text-foreground hover:bg-accent active:scale-[0.98]"
  >
    <Icon className="h-3.5 w-3.5" />
    {label}
  </button>
);

export default ActionChip;
