import { useRef, useMemo, useState } from "react";
import type { Components } from "react-markdown";
import { Bot, User, Download, FileDown, Loader2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import AgentChainPanel from "@/components/AgentChainPanel";
import { resolveArtifactUrl, type SupervisorStreamEvent } from "@/lib/api";
import { wrapTabularPlaintextInFences } from "@/lib/markdownTabular";
// @ts-ignore - html2pdf.js doesn't have official types
import html2pdf from "html2pdf.js";

interface ChatMessageProps {
  content: string;
  role: "user" | "bot";
  chipLabel?: string;
  /** Wider bubble for long markdown reports */
  wide?: boolean;
  /** Collapsible supervisor steps from SSE (chat/stream pipeline) */
  chain?: SupervisorStreamEvent[];
  onLoadCode?: (code: string) => void;
}

function normalizeMarkdownImgSrc(src: string | undefined): string {
  if (!src) return "";
  const s = src.trim();
  if (/^https?:\/\//i.test(s)) return s;
  if (s.startsWith("/artifacts/")) return resolveArtifactUrl(s);
  return s;
}

/** GFM tables: prose + prose-invert fight table layout; use not-prose + explicit cell borders. */
const mdTableComponents: Partial<Components> = {
  table: ({ children, ...props }) => (
    <div className="not-prose my-4 w-full max-w-full overflow-x-auto rounded-lg border border-border bg-background/40">
      <table
        className="min-w-max max-w-none border-collapse border-spacing-0 text-left text-[12px] leading-snug text-foreground sm:text-[13px]"
        {...props}
      >
        {children}
      </table>
    </div>
  ),
  thead: ({ children, ...props }) => (
    <thead className="border-b border-border bg-muted/40 [&_tr]:border-border" {...props}>
      {children}
    </thead>
  ),
  tbody: ({ children, ...props }) => (
    <tbody className="divide-y divide-border/80" {...props}>
      {children}
    </tbody>
  ),
  tr: ({ children, ...props }) => <tr {...props}>{children}</tr>,
  th: ({ children, ...props }) => (
    <th
      className="border border-border/70 px-2.5 py-2 align-top font-semibold text-foreground sm:px-3"
      {...props}
    >
      {children}
    </th>
  ),
  td: ({ children, ...props }) => (
    <td className="border border-border/70 px-2.5 py-2 align-top text-foreground tabular-nums sm:px-3" {...props}>
      {children}
    </td>
  ),
};

const ChatMessage = ({ content, role, chipLabel, wide, chain, onLoadCode }: ChatMessageProps) => {
  const isUser = role === "user";
  const botMarkdown = useMemo(() => wrapTabularPlaintextInFences(content), [content]);
  const contentRef = useRef<HTMLDivElement>(null);
  const [isExporting, setIsExporting] = useState(false);

  const isReport = useMemo(() => {
    return role === "bot" && (content.includes("# Executive Summary") || content.includes("## Charts") || content.includes("Report"));
  }, [role, content]);

  const handleDownloadPdf = async () => {
    const element = contentRef.current;
    if (!element) return;

    setIsExporting(true);
    try {
      // Create a clone of the element to modify it for PDF without affecting the UI
      const opt = {
        margin: [10, 10, 10, 10],
        filename: `Analysis_Report_${new Date().toISOString().split('T')[0]}.pdf`,
        image: { type: 'jpeg', quality: 0.95 },
        html2canvas: { 
          scale: 1, // Reduced from 2 to prevent canvas memory allocation errors
          useCORS: true, 
          logging: false,
          backgroundColor: '#ffffff'
        },
        jsPDF: { unit: 'mm', format: 'a4', orientation: 'portrait' },
        // Removed 'avoid-all' because it causes infinite loops (and OOM) if an element is taller than one page
        pagebreak: { mode: ['css', 'legacy'] }
      };

      // Temporarily add a class to force light-mode styling for the PDF
      element.classList.add('pdf-export-mode');
      
      await html2pdf().from(element).set(opt).save();
      
      element.classList.remove('pdf-export-mode');
    } catch (err) {
      console.error("PDF generation failed:", err);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-lg ${
          isUser ? "bg-accent text-foreground" : "bg-secondary text-muted-foreground"
        }`}
      >
        {isUser ? <User className="h-3.5 w-3.5" /> : <Bot className="h-3.5 w-3.5" />}
      </div>
      <div
        className={`${wide ? "max-w-[min(96vw,52rem)]" : "max-w-[78%]"} min-w-0 space-y-1`}
      >
        {chipLabel && (
          <span className="inline-block rounded-md bg-accent px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
            {chipLabel}
          </span>
        )}
        <div
          className={`rounded-2xl px-4 py-2.5 text-[13px] leading-relaxed ${
            isUser
              ? "bg-accent text-foreground rounded-tr-md"
              : "bg-card text-foreground border border-border/50 rounded-tl-md"
          }`}
        >
          {isUser ? (
            content
          ) : (
            <div className="space-y-3">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  {chain && chain.length > 0 && (
                    <AgentChainPanel events={chain} loading={false} defaultOpen={false} onLoadCode={onLoadCode} />
                  )}
                </div>
                {isReport && (
                  <button
                    onClick={handleDownloadPdf}
                    disabled={isExporting}
                    className="flex shrink-0 items-center gap-1.5 rounded-lg border border-border bg-background/50 px-2.5 py-1.5 text-[11px] font-medium text-muted-foreground transition-all hover:bg-accent hover:text-foreground disabled:opacity-50"
                    title="Download report as PDF"
                  >
                    {isExporting ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <FileDown className="h-3 w-3" />
                    )}
                    {isExporting ? "Generating..." : "PDF"}
                  </button>
                )}
              </div>
            <div
              ref={contentRef}
              className={[
                "prose prose-sm max-w-none",
                "text-foreground",
                "prose-headings:text-foreground prose-headings:font-semibold",
                "prose-p:text-foreground prose-p:leading-relaxed",
                "prose-li:text-foreground prose-li:marker:text-muted-foreground",
                "prose-strong:text-foreground prose-strong:font-semibold",
                "prose-code:text-foreground prose-code:bg-muted/90 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:font-normal prose-code:before:content-none prose-code:after:content-none",
                "prose-pre:bg-muted prose-pre:text-foreground",
                "prose-blockquote:text-foreground prose-blockquote:border-border",
                "prose-a:text-primary prose-a:underline underline-offset-2",
                "prose-th:text-foreground prose-td:text-foreground",
                "prose-table:border-border prose-tr:border-border",
                "prose-img:rounded-md prose-img:max-w-full prose-img:border prose-img:border-border/60",
              ].join(" ")}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  ...mdTableComponents,
                  pre: ({ children, ...props }) => (
                    <pre
                      className="not-prose my-3 max-h-[min(70vh,560px)] max-w-full overflow-auto rounded-lg border border-border/60 bg-muted/70 p-3 font-mono text-[11px] leading-snug text-foreground shadow-inner"
                      {...props}
                    >
                      {children}
                    </pre>
                  ),
                  code: ({ className, children, ...props }) => {
                    const block = typeof className === "string" && /\blanguage-/.test(className);
                    return (
                      <code
                        className={
                          block
                            ? `${className ?? ""} block w-max min-w-full whitespace-pre text-left`
                            : className
                        }
                        {...props}
                      >
                        {children}
                      </code>
                    );
                  },
                  img: ({ src, alt, ...rest }) => {
                    const imgSrc = normalizeMarkdownImgSrc(typeof src === "string" ? src : undefined);
                    const downloadUrl = imgSrc ? `${imgSrc}${imgSrc.includes("?") ? "&" : "?"}download=1` : "#";
                    return (
                      <div className="group relative my-4 inline-block max-w-full overflow-hidden rounded-md border border-border/60">
                        <img
                          {...rest}
                          src={imgSrc}
                          alt={alt ?? ""}
                          className="max-h-[min(70vh,520px)] w-auto"
                          loading="lazy"
                        />
                        <div className="absolute right-2 top-2 translate-y-1 opacity-0 transition-all group-hover:translate-y-0 group-hover:opacity-100">
                          <a
                            href={downloadUrl}
                            download
                            className="flex h-8 w-8 items-center justify-center rounded-full bg-black/50 text-white backdrop-blur-sm hover:bg-black/70"
                            title="Download chart"
                          >
                            <Download className="h-4 w-4" />
                          </a>
                        </div>
                      </div>
                    );
                  },
                }}
              >
                {botMarkdown}
              </ReactMarkdown>
            </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
