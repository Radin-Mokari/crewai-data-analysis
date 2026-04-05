import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { resolveArtifactUrl } from "@/lib/api";

interface ChatMessageProps {
  content: string;
  role: "user" | "bot";
  chipLabel?: string;
  /** Wider bubble for long markdown reports */
  wide?: boolean;
}

function normalizeMarkdownImgSrc(src: string | undefined): string {
  if (!src) return "";
  const s = src.trim();
  if (/^https?:\/\//i.test(s)) return s;
  if (s.startsWith("/artifacts/")) return resolveArtifactUrl(s);
  return s;
}

const ChatMessage = ({ content, role, chipLabel, wide }: ChatMessageProps) => {
  const isUser = role === "user";

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
            <div
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
                "prose-table:text-[13px] prose-th:px-2 prose-td:px-2",
                "dark:prose-invert",
              ].join(" ")}
            >
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  img: ({ src, alt, ...rest }) => (
                    <img
                      {...rest}
                      src={normalizeMarkdownImgSrc(typeof src === "string" ? src : undefined)}
                      alt={alt ?? ""}
                      className="rounded-md max-h-[min(70vh,520px)] w-auto border border-border/60"
                      loading="lazy"
                    />
                  ),
                }}
              >
                {content}
              </ReactMarkdown>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ChatMessage;
