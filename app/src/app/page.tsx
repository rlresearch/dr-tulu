"use client";
import Link from "next/link";
import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  ArrowRightIcon,
  SendHorizontal,
  ExternalLink,
  Search,
  ArrowDownFromLine,
  ArrowUpFromLine,
  Github,
  BookOpen,
  Database,
  Loader2,
  Download,
  Upload,
  Globe,
  FileText,
  Wrench,
} from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";

import {
  PageHeader,
  PageHeaderHeading,
} from "@/app/components/page-header";
import { cn } from "@/lib/utils";
import { useChat, TraceItem, ToolCallData } from "@/lib/sse";

const TITLE =
  "DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research";

const BASE_PATH = "";

const PAPER_URL = "https://allenai.org/papers/drtulu";

const Headline = () => (
  <PageHeader className="page-header pb-2 pt-0">
    <Link
      href={PAPER_URL}
      className="inline-flex items-center rounded-lg bg-muted px-3 py-1 text-sm font-medium"
    >
      <span className="sm:hidden">Check our paper!</span>
      <span className="hidden sm:inline">Check our paper!</span>
      <ArrowRightIcon className="ml-1 h-4 w-4" />
    </Link>
    <PageHeaderHeading className="tracking-tight">{TITLE}</PageHeaderHeading>
    <div className="w-full -mx-4">
      <Separator className="mb-0.25 mt-2" />
    </div>
    <div className="flex items-center justify-start gap-2 w-full -mx-4 pl-2">
      <Link
        href="https://github.com/rlresearch/dr-tulu"
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          buttonVariants({ variant: "ghost", size: "sm" }),
          "gap-1.5 text-muted-foreground hover:text-foreground"
        )}
      >
        <Github className="h-3.5 w-3.5" />
        Code
      </Link>
      <Link
        href="https://huggingface.co/collections/rl-research/dr-tulu"
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          buttonVariants({ variant: "ghost", size: "sm" }),
          "gap-1.5 text-muted-foreground hover:text-foreground"
        )}
      >
        <Database className="h-3.5 w-3.5" />
        Data & Models
      </Link>
      <Link
        href="https://allenai.org/blog/dr-tulu"
        target="_blank"
        rel="noopener noreferrer"
        className={cn(
          buttonVariants({ variant: "ghost", size: "sm" }),
          "gap-1.5 text-muted-foreground hover:text-foreground"
        )}
      >
        <BookOpen className="h-3.5 w-3.5" />
        Blogpost
      </Link>
    </div>
  </PageHeader>
);

type Source = {
  id: string;
  title: string;
  url: string;
  snippet?: string;
};

type Document = {
  id: string;
  title: string;
  url: string;
  snippet: string;
  tool_call_id: string;
  tool_name: string;
};

const parseCitationsWithTooltips = (
  text: string,
  sources: Source[]
): React.ReactNode => {
  const parts: React.ReactNode[] = [];
  const regex = /<cite id="([^"]+)">([^<]+)<\/cite>/g;
  let lastIndex = 0;
  let match;

  const sourceIdToNumber = new Map<string, number>();
  sources.forEach((source, index) => {
    sourceIdToNumber.set(source.id, index + 1);
  });

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const citationIds = match[1].split(",");
    const citedText = match[2];
    const citedSources = sources.filter((s) => citationIds.includes(s.id));
    const citationNumbers = citationIds
      .map((id) => sourceIdToNumber.get(id.trim()))
      .filter((num) => num !== undefined) as number[];

    parts.push(
      <CitationTooltip
        key={`cite-${match.index}`}
        sources={citedSources}
        text={citedText}
        citationNumbers={citationNumbers}
      />
    );

    lastIndex = regex.lastIndex;
  }

  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : text;
};

const CitationTooltip = ({
  sources,
  text,
  citationNumbers,
}: {
  sources: Source[];
  text: string;
  citationNumbers: number[];
}) => {
  return (
    <TooltipProvider>
      <Tooltip delayDuration={200}>
        <TooltipTrigger asChild>
          <span className="text-foreground underline decoration-blue-400 decoration-dotted cursor-help transition-colors duration-200 hover:text-blue-600 pl-1">
            {text}
            <sup className="ml-0.5 text-[10px] text-blue-500 font-medium">
              [{citationNumbers.join(", ")}]
            </sup>
          </span>
        </TooltipTrigger>
        <TooltipContent className="max-w-sm p-3" side="top">
          <div className="space-y-2 pl-2">
            {sources.length > 0 ? (
              sources.map((source, index) => (
                <div key={source.id} className="text-xs">
                  <div className="font-semibold">
                    [{citationNumbers[index]}] {source.title}
                  </div>
                  {source.snippet && (
                    <div className="text-muted-foreground mt-1 line-clamp-2">
                      {source.snippet}
                    </div>
                  )}
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-500 hover:text-blue-700 inline-flex items-center gap-1 mt-1"
                  >
                    View Source <ExternalLink className="h-3 w-3" />
                  </a>
                </div>
              ))
            ) : (
              <div className="text-xs text-muted-foreground">
                No source information available
              </div>
            )}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

// Live streaming trace item component for sidebar
// Returns an array of elements (tool call + tool output as separate cards)
const LiveTraceItem = ({
  item,
  index,
}: {
  item: TraceItem;
  index: number;
}) => {
  const [isThinkingOpen, setIsThinkingOpen] = useState(false);
  const [isToolOutputOpen, setIsToolOutputOpen] = useState(false);

  if (item.type === "thinking") {
    const contentWithoutThinkTags = item.content
      .replace(/<\/?think>/g, "")
      .trim();
    const contentTrimmed = isThinkingOpen
      ? contentWithoutThinkTags
      : contentWithoutThinkTags.split(" ").slice(0, 30).join(" ") +
        (contentWithoutThinkTags.split(" ").length > 30 ? "..." : "");

    return (
      <div className="bg-green-50 rounded-md border border-green-200 overflow-hidden hover:shadow-md hover:border-green-300 hover:bg-green-100 max-w-full">
        <Collapsible open={isThinkingOpen} onOpenChange={setIsThinkingOpen}>
          <CollapsibleTrigger className="flex items-center justify-between p-4 w-full hover:bg-muted/50 transition-colors duration-200">
            <span className="text-xs font-semibold text-green-700 flex items-center gap-2">
              Thinking
              {!item.isComplete && (
                <Loader2 className="h-3 w-3 animate-spin" />
              )}
            </span>
            <div className="transform transition-all duration-300 ease-in-out flex-shrink-0">
              {isThinkingOpen ? (
                <ArrowUpFromLine className="h-3.5 w-3.5 text-muted-foreground" />
              ) : (
                <ArrowDownFromLine className="h-3.5 w-3.5 text-muted-foreground" />
              )}
            </div>
          </CollapsibleTrigger>
          <div className="px-4 pb-4 pt-1 max-w-full overflow-hidden">
            <p
              className={cn(
                "text-xs whitespace-pre-wrap font-mono leading-relaxed break-words max-w-full",
                isThinkingOpen ? "" : "text-muted-foreground"
              )}
            >
              {contentTrimmed || "Processing..."}
            </p>
          </div>
        </Collapsible>
      </div>
    );
  }

  if (item.type === "tool_call") {
    const { tool_name, call_id, documents, output, query, params } = item.data;
    return (
      <>
        {/* Tool Call Card - matches static demo style */}
        <div className="bg-blue-50 p-4 rounded-md border border-blue-200 transition-all duration-200 hover:shadow-md hover:border-blue-300 hover:bg-blue-100">
          <div className="flex items-start justify-between gap-2 mb-2 min-w-0">
            <div className="flex-1 min-w-0">
              <span className="text-xs font-semibold text-blue-700">
                Tool Call: {tool_name}
              </span>
              {/* Display params as key-value pairs like static demo */}
              {params && Object.keys(params).length > 0 && (
                <div className="mt-1 space-y-0.5">
                  {Object.entries(params).map(([key, value]) => (
                    <div
                      key={key}
                      className="text-xs font-mono text-muted-foreground break-words min-w-0"
                    >
                      <span className="">{key}:</span> {String(value)}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
          {/* Display query as main content like static demo */}
          {query && (
            <div className="max-h-48 overflow-y-auto overflow-x-hidden min-w-0 font-mono text-xs">
              <p style={{ wordBreak: "break-word", overflowWrap: "anywhere" }}>
                {query}
              </p>
            </div>
          )}
        </div>

        {/* Tool Output Card - separate card like static demo */}
        {output && (
          <div className="bg-background rounded-md overflow-hidden border hover:shadow-md max-w-full">
            <Collapsible open={isToolOutputOpen} onOpenChange={setIsToolOutputOpen}>
              <CollapsibleTrigger className="flex items-center justify-between p-4 w-full hover:bg-muted/50 transition-colors duration-200">
                <span className="text-xs font-semibold">Tool Output</span>
                <div className="transform transition-all duration-300 ease-in-out flex-shrink-0">
                  {isToolOutputOpen ? (
                    <ArrowUpFromLine className="h-3.5 w-3.5 text-muted-foreground" />
                  ) : (
                    <ArrowDownFromLine className="h-3.5 w-3.5 text-muted-foreground" />
                  )}
                </div>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="px-4 pb-4 pt-1 max-w-full overflow-hidden">
                  <div className="max-h-48 overflow-y-auto overflow-x-hidden min-w-0 max-w-full">
                    <p
                      className="text-xs whitespace-pre-wrap font-mono leading-relaxed text-muted-foreground break-words min-w-0 max-w-full"
                      style={{ wordBreak: "break-word", overflowWrap: "anywhere" }}
                    >
                      {output}
                    </p>
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>
        )}
      </>
    );
  }

  if (item.type === "answer") {
    return null; // Don't show answer in traces - it's shown in chat
  }

  return null;
};

// Loading message component with rotating text and stats
const LoadingMessage = ({ traceItems }: { traceItems: TraceItem[] }) => {
  const [loadingText, setLoadingText] = useState("Researching");
  
  useEffect(() => {
    const loadingTexts = [
      "Infodigging", "Factgrabbing", "Knowledging", "Studifying", 
      "Learninating", "Wisdomizing", "Thoughtsifting", "Bookworming", 
      "Researching", "Cognitating", "Discoverifying", "Deepthinking", 
      "Sciencifying", "Scholarizing", "Ideahunting", "Thinkworking", 
      "Smartifying", "Conclusionizing", "Insightfarming", "Infohoovering", 
      "Factstacking", "Databinging", "Factsniffing", "Mindcooking", 
      "Factweaving", "Infopiling"
    ];
    
    const interval = setInterval(() => {
      const randomText = loadingTexts[Math.floor(Math.random() * loadingTexts.length)];
      setLoadingText(randomText);
    }, 2000);
    
    return () => clearInterval(interval);
  }, []);
  
  // Count tool calls by type
  const toolCallCount = traceItems.filter(item => item.type === "tool_call").length;
  const searchCalls = traceItems.filter(
    item => item.type === "tool_call" && 
    (item.data.tool_name.includes("search") || item.data.tool_name.includes("Search"))
  ).length;
  const browseCalls = traceItems.filter(
    item => item.type === "tool_call" && 
    (item.data.tool_name.includes("browse") || item.data.tool_name.includes("Browse"))
  ).length;
  
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        <span className="font-medium">{loadingText}...</span>
      </div>
      {toolCallCount > 0 && (
        <div className="text-xs text-muted-foreground pl-6 space-y-1">
          {searchCalls > 0 && (
            <div className="flex items-center gap-1.5">
              <Search className="h-3 w-3" />
              <span>Searched {searchCalls} {searchCalls === 1 ? 'query' : 'queries'}</span>
            </div>
          )}
          {browseCalls > 0 && (
            <div className="flex items-center gap-1.5">
              <Globe className="h-3 w-3" />
              <span>Browsed {browseCalls} {browseCalls === 1 ? 'page' : 'pages'}</span>
            </div>
          )}
          {toolCallCount > 0 && (
            <div className="flex items-center gap-1.5">
              <Wrench className="h-3 w-3" />
              <span>{toolCallCount} tool {toolCallCount === 1 ? 'call' : 'calls'} total</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// Live Chat Interface Component
const LiveChatInterface = ({
  isPanelOpen,
  onPanelToggle,
}: {
  isPanelOpen: boolean;
  onPanelToggle: () => void;
}) => {
  const [input, setInput] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const panelRef = useRef<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    isConnected,
    isLoading,
    error,
    thinkingContent,
    answerContent,
    traceItems,
    messages,
    metadata,
    sendQuery,
    cancel,
  } = useChat();

  // Handle panel collapse/expand
  useEffect(() => {
    if (panelRef.current) {
      if (isPanelOpen) {
        panelRef.current.expand();
      } else {
        panelRef.current.collapse();
      }
    }
  }, [isPanelOpen]);

  // Auto-scroll when content updates
  useEffect(() => {
    if (scrollAreaRef.current && (isLoading || answerContent || messages.length > 0)) {
      const scrollContainer = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
  }, [answerContent, isLoading, messages]);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!input.trim() || isLoading) return;

      sendQuery(input.trim());
      setInput("");
    },
    [input, isLoading, sendQuery]
  );

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleExportChat = useCallback(() => {
    // Include current answer and traces if they exist
    const exportMessages = [...messages];
    if (answerContent || traceItems.length > 0) {
      exportMessages.push({
        role: "assistant" as const,
        content: answerContent,
        traceItems: traceItems,
      });
    }

    const exportData = {
      messages: exportMessages,
      exportDate: new Date().toISOString(),
    };

    const dataStr = JSON.stringify(exportData, null, 2);
    const blob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `dr-tulu-chat-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, [messages, answerContent, traceItems]);

  const handleLoadChat = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      sessionStorage.setItem("chatData", content);
      window.location.reload();
    };
    reader.readAsText(file);
    
    // Reset the input so the same file can be loaded again
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }, []);

  // Combine all traces (history + current)
  const allTraceItems = [
    ...messages.flatMap((m) => m.traceItems || []),
    ...traceItems,
  ];

  // Extract documents from tool calls for side panel
  const allDocuments: Document[] = allTraceItems
    .filter(
      (item): item is { type: "tool_call"; data: ToolCallData } =>
        item.type === "tool_call"
    )
    .flatMap((item) =>
      (item.data.documents || []).map((doc) => ({
        id: doc.id,
        title: doc.title,
        url: doc.url,
        snippet: doc.snippet,
        tool_call_id: item.data.call_id || "",
        tool_name: item.data.tool_name,
      }))
    );

  // Build sources for citation tooltips (convert Document[] to Source[])
  const sources: Source[] = allDocuments.map((doc) => ({
    id: doc.id,
    title: doc.title,
    url: doc.url,
    snippet: doc.snippet,
  }));

  // Filter documents based on search query
  const filteredDocuments = allDocuments.filter((doc) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      doc.title.toLowerCase().includes(query) ||
      doc.snippet.toLowerCase().includes(query) ||
      doc.url.toLowerCase().includes(query)
    );
  });

  // Count tool calls
  const toolCallCount = allTraceItems.filter(
    (item) => item.type === "tool_call"
  ).length;

  // Check if we have content to show
  const hasContent = messages.length > 0 || isLoading || answerContent;

  return (
    <ResizablePanelGroup direction="horizontal" className="h-[600px]">
      <ResizablePanel defaultSize={65} minSize={30}>
        <div className="flex flex-col h-[600px]">
          <div className="px-4 pl-8 py-3 border-b flex items-center justify-between">
            <h3 className="text-sm font-medium text-muted-foreground">DR Tulu Deep Research Agent</h3>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleExportChat}
                disabled={messages.length === 0}
                className="gap-2 h-8"
              >
                <Download className="h-3.5 w-3.5" />
                Export
              </Button>
              <input
                ref={fileInputRef}
                type="file"
                accept=".json"
                onChange={handleLoadChat}
                className="hidden"
              />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                className="gap-2 h-8"
              >
                <Upload className="h-3.5 w-3.5" />
                Load
              </Button>
            </div>
          </div>
          <ScrollArea ref={scrollAreaRef} className="flex-1 p-4 pl-8">
            <div className="space-y-4">
              {!hasContent ? (
                <div className="flex flex-col items-center justify-center h-full py-16 text-center">
                  <div className="text-muted-foreground mb-4">
                    <p>Ask a research question to get started!</p>
                  </div>
                </div>
              ) : (
                <>
                  {/* Message History */}
                  {messages.map((msg, index) => (
                    <div
                      key={index}
                      className={cn(
                        "flex gap-3",
                        msg.role === "user" ? "justify-end" : "justify-start"
                      )}
                    >
                      {msg.role === "assistant" && (
                        <Avatar className="h-8 w-8">
                          <AvatarImage
                            src={`${BASE_PATH}/images/logo.png`}
                            alt="DR Tulu"
                          />
                          <AvatarFallback className="bg-primary text-primary-foreground">
                            DT
                          </AvatarFallback>
                        </Avatar>
                      )}
                      <div className="flex flex-col gap-2 max-w-[80%]">
                        {msg.role === "assistant" && (
                          <i className="text-xs text-muted-foreground">
                            Answer based on cited docs in the sidebar
                          </i>
                        )}
                        <div
                          className={cn(
                            "rounded-lg px-4 py-3",
                            msg.role === "user"
                              ? "bg-primary text-primary-foreground"
                              : "bg-muted"
                          )}
                        >
                          <div className="text-sm whitespace-pre-wrap leading-relaxed">
                            {msg.role === "assistant"
                              ? parseCitationsWithTooltips(msg.content, sources)
                              : msg.content}
                          </div>
                        </div>
                      </div>
                      {msg.role === "user" && (
                        <Avatar className="h-8 w-8">
                          <AvatarFallback>U</AvatarFallback>
                        </Avatar>
                      )}
                    </div>
                  ))}

                  {/* Current Assistant Response (Streaming) */}
                  {(isLoading || answerContent) && (
                    <div className="flex gap-3 justify-start">
                      <Avatar className="h-8 w-8">
                        <AvatarImage
                          src={`${BASE_PATH}/images/logo.png`}
                          alt="DR Tulu"
                        />
                        <AvatarFallback className="bg-primary text-primary-foreground">
                          DT
                        </AvatarFallback>
                      </Avatar>
                      <div className="flex flex-col gap-2 max-w-[80%]">
                        {answerContent ? (
                          <>
                            <i className="text-xs text-muted-foreground">
                              Answer based on cited docs in the sidebar
                            </i>
                            <div className="rounded-lg px-4 py-3 bg-muted">
                              <div className="text-sm whitespace-pre-wrap leading-relaxed">
                                {sources.length > 0
                                  ? parseCitationsWithTooltips(
                                      answerContent,
                                      sources
                                    )
                                  : answerContent}
                              </div>
                            </div>
                          </>
                        ) : (
                          <div className="rounded-lg px-4 py-3 bg-muted">
                            <LoadingMessage traceItems={traceItems} />
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Error */}
                  {error && (
                    <div className="bg-red-50 border border-red-200 text-red-700 p-4 rounded-md text-sm">
                      Error: {error}
                    </div>
                  )}

                  {/* Metadata */}
                  {metadata && (
                    <div className="text-xs text-muted-foreground border-t pt-4">
                      <span>Tool calls: {metadata.total_tool_calls}</span>
                      {metadata.failed_tool_calls > 0 && (
                        <span className="ml-4 text-amber-600">
                          Failed: {metadata.failed_tool_calls}
                        </span>
                      )}
                      {metadata.searched_links.length > 0 && (
                        <span className="ml-4">
                          Searched: {metadata.searched_links.length} links
                        </span>
                      )}
                      {metadata.browsed_links.length > 0 && (
                        <span className="ml-4">
                          Browsed: {metadata.browsed_links.length} pages
                        </span>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </ScrollArea>

          <form
            onSubmit={handleSubmit}
            className="p-4 pl-8 border-t bg-muted/10"
          >
            <div className="relative">
              <Textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Ask a research question..."
                className="min-h-[80px] max-h-[200px] resize-none pr-12"
                disabled={isLoading}
              />
              <Button
                type="submit"
                size="icon"
                variant="ghost"
                disabled={isLoading || !input.trim()}
                className="absolute bottom-2 right-2 h-8 w-8 rounded-lg hover:bg-muted"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <SendHorizontal className="h-4 w-4" />
                )}
              </Button>
            </div>
            <div className="flex items-center pl-2 mt-2">
              <p className="text-xs text-muted-foreground/60">
                Press Enter to send, Shift+Enter for new line
              </p>
            </div>
          </form>
        </div>
      </ResizablePanel>

      {/* Side Panel with Tabs (like static demo) */}
      {hasContent && (
        <>
          <ResizableHandle withHandle className="w-0" />
          <ResizablePanel
            ref={panelRef}
            defaultSize={35}
            minSize={20}
            maxSize={60}
            collapsible
            collapsedSize={0}
            className="h-[600px]"
          >
            <div className="bg-muted/20 flex flex-col h-full overflow-hidden border-l">
              <Tabs defaultValue="traces" className="flex flex-col h-full">
                <div className="p-4 border-b bg-background">
                  <TabsList className="grid w-full grid-cols-2 transition-all">
                    <TabsTrigger
                      value="documents"
                      className="transition-all duration-300 data-[state=active]:scale-[1.02]"
                    >
                      Cited Docs
                    </TabsTrigger>
                    <TabsTrigger
                      value="traces"
                      className="transition-all duration-300 data-[state=active]:scale-[1.02]"
                    >
                      Full Traces
                    </TabsTrigger>
                  </TabsList>
                </div>

                {/* Traces Tab */}
                <TabsContent
                  value="traces"
                  className="flex-1 overflow-hidden mt-0 data-[state=active]:animate-in data-[state=inactive]:animate-out data-[state=inactive]:fade-out-0 data-[state=active]:fade-in-0 duration-300"
                >
                  <div className="p-4 border-b bg-background">
                    <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                      <span className="flex items-center gap-1 font-medium">
                        <Wrench className="h-3 w-3" />
                        {toolCallCount} tool {toolCallCount === 1 ? 'call' : 'calls'}
                      </span>
                      {(() => {
                        const searchCount = allTraceItems.filter(
                          item => item.type === "tool_call" && 
                          (item.data.tool_name.includes("search") || item.data.tool_name.includes("Search"))
                        ).length;
                        const browseCount = allTraceItems.filter(
                          item => item.type === "tool_call" && 
                          (item.data.tool_name.includes("browse") || item.data.tool_name.includes("Browse"))
                        ).length;
                        return (
                          <>
                            {searchCount > 0 && (
                              <span className="flex items-center gap-1">
                                <Search className="h-3 w-3" />
                                {searchCount} searches
                              </span>
                            )}
                            {browseCount > 0 && (
                              <span className="flex items-center gap-1">
                                <Globe className="h-3 w-3" />
                                {browseCount} pages
                              </span>
                            )}
                          </>
                        );
                      })()}
                      {isLoading && (
                        <span className="flex items-center gap-1 text-blue-600 font-medium">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Processing...
                        </span>
                      )}
                    </div>
                  </div>
                  <ScrollArea className="h-[calc(100%-3rem)] p-4">
                    <div className="space-y-3">
                      {allTraceItems.length === 0 && isLoading ? (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2 text-muted-foreground text-sm">
                            <Loader2 className="h-4 w-4 animate-spin" />
                            <span>Initializing research agents...</span>
                          </div>
                          <p className="text-xs text-muted-foreground pl-6">
                            Setting up search and reasoning pipeline
                          </p>
                        </div>
                      ) : (
                        allTraceItems.map((item, index) => (
                          <LiveTraceItem
                            key={index}
                            item={item}
                            index={index}
                          />
                        ))
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>

                {/* Documents Tab */}
                <TabsContent
                  value="documents"
                  className="flex-1 overflow-hidden mt-0"
                >
                  <div className="p-4 border-b bg-background">
                    <div className="relative flex items-center">
                      <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Search documents..."
                        value={searchQuery}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                          setSearchQuery(e.target.value)
                        }
                        className="pl-8 pr-32 h-9 text-xs"
                      />
                      <div className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-muted-foreground pointer-events-none">
                        {searchQuery ? (
                          <span>
                            Showing {filteredDocuments.length} of{" "}
                            {allDocuments.length} result
                            {allDocuments.length !== 1 ? "s" : ""}
                          </span>
                        ) : (
                          <span>{allDocuments.length} retrieved</span>
                        )}
                      </div>
                    </div>
                  </div>
                  <ScrollArea className="h-[calc(100%-5rem)] p-4">
                    <div className="space-y-4">
                      {filteredDocuments.length > 0 ? (
                        filteredDocuments.map((doc, index) => (
                          <div
                            key={`${doc.tool_call_id}-${doc.id}`}
                            className="bg-background p-4 rounded-md border transition-all duration-200 hover:shadow-md hover:border-primary/30 hover:bg-muted/30 cursor-pointer"
                          >
                            <div className="flex items-start justify-between gap-2 mb-2">
                              <h4 className="font-semibold text-sm flex-1">
                                {doc.title}
                              </h4>
                              <span className="text-xs text-muted-foreground bg-muted px-2 py-0.5 rounded">
                                #{index + 1}
                              </span>
                            </div>
                            <p className="text-xs text-muted-foreground mb-2">
                              {doc.snippet}
                            </p>
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-xs text-muted-foreground">
                                {doc.tool_name}
                              </span>
                              <a
                                href={doc.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-xs text-blue-500 hover:text-blue-700 inline-flex items-center gap-1"
                              >
                                View <ExternalLink className="h-3 w-3" />
                              </a>
                            </div>
                          </div>
                        ))
                      ) : allDocuments.length === 0 ? (
                        <div className="text-center text-sm text-muted-foreground py-8">
                          {isLoading
                            ? "Searching for documents..."
                            : "No documents retrieved yet"}
                        </div>
                      ) : (
                        <div className="text-center text-sm text-muted-foreground py-8">
                          No documents found matching &quot;{searchQuery}&quot;
                        </div>
                      )}
                    </div>
                  </ScrollArea>
                </TabsContent>
              </Tabs>
            </div>
          </ResizablePanel>
        </>
      )}
    </ResizablePanelGroup>
  );
};


export default function Home() {
  const [isPanelOpen, setIsPanelOpen] = useState<boolean>(true);

  return (
    <div className="min-h-screen flex flex-col">
      <div className="container relative p-16 flex-1">
        <Headline />
        <div className="mt-8 rounded-[0.5rem] border bg-background shadow overflow-hidden">
          <div className="px-0 pt-0 pb-0">
            <LiveChatInterface
              isPanelOpen={isPanelOpen}
              onPanelToggle={() => setIsPanelOpen(!isPanelOpen)}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
