"use client";
import Link from "next/link";
import Image from "next/image";
import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  ArrowRightIcon,
  SendHorizontal,
  ChevronRight,
  ExternalLink,
  PanelLeftClose,
  PanelLeftOpen,
  Search,
  ArrowDownFromLine,
  ArrowUpFromLine,
  Expand,
  Github,
  BookOpen,
  Database,
  Wifi,
  WifiOff,
  Loader2,
} from "lucide-react";
import { Button, buttonVariants } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  HoverCard,
  HoverCardTrigger,
  HoverCardContent,
} from "@/components/ui/hover-card";
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { nanoid } from "nanoid";

import {
  PageHeader,
  PageHeaderDescription,
  PageHeaderHeading,
} from "@/app/components/page-header";
import { cn } from "@/lib/utils";
import { useChat, TraceItem, ToolCallData, DocumentData, Message as SSEMessage } from "@/lib/sse";

const TITLE =
  "DR Tulu: Reinforcement Learning with Evolving Rubrics for Deep Research";

const BASE_PATH = "";

const FULL_AUTHORS = [
  {
    name: "Rulin Shao",
    affiliation: "University of Washington",
    isFirstAuthor: true,
    isCoreContributor: true,
  },
  {
    name: "Akari Asai",
    affiliation: "Allen Institute for AI, Carnegie Mellon University",
    isFirstAuthor: true,
    isCoreContributor: true,
  },
  {
    name: "Shannon Zejiang Shen",
    affiliation: "Massachusetts Institute of Technology",
    isFirstAuthor: true,
    isCoreContributor: true,
  },
  {
    name: "Hamish Ivison",
    affiliation: "University of Washington, Allen Institute for AI",
    isFirstAuthor: true,
    isCoreContributor: true,
  },
  {
    name: "Varsha Kishore",
    affiliation: "University of Washington, Allen Institute for AI",
    isCoreContributor: true,
  },
  {
    name: "Jingming Zhuo",
    affiliation: "University of Washington",
    isCoreContributor: true,
  },
  { name: "Xinran Zhao", affiliation: "Carnegie Mellon University" },
  { name: "Molly Park", affiliation: "University of Washington" },
  {
    name: "Samuel Finlayson",
    affiliation: "University of Washington, Seattle Children's Hospital",
  },
  {
    name: "David Sontag",
    affiliation: "Massachusetts Institute of Technology",
  },
  { name: "Tyler Murray", affiliation: "Allen Institute for AI" },
  {
    name: "Sewon Min",
    affiliation: "Allen Institute for AI, University of California, Berkeley",
  },
  { name: "Pradeep Dasigi", affiliation: "Allen Institute for AI" },
  { name: "Luca Soldaini", affiliation: "Allen Institute for AI" },
  { name: "Faeze Brahman", affiliation: "Allen Institute for AI" },
  { name: "Wen-tau Yih", affiliation: "University of Washington" },
  { name: "Tongshuang Wu", affiliation: "Carnegie Mellon University" },
  { name: "Luke Zettlemoyer", affiliation: "University of Washington" },
  { name: "Yoon Kim", affiliation: "Massachusetts Institute of Technology" },
  {
    name: "Hannaneh Hajishirzi",
    affiliation: "University of Washington, Allen Institute for AI",
  },
  {
    name: "Pang Wei Koh",
    affiliation: "University of Washington, Allen Institute for AI",
  },
];

const PAPER_URL = "https://allenai.org/papers/drtulu";

const scrollToAuthors = () => {
  const authorsSection = document.getElementById("authors-section");
  if (authorsSection) {
    authorsSection.scrollIntoView({ behavior: "smooth" });
  }
};

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
    {/* <PageHeaderDescription>
      An advanced AI assistant for long-form deep research with adaptive evaluation rubrics.
    </PageHeaderDescription> */}
    <div className="w-full -mx-4">
      <Separator className="mb-0.25 mt-2" />
    </div>
    <div className="flex items-center justify-between gap-4 w-full -mx-4 px-4">
      <div className="flex items-center gap-2">
        <span className="text-sm text-muted-foreground">DR Tulu Authors</span>
        <Button
          variant="link"
          className="px-0 text-sm inline-flex items-center gap-1 hover:gap-2 transition-all"
          onClick={scrollToAuthors}
        >
          See full author list below
          <ArrowRightIcon className="h-3 w-3" />
        </Button>
      </div>
      <div className="flex items-center gap-2">
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
    </div>
  </PageHeader>
);

type Source = {
  id: string;
  title: string;
  url: string;
  snippet?: string;
};

type FullTraces = {
  generated_text: string;
  total_tokens: number;
  tool_call_count: number;
};

type Document = {
  id: string;
  title: string;
  url: string;
  snippet: string;
  tool_call_id: string;
  tool_name: string;
};

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: Source[];
  fullTraces?: FullTraces;
  documents?: Document[];
};

type ExampleData = {
  example_id: string;
  problem: string;
  final_response: string;
  full_traces: {
    generated_text: string;
    total_tokens: number;
    tool_call_count: number;
    tool_calls: Array<{
      tool_name: string;
      call_id: string;
      documents: Array<{
        id: string;
        title: string;
        url: string;
        snippet: string;
      }>;
    }>;
  };
};

type ExampleListItem = {
  dataset_name: string;
  example_title: string;
  json_file_name: string;
};

// Utility: Parse citations in text and return React elements with tooltips
const parseCitationsWithTooltips = (
  text: string,
  sources: Source[]
): React.ReactNode => {
  const parts: React.ReactNode[] = [];
  const regex = /<cite id="([^"]+)">([^<]+)<\/cite>/g;
  let lastIndex = 0;
  let match;

  // Create a mapping from source IDs to citation numbers
  const sourceIdToNumber = new Map<string, number>();
  sources.forEach((source, index) => {
    sourceIdToNumber.set(source.id, index + 1);
  });

  while ((match = regex.exec(text)) !== null) {
    // Add text before the citation
    if (match.index > lastIndex) {
      parts.push(text.slice(lastIndex, match.index));
    }

    const citationIds = match[1].split(",");
    const citedText = match[2];
    const citedSources = sources.filter((s) => citationIds.includes(s.id));
    const citationNumbers = citationIds
      .map((id) => sourceIdToNumber.get(id.trim()))
      .filter((num) => num !== undefined) as number[];

    // Create citation with tooltip
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

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }

  return parts.length > 0 ? parts : text;
};

// Citation Tooltip Component
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
          <span className="text-foreground underline decoration-blue-400 decoration-dotted cursor-help transition-colors duration-200 hover:text-blue-600">
            {text}
            <sup className="ml-0.5 text-[10px] text-blue-500 font-medium">
              [{citationNumbers.join(", ")}]
            </sup>
          </span>
        </TooltipTrigger>
        <TooltipContent className="max-w-sm p-3" side="top">
          <div className="space-y-2">
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

// Sources Collapsible Component
const SourcesCollapsible = ({ sources }: { sources: Source[] }) => {
  const [isOpen, setIsOpen] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isOpen && contentRef.current) {
      setTimeout(() => {
        const scrollContainer = contentRef.current?.closest(
          "[data-radix-scroll-area-viewport]"
        );
        if (scrollContainer) {
          scrollContainer.scrollBy({
            top: 60,
            behavior: "auto",
          });
        }
      }, 150);
    }
  }, [isOpen]);

  return (
    <div className="px-2">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CollapsibleTrigger className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
          <ChevronRight
            className={cn(
              "h-3 w-3 transition-transform duration-200",
              isOpen && "rotate-90"
            )}
          />
          <span>Sources ({sources.length})</span>
        </CollapsibleTrigger>
        <CollapsibleContent
          ref={contentRef}
          className="mt-2 animate-in slide-in-from-top-1 duration-200"
        >
          <ol className="list-decimal list-inside space-y-1 text-xs">
            {sources.map((source) => (
              <li key={source.id} className="text-muted-foreground">
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 hover:underline"
                >
                  {source.title}
                </a>
              </li>
            ))}
          </ol>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
};

// Parse full traces into structured sections
type TraceSection = {
  type: "text" | "tool_call" | "tool_output";
  content: string;
  toolName?: string;
  toolParams?: Record<string, string>;
};

const parseFullTraces = (generatedText: string): TraceSection[] => {
  const sections: TraceSection[] = [];
  const toolCallRegex =
    /<call_tool name="([^"]+)"([^>]*)>([\s\S]*?)<\/call_tool>/g;
  const toolOutputRegex = /<tool_output>([\s\S]*?)<\/tool_output>/g;

  let lastIndex = 0;
  const matches: Array<{
    type: "call" | "output";
    index: number;
    endIndex: number;
    content: string;
    name?: string;
    params?: Record<string, string>;
  }> = [];

  // Find all tool calls
  let match;
  while ((match = toolCallRegex.exec(generatedText)) !== null) {
    const toolName = match[1];
    const paramsString = match[2];
    const content = match[3];

    // Parse parameters
    const params: Record<string, string> = {};
    const paramRegex = /(\w+)="([^"]*)"/g;
    let paramMatch;
    while ((paramMatch = paramRegex.exec(paramsString)) !== null) {
      params[paramMatch[1]] = paramMatch[2];
    }

    matches.push({
      type: "call",
      index: match.index,
      endIndex: match.index + match[0].length,
      content,
      name: toolName,
      params,
    });
  }

  // Find all tool outputs
  toolOutputRegex.lastIndex = 0;
  while ((match = toolOutputRegex.exec(generatedText)) !== null) {
    matches.push({
      type: "output",
      index: match.index,
      endIndex: match.index + match[0].length,
      content: match[1],
    });
  }

  // Sort matches by index
  matches.sort((a, b) => a.index - b.index);

  // Build sections
  matches.forEach((match) => {
    // Add text before this match
    if (match.index > lastIndex) {
      const textContent = generatedText.slice(lastIndex, match.index).trim();
      if (textContent) {
        sections.push({
          type: "text",
          content: textContent,
        });
      }
    }

    // Add the match itself
    if (match.type === "call") {
      sections.push({
        type: "tool_call",
        content: match.content.trim(),
        toolName: match.name,
        toolParams: match.params,
      });
    } else {
      sections.push({
        type: "tool_output",
        content: match.content.trim(),
      });
    }

    lastIndex = match.endIndex;
  });

  // Add any remaining text
  if (lastIndex < generatedText.length) {
    const textContent = generatedText.slice(lastIndex).trim();
    if (textContent) {
      sections.push({
        type: "text",
        content: textContent,
      });
    }
  }

  return sections;
};

// Component to render a single trace section
const TraceSection = ({
  section,
  index,
}: {
  section: TraceSection;
  index: number;
}) => {
  const [isThinkingOpen, setIsThinkingOpen] = useState(false);
  const [isToolCallOpen, setIsToolCallOpen] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  if (section.type === "text") {
    const contentWithoutThinkTags = section.content
      .replace(/<\/?think>/g, "")
      .trim();
    const contentTrimed = isThinkingOpen
      ? contentWithoutThinkTags
      : contentWithoutThinkTags.split(" ").slice(0, 30).join(" ") + "...";
    return (
      <div className="bg-green-50 rounded-md border border-green-200 overflow-hidden hover:shadow-md hover:border-green-300 hover:bg-green-100">
        <Collapsible open={isThinkingOpen} onOpenChange={setIsThinkingOpen}>
          <CollapsibleTrigger className="flex items-center justify-between p-4 w-full hover:bg-muted/50 transition-colors duration-200 ">
            <span className="text-xs font-semibold  text-green-700">
              Thinking
            </span>
            <div
              className={cn(
                "transform transition-all duration-300 ease-in-out",
                isThinkingOpen ? "rotate-0 scale-100" : "rotate-0 scale-100"
              )}
            >
              {isThinkingOpen ? (
                <ArrowUpFromLine className="h-3.5 w-3.5 text-muted-foreground" />
              ) : (
                <ArrowDownFromLine className="h-3.5 w-3.5 text-muted-foreground" />
              )}
            </div>
          </CollapsibleTrigger>
          <div className="px-4 pb-4 pt-1">
            <p
              className={cn(
                "text-xs whitespace-pre-wrap font-mono leading-relaxed break-words",
                isThinkingOpen ? "" : "text-muted-foreground"
              )}
            >
              {contentTrimed}
            </p>
          </div>
        </Collapsible>
      </div>
    );
  }

  if (section.type === "tool_call") {
    return (
      <div className="bg-blue-50 p-4 rounded-md border border-blue-200 transition-all duration-200 hover:shadow-md hover:border-blue-300 hover:bg-blue-100 cursor-pointer">
        <div className="flex items-start justify-between gap-2 mb-2 min-w-0">
          <div className="flex-1 min-w-0">
            <span className="text-xs font-semibold text-blue-700">
              Tool Call: {section.toolName}
            </span>
            {section.toolParams &&
              Object.keys(section.toolParams).length > 0 && (
                <div className="mt-1 space-y-0.5">
                  {Object.entries(section.toolParams).map(([key, value]) => (
                    <div
                      key={key}
                      className="text-xs font-mono text-muted-foreground break-words min-w-0"
                    >
                      <span className="">{key}:</span> {value}
                    </div>
                  ))}
                </div>
              )}
          </div>
          {/* <span className="text-xs text-muted-foreground bg-blue-100 px-2 py-0.5 rounded flex-shrink-0">
            #{index + 1}
          </span> */}
        </div>
        <div className="max-h-48 overflow-y-auto overflow-x-hidden min-w-0 font-mono text-xs">
          <p style={{ wordBreak: "break-word", overflowWrap: "anywhere" }}>
            {section.content}
          </p>
        </div>
      </div>
    );
  }

  if (section.type === "tool_output") {
    return (
      <div className="bg-background rounded-md overflow-hidden border  hover:shadow-md">
        <Collapsible open={isToolCallOpen} onOpenChange={setIsToolCallOpen}>
          <CollapsibleTrigger className="flex items-center justify-between p-4 w-full hover:bg-muted/50 transition-colors duration-200 ">
            <span className="text-xs font-semibold">Tool Output</span>
            <div className="transform transition-all duration-300 ease-in-out">
              {isToolCallOpen ? (
                <ArrowUpFromLine className="h-3.5 w-3.5 text-muted-foreground" />
              ) : (
                <ArrowDownFromLine className="h-3.5 w-3.5 text-muted-foreground" />
              )}
            </div>
          </CollapsibleTrigger>
          <CollapsibleContent>
            <div className="px-4 pb-4 pt-1">
              <div className="max-h-48 overflow-y-auto overflow-x-hidden min-w-0">
                <p
                  className="text-xs whitespace-pre-wrap font-mono leading-relaxed text-muted-foreground break-words min-w-0 max-w-full"
                  style={{ wordBreak: "break-word", overflowWrap: "anywhere" }}
                >
                  {section.content}
                </p>
              </div>
            </div>
          </CollapsibleContent>
        </Collapsible>
      </div>
    );
  }

  return null;
};

// Side Panel Component with Tabs
const SidePanel = ({
  fullTraces,
  documents,
  sources,
}: {
  fullTraces: FullTraces;
  documents: Document[];
  sources: Source[];
}) => {
  const [searchQuery, setSearchQuery] = useState("");
  const [parsedTraces, setParsedTraces] = useState<TraceSection[]>([]);

  useEffect(() => {
    const sections = parseFullTraces(fullTraces.generated_text);
    setParsedTraces(sections);
  }, [fullTraces.generated_text]);

  // Create mapping from document ID to citation number
  const docIdToCitationNumber = new Map<string, number>();
  sources.forEach((source, index) => {
    docIdToCitationNumber.set(source.id, index + 1);
  });

  const filteredDocuments = documents
    .filter((doc) => {
      const query = searchQuery.toLowerCase();
      return (
        doc.title.toLowerCase().includes(query) ||
        doc.snippet.toLowerCase().includes(query) ||
        doc.url.toLowerCase().includes(query)
      );
    })
    .sort((a, b) => {
      const numA = docIdToCitationNumber.get(a.id) || 999;
      const numB = docIdToCitationNumber.get(b.id) || 999;
      return numA - numB;
    });

  return (
    <div className="bg-muted/20 flex flex-col h-full overflow-hidden border-l ">
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

        <TabsContent
          value="traces"
          className="flex-1 overflow-hidden mt-0 data-[state=active]:animate-in data-[state=inactive]:animate-out data-[state=inactive]:fade-out-0 data-[state=active]:fade-in-0 data-[state=inactive]:slide-out-to-left-2 data-[state=active]:slide-in-from-right-2 duration-300"
        >
          <div className="p-4 border-b bg-background">
            <div className="flex gap-4 text-xs text-muted-foreground">
              <span>Tokens: {fullTraces.total_tokens.toLocaleString()}</span>
              <span>Tool Calls: {fullTraces.tool_call_count}</span>
            </div>
          </div>
          <ScrollArea className="h-[calc(100%-3rem)] p-4">
            <div className="space-y-3">
              {parsedTraces.map((section, index) => (
                <TraceSection key={index} section={section} index={index} />
              ))}
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent value="documents" className="flex-1 overflow-hidden mt-0">
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
                    Showing {filteredDocuments.length} of {documents.length}{" "}
                    result{documents.length !== 1 ? "s" : ""}
                  </span>
                ) : (
                  <span>{documents.length} retrieved</span>
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
                        #
                        {docIdToCitationNumber.get(doc.id) ||
                          documents.indexOf(doc) + 1}
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
  );
};

// List available examples
const listExamples = async (): Promise<ExampleListItem[]> => {
  const response = await fetch(`${BASE_PATH}/example-list.json`);
  if (!response.ok) {
    console.error("Failed to load example-list.json");
    return [];
  }
  return response.json();
};

// Load example JSON
const loadExampleData = async (
  jsonFileName: string
): Promise<ExampleData | null> => {
  const response = await fetch(`${BASE_PATH}/${jsonFileName}`);
  if (!response.ok) {
    console.error(`Failed to load ${jsonFileName}`);
    return null;
  }
  return response.json();
};

// Extract snippet IDs that are actually cited in the final response
const getCitedSnippetIds = (finalResponse: string): Set<string> => {
  const citedIds = new Set<string>();
  const regex = /<cite id="([^"]+)">/g;
  let match;

  while ((match = regex.exec(finalResponse)) !== null) {
    const ids = match[1].split(",").map((id) => id.trim());
    ids.forEach((id) => citedIds.add(id));
  }

  return citedIds;
};

// Parse snippet blocks from generated_text
const parseSnippetsFromGeneratedText = (
  generatedText: string
): Map<string, Source> => {
  const snippetMap = new Map<string, Source>();

  // Step 1: Find all <snippet id=xxx>...</snippet> blocks
  const snippetBlockRegex = /<snippet id=([^\s>]+)>([\s\S]*?)<\/snippet>/g;
  let blockMatch;

  while ((blockMatch = snippetBlockRegex.exec(generatedText)) !== null) {
    const id = blockMatch[1].trim();
    const content = blockMatch[2];

    // Step 2: Extract fields from the content
    // Match lines that start with "Title:", "URL:", "Snippet:"
    const titleMatch = content.match(/^\s*Title:\s*(.+)$/m);
    const urlMatch = content.match(/^\s*URL:\s*(.+)$/m);
    const snippetMatch = content.match(
      /^\s*Snippet:\s*([\s\S]+?)(?=^\s*(?:Title:|URL:|Snippet:|$))/m
    );

    const title = titleMatch ? titleMatch[1].trim() : "";
    const url = urlMatch ? urlMatch[1].trim() : "";
    const snippet = snippetMatch ? snippetMatch[1].trim() : content.trim();

    snippetMap.set(id, {
      id,
      title,
      url,
      snippet,
    });
  }

  return snippetMap;
};

// Extract sources from example data - only include cited sources
const extractSources = (data: ExampleData): Source[] => {
  const citedIds = getCitedSnippetIds(data.final_response);
  const snippetMap = parseSnippetsFromGeneratedText(
    data.full_traces.generated_text
  );

  const sources: Source[] = [];

  // Only include sources that are actually cited
  citedIds.forEach((id) => {
    const snippet = snippetMap.get(id);
    if (snippet) {
      sources.push(snippet);
    }
  });

  return sources;
};

// Extract documents from tool calls - only include cited documents
const extractDocuments = (data: ExampleData): Document[] => {
  const citedIds = getCitedSnippetIds(data.final_response);
  const documents: Document[] = [];

  if (data.full_traces.tool_calls) {
    data.full_traces.tool_calls.forEach((toolCall) => {
      if (toolCall.documents) {
        toolCall.documents.forEach((doc, index) => {
          // Create snippet ID in the format: call_id-index (e.g., "fb888718-0")
          const snippetId = `${toolCall.call_id}-${index}`;

          // Only include documents that are actually cited in the response
          if (citedIds.has(snippetId)) {
            documents.push({
              id: snippetId,
              title: doc.title,
              url: doc.url,
              snippet: doc.snippet,
              tool_call_id: toolCall.call_id,
              tool_name: toolCall.tool_name,
            });
          }
        });
      }
    });
  }

  return documents;
};

// Convert example data to messages
const convertExampleToMessages = (data: ExampleData): Message[] => {
  const sources = extractSources(data);
  const documents = extractDocuments(data);

  return [
    {
      id: nanoid(),
      role: "user",
      content: data.problem,
      timestamp: new Date(),
    },
    {
      id: nanoid(),
      role: "assistant",
      content: data.final_response,
      timestamp: new Date(),
      sources,
      documents,
      fullTraces: {
        generated_text: data.full_traces.generated_text,
        total_tokens: data.full_traces.total_tokens,
        tool_call_count: data.full_traces.tool_call_count,
      },
    },
  ];
};

const ChatInterface = ({
  selectedExample,
  isPanelOpen,
  onPanelToggle,
}: {
  selectedExample: string;
  isPanelOpen: boolean;
  onPanelToggle: () => void;
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const panelRef = useRef<any>(null);
  const isInitialLoadRef = useRef<boolean>(true);

  // Load example data on mount
  useEffect(() => {
    const loadData = async () => {
      setIsLoading(true);
      const data = await loadExampleData(selectedExample);
      if (data) {
        const msgs = convertExampleToMessages(data);
        setMessages(msgs);
      }
      setIsLoading(false);
    };
    loadData();
  }, [selectedExample]);

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

  useEffect(() => {
    // Skip auto-scroll on initial load, only auto-scroll when new messages are added
    if (isInitialLoadRef.current && messages.length > 0) {
      isInitialLoadRef.current = false;
      return;
    }
    /*
    // Auto-scroll to bottom when new messages arrive (but not on initial load)
    if (scrollAreaRef.current && !isInitialLoadRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight;
      }
    }
      */
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // Disabled for now - will be enabled later
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <ResizablePanelGroup direction="horizontal" className="h-[600px]">
      <ResizablePanel defaultSize={65} minSize={30}>
        <div className="flex flex-col h-[600px]">
          <ScrollArea ref={scrollAreaRef} className="flex-1 p-4 pl-8">
            <div className="space-y-4">
              {isLoading ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-muted-foreground">
                    Loading example...
                  </div>
                </div>
              ) : (
                messages.map((message) => (
                  <div
                    key={message.id}
                    className={cn(
                      "flex gap-3",
                      message.role === "user" ? "justify-end" : "justify-start"
                    )}
                  >
                    {message.role === "assistant" && (
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
                      {message.role === "assistant" && (
                        <i className="text-xs text-muted-foreground">
                          Answer based on cited docs in the sidebar
                        </i>
                      )}
                      <div
                        className={cn(
                          "rounded-lg px-4 py-3",
                          message.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted"
                        )}
                      >
                        <div className="text-sm whitespace-pre-wrap leading-relaxed">
                          {message.role === "assistant" && message.sources
                            ? parseCitationsWithTooltips(
                                message.content,
                                message.sources
                              )
                            : message.content}
                        </div>
                      </div>

                      {/*message.role === "assistant" &&
                        message.sources &&
                        message.sources.length > 0 && (
                          <SourcesCollapsible sources={message.sources} />
                        )*/}
                    </div>
                    {message.role === "user" && (
                      <Avatar className="h-8 w-8">
                        <AvatarFallback>U</AvatarFallback>
                      </Avatar>
                    )}
                  </div>
                ))
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
                placeholder="Chat functionality coming soon..."
                className="min-h-[80px] max-h-[200px] resize-none pr-12 opacity-50"
                disabled={true}
              />
              <Button
                type="submit"
                size="icon"
                variant="ghost"
                disabled={true}
                className="absolute bottom-2 right-2 h-8 w-8 rounded-lg hover:bg-muted"
              >
                <SendHorizontal className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex pl-2 mt-2">
              <p className="text-xs text-muted-foreground/60">
                Interactive chat coming soon. Currently displaying randomly
                sampled example research output.
              </p>
            </div>
          </form>
        </div>
      </ResizablePanel>

      {/* Side Panel for Full Traces and Documents */}
      {messages.length > 0 && messages[1]?.fullTraces && (
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
            <SidePanel
              fullTraces={messages[1].fullTraces}
              documents={messages[1].documents || []}
              sources={messages[1].sources || []}
            />
          </ResizablePanel>
        </>
      )}
    </ResizablePanelGroup>
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
      <div className="bg-green-50 rounded-md border border-green-200 overflow-hidden hover:shadow-md hover:border-green-300 hover:bg-green-100">
        <Collapsible open={isThinkingOpen} onOpenChange={setIsThinkingOpen}>
          <CollapsibleTrigger className="flex items-center justify-between p-4 w-full hover:bg-muted/50 transition-colors duration-200">
            <span className="text-xs font-semibold text-green-700 flex items-center gap-2">
              Thinking
              {!item.isComplete && (
                <Loader2 className="h-3 w-3 animate-spin" />
              )}
            </span>
            <div className="transform transition-all duration-300 ease-in-out">
              {isThinkingOpen ? (
                <ArrowUpFromLine className="h-3.5 w-3.5 text-muted-foreground" />
              ) : (
                <ArrowDownFromLine className="h-3.5 w-3.5 text-muted-foreground" />
              )}
            </div>
          </CollapsibleTrigger>
          <div className="px-4 pb-4 pt-1">
            <p
              className={cn(
                "text-xs whitespace-pre-wrap font-mono leading-relaxed break-words",
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
          <div className="bg-background rounded-md overflow-hidden border hover:shadow-md">
            <Collapsible open={isToolOutputOpen} onOpenChange={setIsToolOutputOpen}>
              <CollapsibleTrigger className="flex items-center justify-between p-4 w-full hover:bg-muted/50 transition-colors duration-200">
                <span className="text-xs font-semibold">Tool Output</span>
                <div className="transform transition-all duration-300 ease-in-out">
                  {isToolOutputOpen ? (
                    <ArrowUpFromLine className="h-3.5 w-3.5 text-muted-foreground" />
                  ) : (
                    <ArrowDownFromLine className="h-3.5 w-3.5 text-muted-foreground" />
                  )}
                </div>
              </CollapsibleTrigger>
              <CollapsibleContent>
                <div className="px-4 pb-4 pt-1">
                  <div className="max-h-48 overflow-y-auto overflow-x-hidden min-w-0">
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
          <ScrollArea ref={scrollAreaRef} className="flex-1 p-4 pl-8">
            <div className="space-y-4">
              {!hasContent ? (
                <div className="flex flex-col items-center justify-center h-full py-16 text-center">
                  <div className="text-muted-foreground mb-4">
                    {isConnected ? (
                      <>
                        <Wifi className="h-8 w-8 mx-auto mb-2 text-green-500" />
                        <p>
                          Connected to server. Ask a question to start
                          researching!
                        </p>
                      </>
                    ) : (
                      <>
                        <WifiOff className="h-8 w-8 mx-auto mb-2 text-red-500" />
                        <p>Connecting to server...</p>
                      </>
                    )}
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
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                              <Loader2 className="h-4 w-4 animate-spin" />
                              <span>Researching...</span>
                            </div>
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
                placeholder={
                  isConnected
                    ? "Ask a research question..."
                    : "Connecting to server..."
                }
                className={cn(
                  "min-h-[80px] max-h-[200px] resize-none pr-12",
                  !isConnected && "opacity-50"
                )}
                disabled={!isConnected || isLoading}
              />
              <Button
                type="submit"
                size="icon"
                variant="ghost"
                disabled={!isConnected || isLoading || !input.trim()}
                className="absolute bottom-2 right-2 h-8 w-8 rounded-lg hover:bg-muted"
              >
                {isLoading ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <SendHorizontal className="h-4 w-4" />
                )}
              </Button>
            </div>
            <div className="flex items-center pl-2 mt-2 gap-4">
              <div className="flex items-center gap-1.5">
                {isConnected ? (
                  <Wifi className="h-3 w-3 text-green-500" />
                ) : (
                  <WifiOff className="h-3 w-3 text-red-500" />
                )}
                <span className="text-xs text-muted-foreground">
                  {isConnected ? "Connected" : "Disconnected"}
                </span>
              </div>
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
                    <div className="flex gap-4 text-xs text-muted-foreground">
                      <span>Tool Calls: {toolCallCount}</span>
                      {isLoading && (
                        <span className="flex items-center gap-1">
                          <Loader2 className="h-3 w-3 animate-spin" />
                          Processing...
                        </span>
                      )}
                    </div>
                  </div>
                  <ScrollArea className="h-[calc(100%-3rem)] p-4">
                    <div className="space-y-3">
                      {allTraceItems.length === 0 && isLoading ? (
                        <div className="flex items-center gap-2 text-muted-foreground text-sm">
                          <Loader2 className="h-4 w-4 animate-spin" />
                          <span>Starting research...</span>
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

const AuthorsSection = () => (
  <section id="authors-section" className="mt-16 py-12 bg-muted/10">
    <div className="container px-16">
      <div className="space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-semibold tracking-tight">Authors</h2>
          <p className="text-sm text-muted-foreground">
            ♡ Joint first authors, † Core contributors
          </p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {FULL_AUTHORS.map((author, index) => (
            <div
              key={index}
              className="bg-background rounded-lg border p-4 hover:shadow-md transition-all duration-200 hover:border-primary/30"
            >
              <div className="space-y-1">
                <h3 className="font-medium text-sm">
                  {author.name}
                  {author.isFirstAuthor && author.isCoreContributor && (
                    <sup className="text-xs ml-0.5 text-primary">♡†</sup>
                  )}
                  {!author.isFirstAuthor && author.isCoreContributor && (
                    <sup className="text-xs ml-0.5 text-primary">†</sup>
                  )}
                </h3>
                <p className="text-xs text-muted-foreground">
                  {author.affiliation}
                </p>
              </div>
            </div>
          ))}
        </div>
        <div className="text-center">
          <Link
            href={PAPER_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors inline-flex items-center gap-1"
          >
            See full author contributions in paper
            <ExternalLink className="h-3 w-3" />
          </Link>
        </div>
      </div>
    </div>
  </section>
);

const Footer = () => (
  <footer className="border-t bg-muted/20">
    <div className="container py-6 px-16">
      <div className="flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="text-sm text-muted-foreground">
          © 2025 DR Tulu Authors. All rights reserved.
        </div>
        <div className="flex items-center gap-6">
          <Link
            href={PAPER_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Paper
          </Link>
          <Link
            href="https://rl-rag.github.io/dr-tulu"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Project Page
          </Link>
          <Link
            href="mailto:dr.tulu@gmail.com"
            className="text-sm text-muted-foreground hover:text-foreground transition-colors"
          >
            Contact
          </Link>
        </div>
      </div>
    </div>
  </footer>
);

const MobileView = () => (
  <div className="min-h-screen flex flex-col">
    <div className="container relative p-4">
      <div className="mb-6">
        <h1 className="text-2xl font-bold tracking-tight mb-2">{TITLE}</h1>
        <p className="text-sm text-muted-foreground mb-4">
          For the best experience, please visit this demo on a desktop or tablet
          device.
        </p>
      </div>
      <div className="rounded-lg border bg-background shadow-lg overflow-hidden">
        <Image
          src="/images/demo.png"
          alt="DR Tulu Demo Screenshot"
          width={1200}
          height={800}
          className="w-full h-auto"
          priority
        />
      </div>
      <div className="mt-6 mb-6 text-center">
        <Link
          href={PAPER_URL}
          target="_blank"
          rel="noopener noreferrer"
          className={cn(
            buttonVariants({ variant: "default" }),
            "inline-flex items-center gap-2"
          )}
        >
          Read Our Paper
          <ExternalLink className="h-4 w-4" />
        </Link>
      </div>
    </div>
    <Footer />
  </div>
);

type ChatMode = "examples" | "live";

export default function Home() {
  const [chatMode, setChatMode] = useState<ChatMode>("examples");
  const [selectedExample, setSelectedExample] = useState<string>("");
  const [isPanelOpen, setIsPanelOpen] = useState<boolean>(true);
  const [isMobile, setIsMobile] = useState<boolean>(false);
  const [examplesList, setExamplesList] = useState<ExampleListItem[]>([]);
  const [isLoadingList, setIsLoadingList] = useState<boolean>(true);

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768);
    };

    checkMobile();
    window.addEventListener("resize", checkMobile);

    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  // Load available examples on mount
  useEffect(() => {
    const loadExamplesList = async () => {
      setIsLoadingList(true);
      const examples = await listExamples();
      setExamplesList(examples);

      // Set the first available example as selected
      if (examples.length > 0) {
        setSelectedExample(examples[0].json_file_name);
      }
      setIsLoadingList(false);
    };
    loadExamplesList();
  }, []);

  // Group examples by dataset
  const groupedExamples = examplesList.reduce((acc, example) => {
    if (!acc[example.dataset_name]) {
      acc[example.dataset_name] = [];
    }
    acc[example.dataset_name].push(example);
    return acc;
  }, {} as Record<string, ExampleListItem[]>);

  if (isMobile) {
    return <MobileView />;
  }

  return (
    <div className="min-h-screen flex flex-col">
      <div className="container relative p-16 flex-1">
        <Headline />
        <div className="mt-8 rounded-[0.5rem] border bg-background shadow overflow-hidden">
          <div className="px-0 pt-6 pb-0">
            <div className="mr-6 ml-8 flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold">
                DR Tulu for Deep Research
              </h2>
              <div className="flex items-center gap-3">
                {/* Mode Toggle */}
                <Tabs value={chatMode} onValueChange={(v) => setChatMode(v as ChatMode)}>
                  <TabsList className="h-9">
                    <TabsTrigger value="examples" className="text-xs px-3">
                      Examples
                    </TabsTrigger>
                    <TabsTrigger value="live" className="text-xs px-3">
                      Live Chat
                    </TabsTrigger>
                  </TabsList>
                </Tabs>

                {/* Example selector (only show in examples mode) */}
                {chatMode === "examples" && (
                  <>
                    <Separator orientation="vertical" className="h-6" />
                    <label className="text-sm font-medium">
                      See more randomly sampled examples:
                    </label>
                    <Select
                      value={selectedExample}
                      onValueChange={setSelectedExample}
                      disabled={isLoadingList || examplesList.length === 0}
                    >
                      <SelectTrigger className="w-96">
                        <SelectValue placeholder="Choose an example" />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(groupedExamples).map(
                          ([datasetName, examples]) => (
                            <React.Fragment key={datasetName}>
                              <div className="px-2 py-1.5 text-sm font-semibold text-muted-foreground">
                                {datasetName}
                              </div>
                              {examples.map((example) => (
                                <SelectItem
                                  key={example.json_file_name}
                                  value={example.json_file_name}
                                  className="pl-6"
                                >
                                  {example.example_title.slice(0, 60)}
                                  {example.example_title.length > 60 ? "..." : ""}
                                </SelectItem>
                              ))}
                            </React.Fragment>
                          )
                        )}
                      </SelectContent>
                    </Select>
                  </>
                )}
              </div>
            </div>
            <Separator className="mt-2" />
            
            {/* Conditional rendering based on mode */}
            {chatMode === "examples" && selectedExample && (
              <ChatInterface
                selectedExample={selectedExample}
                isPanelOpen={isPanelOpen}
                onPanelToggle={() => setIsPanelOpen(!isPanelOpen)}
              />
            )}
            {chatMode === "live" && (
              <LiveChatInterface
                isPanelOpen={isPanelOpen}
                onPanelToggle={() => setIsPanelOpen(!isPanelOpen)}
              />
            )}
          </div>
        </div>
      </div>
      <AuthorsSection />
      <Footer />
    </div>
  );
}
