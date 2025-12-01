/**
 * SSE client hook for live chat with DR-Tulu backend.
 */

import { useState, useCallback, useRef, useEffect } from "react";

// Message types from server
export type ServerMessageType =
  | "started"
  | "thinking"
  | "tool_call"
  | "answer"
  | "done"
  | "error";

export type DocumentData = {
  id: string;
  title: string;
  url: string;
  snippet: string;
};

export type ToolCallData = {
  tool_name: string;
  call_id: string | null;
  output: string;
  error: string | null;
  query?: string;
  params?: Record<string, any>;
  documents?: DocumentData[];
};

export type SnippetData = {
  id: string;
  title: string;
  url: string;
  snippet: string;
  tool_name: string;
};

export type ServerMessage = {
  type: ServerMessageType;
  content?: string;
  is_complete?: boolean;
  is_final?: boolean;
  segment_id?: number;
  tool_name?: string;
  call_id?: string;
  query?: string;
  params?: Record<string, any>;
  documents?: DocumentData[];
  output?: string;
  error?: string;
  message?: string;
  metadata?: {
    total_tool_calls: number;
    failed_tool_calls: number;
    browsed_links: string[];
    searched_links: string[];
    snippets: Record<string, SnippetData>;
  };
};

// Trace section for display
export type TraceItem =
  | { type: "thinking"; content: string; isComplete: boolean; segmentId: number }
  | { type: "tool_call"; data: ToolCallData }
  | { type: "answer"; content: string };

export type Message = {
  role: "user" | "assistant";
  content: string;
  traceItems?: TraceItem[];
};

export type ChatState = {
  isConnected: boolean;
  isLoading: boolean;
  error: string | null;
  thinkingContent: string;
  answerContent: string;
  traceItems: TraceItem[];
  metadata: ServerMessage["metadata"] | null;
  messages: Message[];
};

// Use window.location.origin when in browser to support accessing from different machines
// Falls back to localhost for SSR or when NEXT_PUBLIC_API_URL is set
const getDefaultApiUrl = () => {
  if (typeof window !== "undefined") {
    // In browser: use the same origin as the current page
    return window.location.origin;
  }
  // SSR or build time: use env var or localhost
  return process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
};

export function useChat(apiUrl?: string) {
  const resolvedApiUrl = apiUrl || getDefaultApiUrl();
  const abortControllerRef = useRef<AbortController | null>(null);
  const [isServerAvailable, setIsServerAvailable] = useState(false);

  const [state, setState] = useState<ChatState>({
    isConnected: false,
    isLoading: false,
    error: null,
    thinkingContent: "",
    answerContent: "",
    traceItems: [],
    metadata: null,
    messages: [],
  });

  // Check server availability
  const checkServer = useCallback(async () => {
    try {
      const response = await fetch(`${resolvedApiUrl}/health`);
      const data = await response.json();
      const available = data.status === "ok" && data.workflow_loaded;
      setIsServerAvailable(available);
      setState((prev) => ({ ...prev, isConnected: available }));
      return available;
    } catch {
      setIsServerAvailable(false);
      setState((prev) => ({ ...prev, isConnected: false }));
      return false;
    }
  }, [resolvedApiUrl]);

  // Handle incoming SSE message
  const handleMessage = useCallback((message: ServerMessage) => {
    switch (message.type) {
      case "started":
        // Query started processing
        break;

      case "thinking":
        setState((prev) => {
          const content = message.content || "";
          const segmentId = message.segment_id ?? 0;
          const isComplete = message.is_complete || false;

          // Find existing thinking item with same segment_id
          const existingIdx = prev.traceItems.findIndex(
            (item) => item.type === "thinking" && item.segmentId === segmentId
          );

          let newTraceItems = [...prev.traceItems];
          if (existingIdx >= 0) {
            // Update existing segment
            newTraceItems[existingIdx] = {
              type: "thinking",
              content,
              isComplete,
              segmentId,
            };
          } else {
            // Add new segment
            newTraceItems.push({
              type: "thinking",
              content,
              isComplete,
              segmentId,
            });
          }

          return {
            ...prev,
            thinkingContent: content,
            traceItems: newTraceItems,
          };
        });
        break;

      case "tool_call":
        setState((prev) => ({
          ...prev,
          traceItems: [
            ...prev.traceItems,
            {
              type: "tool_call",
              data: {
                tool_name: message.tool_name || "",
                call_id: message.call_id || null,
                output: message.output || "",
                error: message.error || null,
                query: message.query,
                params: message.params,
                documents: message.documents,
              },
            },
          ],
        }));
        break;

      case "answer":
        setState((prev) => {
          const content = message.content || "";
          if (message.is_final) {
            return {
              ...prev,
              answerContent: content,
              traceItems: [
                ...prev.traceItems,
                { type: "answer", content },
              ],
            };
          }
          return {
            ...prev,
            answerContent: content,
          };
        });
        break;

      case "done":
        setState((prev) => ({
          ...prev,
          isLoading: false,
          metadata: message.metadata || null,
        }));
        break;

      case "error":
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: message.message || "Unknown error",
        }));
        break;
    }
  }, []);

  // Send a query using SSE
  const sendQuery = useCallback(
    async (content: string, datasetName: string = "long_form") => {
      // Cancel any existing request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      // Calculate new messages list
      const newMessages = [...state.messages];
      if (state.answerContent || state.traceItems.length > 0) {
        newMessages.push({
          role: "assistant",
          content: state.answerContent,
          traceItems: state.traceItems,
        });
      }
      newMessages.push({ role: "user", content });

      // Update state for new query
      setState((prev) => ({
        ...prev,
        messages: newMessages,
        isLoading: true,
        error: null,
        thinkingContent: "",
        answerContent: "",
        traceItems: [],
        metadata: null,
      }));

      try {
        const response = await fetch(`${resolvedApiUrl}/chat/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            messages: newMessages.map((m) => ({
              role: m.role,
              content: m.content,
            })),
            dataset_name: datasetName,
          }),
          signal: abortController.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error: ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) {
          throw new Error("No response body");
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Process complete SSE events
          const lines = buffer.split("\n");
          buffer = lines.pop() || ""; // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              const jsonStr = line.slice(6);
              if (jsonStr.trim()) {
                try {
                  const message: ServerMessage = JSON.parse(jsonStr);
                  handleMessage(message);
                } catch (e) {
                  console.error("Failed to parse SSE message:", e, jsonStr);
                }
              }
            }
          }
        }

        // Process any remaining data in buffer
        if (buffer.startsWith("data: ")) {
          const jsonStr = buffer.slice(6);
          if (jsonStr.trim()) {
            try {
              const message: ServerMessage = JSON.parse(jsonStr);
              handleMessage(message);
            } catch (e) {
              console.error("Failed to parse final SSE message:", e);
            }
          }
        }

        return true;
      } catch (e) {
        if ((e as Error).name === "AbortError") {
          return false;
        }
        setState((prev) => ({
          ...prev,
          isLoading: false,
          error: (e as Error).message || "Failed to send query",
        }));
        return false;
      }
    },
    [
      resolvedApiUrl,
      handleMessage,
      state.messages,
      state.answerContent,
      state.traceItems,
    ]
  );

  // Cancel current request
  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setState((prev) => ({ ...prev, isLoading: false }));
    }
  }, []);

  // Load chat data from sessionStorage on mount
  useEffect(() => {
    const savedData = sessionStorage.getItem("chatData");
    if (savedData) {
      sessionStorage.removeItem("chatData");
      const data = JSON.parse(savedData);
      setState((prev) => ({
        ...prev,
        messages: data.messages || [],
      }));
    }
  }, []);

  // Check server on mount and periodically
  useEffect(() => {
    checkServer();
    const interval = setInterval(checkServer, 5000);
    return () => {
      clearInterval(interval);
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [checkServer]);

  return {
    ...state,
    isConnected: isServerAvailable,
    sendQuery,
    cancel,
    checkServer,
  };
}
