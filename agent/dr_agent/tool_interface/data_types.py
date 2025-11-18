import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Type alias for dictionary-based tool inputs
ToolInput = Dict[str, Any]


class ToolOutput(BaseModel):
    """Output from a tool execution"""

    tool_name: str
    output: str
    called: bool
    timeout: bool = False
    runtime: float = 0.0
    error: Optional[str] = None
    call_id: Optional[str] = None
    raw_output: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None


class Document(BaseModel):
    """A document with metadata and content"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    url: str
    snippet: Optional[str] = None
    text: Optional[str] = None
    summary: Optional[str] = None
    score: Optional[float] = None
    error: Optional[str] = None

    def simple_stringify(self, prioritize_summary: bool = False) -> str:
        """
        Format document as a simple string with title, URL, and snippet.
        """
        content_components = []

        if self.title:
            content_components.append(f"Title: {self.title}")
        if self.url:
            content_components.append(f"URL: {self.url}")
        if self.snippet:
            content_components.append(f"Search Snippet: {self.snippet}")

        if not prioritize_summary or not self.summary:
            if self.text:
                full_text = f"Full Text: {self.text[:2000]}"
                if len(self.text) > 2000:
                    content_components.append("...")
                content_components.append(full_text)

        if self.summary:
            content_components.append(f"Summary: {self.summary}")

        return "\n".join(content_components)

    def stringify(
        self,
        webpage_title: Optional[str] = None,
        use_localized_snippets: bool = True,
        context_chars: int = 2000,
        fallback_message: Optional[str] = None,
    ) -> str:
        """
        Format document as webpage content with title, URL, and snippet.
        Optionally enriches with webpage content and localizes snippets.

        Args:
            raw_content: Full content extracted from webpage, if available
            webpage_title: Title extracted from webpage, if available
            use_localized_snippets: Whether to localize original snippet in content
            context_chars: Number of characters to include around localized snippets
            fallback_message: Optional message to append (e.g., error notes)

        Returns:
            Formatted webpage content string
        """
        # Determine final title: webpage title > original title > default
        final_title = webpage_title or self.title or "No title available"

        content_parts = [f"Title: {final_title}"]

        if self.url:
            content_parts.append(f"URL: {self.url}")

        # Determine snippet content
        localized_snippet = None

        if not self.text:
            # No webpage content available, use original snippet
            localized_snippet = self.snippet
        else:
            if use_localized_snippets and self.snippet:
                # Try to localize the original snippet in the webpage content
                from .utils import extract_snippet_with_context

                success, localized_content = extract_snippet_with_context(
                    self.text, self.snippet, context_chars=context_chars
                )
                if success:
                    localized_snippet = localized_content
                else:
                    # Fallback to original snippet if localization fails
                    localized_snippet = self.snippet
            elif self.snippet and not use_localized_snippets:
                # Use original snippet without localization
                localized_snippet = self.snippet
            else:
                # No original snippet or not using localization, use first part of webpage content
                truncated_content = self.text[:context_chars].strip()
                if len(self.text) > context_chars:
                    truncated_content += "..."
                localized_snippet = truncated_content

        # Use localized snippet or fallback to original snippet or default
        snippet_content = localized_snippet or self.snippet or "No content available"
        content_parts.append(f"Snippet: {snippet_content}")

        # Add fallback message if provided
        if fallback_message:
            content_parts.append(fallback_message)

        return "\n".join(content_parts)


class DocumentToolOutput(ToolOutput):
    """Enhanced ToolOutput that includes structured document information"""

    documents: List[Document] = []
    query: Optional[str] = None  # Original query that generated these documents
