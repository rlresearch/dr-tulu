"""
Resolve the citation ids in a generated answer back to their source documents.

Search and browse tools label every document they return with an id of the form
``{call_id}-{index}`` (see :func:`make_snippet_id`) and surface it to the model
as ``<snippet id=...>`` / ``<webpage id=...>``.  The model then grounds its
claims with ``<cite id="...">...</cite>`` tags.  The ids alone are opaque, so
this module turns that convention into a small API for getting titles and URLs
back out of a run:

    from dr_agent.citations import build_snippet_index, resolve_citations

    result = await workflow(problem="...")
    index = build_snippet_index(result)
    for citation in resolve_citations(result.generated_text, index):
        print(citation.text, [doc.url for doc in citation.documents])

If all you kept was the serialized trace (for example the ``generated_text``
field of an evaluation ``.jsonl``), build the index from the text instead:

    index = parse_snippets_from_trace(trace_text)

Nothing here talks to a model or a tool server, so it is safe to use as a plain
post-processing step over saved runs.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from .tool_interface.data_types import (
    SNIPPET_ID_TEMPLATE,
    Document,
    make_snippet_id,
)

__all__ = [
    "SNIPPET_ID_TEMPLATE",
    "Citation",
    "ResolvedCitation",
    "make_snippet_id",
    "parse_citations",
    "parse_snippets_from_trace",
    "build_snippet_index",
    "resolve_citations",
    "collect_sources",
    "format_bibliography",
    "format_answer_with_references",
    "strip_citation_tags",
]


# <cite id="A,B">claim</cite>, <cite ids='A'>claim</cite> and <cite id=A>claim</cite>
# are all produced in practice, so accept the attribute both ways, quoted or not.
_CITE_TAG = r"<cite\s+ids?\s*=\s*(?P<quote>[\"']?)(?P<ids>[^\"'>]+?)(?P=quote)\s*/?>"
CITE_PATTERN = re.compile(
    _CITE_TAG + r"(?P<text>.*?)</cite>", re.DOTALL | re.IGNORECASE
)
CITE_OPEN_TAG_PATTERN = re.compile(_CITE_TAG, re.IGNORECASE)

#: Tool results are wrapped in these tags inside the generation trace.
TRACE_SNIPPET_PATTERN = re.compile(
    r"<(?P<tag>snippet|webpage)\s+id\s*=\s*[\"']?(?P<id>[^\"'>\s]+)[\"']?\s*>"
    r"(?P<body>.*?)</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)

_ID_SEPARATORS = re.compile(r"[,;\s]+")


def _split_ids(raw_ids: str) -> List[str]:
    """Split a cite attribute (``"A,B"``, ``"A B"``, ``"A"``) into ids."""
    return [part for part in _ID_SEPARATORS.split(raw_ids.strip()) if part]


class Citation(BaseModel):
    """A single ``<cite>`` span found in generated text."""

    ids: List[str]
    text: str
    start: int
    end: int


class ResolvedCitation(Citation):
    """A citation whose ids have been looked up in a snippet index."""

    documents: List[Document] = Field(default_factory=list)
    unresolved_ids: List[str] = Field(default_factory=list)

    @property
    def is_resolved(self) -> bool:
        """True when every cited id was found in the index."""
        return not self.unresolved_ids

    @property
    def urls(self) -> List[str]:
        """URLs of the resolved sources, in citation order, without duplicates."""
        seen: List[str] = []
        for doc in self.documents:
            if doc.url and doc.url not in seen:
                seen.append(doc.url)
        return seen


def parse_citations(text: str, *, include_unclosed: bool = True) -> List[Citation]:
    """
    Extract every ``<cite>`` span from ``text``.

    Args:
        text: Generated text, either the full trace or just the answer.
        include_unclosed: Also report opening tags with no matching ``</cite>``,
            which happens when a generation is truncated mid-answer.  Their
            ``text`` is empty.

    Returns:
        Citations in order of appearance.
    """
    if not text:
        return []

    citations = [
        Citation(
            ids=_split_ids(match.group("ids")),
            text=match.group("text").strip(),
            start=match.start(),
            end=match.end(),
        )
        for match in CITE_PATTERN.finditer(text)
    ]

    if include_unclosed:
        closed_starts = {citation.start for citation in citations}
        for match in CITE_OPEN_TAG_PATTERN.finditer(text):
            if match.start() in closed_starts:
                continue
            citations.append(
                Citation(
                    ids=_split_ids(match.group("ids")),
                    text="",
                    start=match.start(),
                    end=match.end(),
                )
            )
        citations.sort(key=lambda citation: citation.start)

    return [citation for citation in citations if citation.ids]


def _parse_document_block(body: str) -> Document:
    """Rebuild a :class:`Document` from the text a tool wrote into the trace."""
    fields: Dict[str, List[str]] = {}
    current: Optional[str] = None

    for line in body.strip().splitlines():
        header = re.match(r"\s*(Title|URL|Snippet|Search Snippet|Full Text|Summary)\s*:\s*(.*)", line)
        if header:
            current = header.group(1).lower()
            fields.setdefault(current, []).append(header.group(2))
        elif current:
            fields[current].append(line)

    def joined(key: str) -> Optional[str]:
        if key not in fields:
            return None
        return "\n".join(fields[key]).strip() or None

    snippet = joined("snippet") or joined("search snippet")
    return Document(
        title=joined("title") or "",
        url=joined("url") or "",
        snippet=snippet if snippet is not None else body.strip(),
        text=joined("full text"),
        summary=joined("summary"),
    )


def parse_snippets_from_trace(text: str) -> Dict[str, Document]:
    """
    Recover an id -> document index from a serialized generation trace.

    Use this when the tool outputs are no longer available as objects and only
    the text of the run was kept.  Both ``<snippet>`` and ``<webpage>`` blocks
    are indexed; later blocks win if an id somehow repeats.
    """
    if not text:
        return {}
    return {
        match.group("id"): _parse_document_block(match.group("body"))
        for match in TRACE_SNIPPET_PATTERN.finditer(text)
    }


def _as_mapping(obj: Any) -> Optional[Dict[str, Any]]:
    """View a tool output as a dict, whether it is a model or already a dict."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, BaseModel):
        return {name: getattr(obj, name) for name in type(obj).model_fields}
    return None


def _iter_tool_calls(source: Any) -> Iterable[Any]:
    """Yield the tool outputs carried by a workflow result, model or dict."""
    if source is None:
        return []
    if isinstance(source, (list, tuple)):
        return list(source)
    mapping = _as_mapping(source)
    if mapping is not None and "tool_calls" in mapping:
        return list(mapping.get("tool_calls") or [])
    if mapping is not None and "documents" in mapping:
        return [source]
    return []


def build_snippet_index(
    source: Any,
    *,
    include_nested: bool = True,
    _depth: int = 0,
) -> Dict[str, Document]:
    """
    Build the id -> document index for a run.

    Args:
        source: A ``GenerateWithToolsOutput`` (or its ``model_dump()``), a list
            of tool outputs, or a single tool output.
        include_nested: Also index documents retrieved by sub-agents, which
            store their own result under ``raw_output``.  Sub-agent ids are
            unique, so they share one flat namespace with the outer run.

    Returns:
        A mapping from the ids used in ``<cite>`` tags to their documents.
    """
    index: Dict[str, Document] = {}
    if _depth > 8:  # sub-agents can nest, but not this deeply
        return index

    for tool_call in _iter_tool_calls(source):
        mapping = _as_mapping(tool_call)
        if mapping is None:
            continue

        call_id = mapping.get("call_id")
        for position, document in enumerate(mapping.get("documents") or []):
            if isinstance(document, dict):
                document = Document(**document)
            snippet_id = make_snippet_id(call_id, position) if call_id else document.id
            index[snippet_id] = document

        if include_nested:
            nested = mapping.get("raw_output")
            if isinstance(nested, dict) and "tool_calls" in nested:
                index.update(
                    build_snippet_index(
                        nested, include_nested=True, _depth=_depth + 1
                    )
                )

    return index


def resolve_citations(
    text: str, index: Dict[str, Document], *, include_unclosed: bool = True
) -> List[ResolvedCitation]:
    """
    Parse the ``<cite>`` tags in ``text`` and look their ids up in ``index``.

    Ids with no entry are reported in ``unresolved_ids`` rather than dropped:
    an off-the-shelf model can invent ids it never saw, and knowing which
    claims rest on them is the point of checking.
    """
    resolved = []
    for citation in parse_citations(text, include_unclosed=include_unclosed):
        documents = []
        unresolved = []
        for snippet_id in citation.ids:
            document = index.get(snippet_id)
            if document is None:
                unresolved.append(snippet_id)
            else:
                documents.append(document)
        resolved.append(
            ResolvedCitation(
                **citation.model_dump(), documents=documents, unresolved_ids=unresolved
            )
        )
    return resolved


def collect_sources(citations: Sequence[ResolvedCitation]) -> List[Document]:
    """Deduplicate the cited documents, in order of first citation."""
    sources: List[Document] = []
    seen = set()
    for citation in citations:
        for document in citation.documents:
            key = document.url or document.id
            if key in seen:
                continue
            seen.add(key)
            sources.append(document)
    return sources


def format_bibliography(sources: Sequence[Document], *, start: int = 1) -> str:
    """Render numbered references, one per line, as ``[n] Title. URL``."""
    lines = []
    for number, document in enumerate(sources, start=start):
        parts = [f"[{number}]"]
        if document.title:
            parts.append(f"{document.title}.")
        if document.url:
            parts.append(document.url)
        lines.append(" ".join(parts))
    return "\n".join(lines)


def format_answer_with_references(
    text: str,
    index: Dict[str, Document],
    *,
    heading: str = "References",
) -> str:
    """
    Rewrite an answer for human consumption.

    Each ``<cite>`` tag becomes its claim followed by numbered markers, and the
    matching sources are appended under ``heading`` with their URLs.  Ids that
    are not in ``index`` are marked ``[?]`` so hallucinated citations stay
    visible instead of silently disappearing.
    """
    citations = resolve_citations(text, index, include_unclosed=False)
    sources = collect_sources(citations)
    numbers = {
        (document.url or document.id): position
        for position, document in enumerate(sources, start=1)
    }

    pieces: List[str] = []
    cursor = 0
    for citation in citations:
        markers = [f"[{numbers[doc.url or doc.id]}]" for doc in citation.documents]
        markers += ["[?]"] * len(citation.unresolved_ids)
        pieces.append(text[cursor : citation.start])
        pieces.append(citation.text + "".join(markers))
        cursor = citation.end
    pieces.append(text[cursor:])

    body = "".join(pieces)
    if not sources:
        return body
    return f"{body}\n\n{heading}\n{format_bibliography(sources)}"


def strip_citation_tags(text: str) -> str:
    """Remove ``<cite>`` markup, keeping the cited text itself."""
    without_spans = CITE_PATTERN.sub(lambda match: match.group("text"), text)
    without_orphans = CITE_OPEN_TAG_PATTERN.sub("", without_spans)
    return re.sub(r"</cite>", "", without_orphans, flags=re.IGNORECASE)
