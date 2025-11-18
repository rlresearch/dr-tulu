from typing import Dict, Generic, List, Optional, Set, Tuple, TypeVar

from pydantic import BaseModel
from typing_extensions import TypedDict


# Semantic Scholar API Data
class SemanticScholarAuthorData(BaseModel):
    name: str
    authorId: Optional[str] = None
    affiliations: Optional[List[str]] = None
    homepage: Optional[str] = None
    paperCount: Optional[int] = None
    citationCount: Optional[int] = None
    hIndex: Optional[int] = None
    externalIds: Optional[dict] = None
    url: Optional[str] = None


class OpenAccessPdf(BaseModel):
    url: str
    status: Optional[str] = None


class SemanticScholarPaperData(BaseModel):
    paperId: str
    corpusId: int
    url: str
    title: Optional[str] = None
    venue: Optional[str] = None
    publicationVenue: Optional[str] = None
    year: Optional[int] = None
    authors: List[SemanticScholarAuthorData] = []
    externalIds: Optional[dict] = None
    abstract: Optional[str] = None
    referenceCount: Optional[int] = None
    citationCount: Optional[int] = None
    influentialCitationCount: Optional[int] = None
    isOpenAccess: Optional[bool] = None
    openAccessPdf: Optional[OpenAccessPdf] = None
    fieldsOfStudy: Optional[List[str]] = None
    s2FieldsOfStudy: Optional[List[str]] = None
    publicationTypes: Optional[List[str]] = None
    publicationDate: Optional[str] = None  # YYYY-MM-DD if available
    journal: Optional[str] = None
    citationStyles: Optional[List[str]] = None
    embedding: Optional[str] = None  # e.g., embedding.specter_v1


class BulkSemanticSearchResult(TypedDict):
    id: int
    status: str
    s2_search_status: Optional[Dict]
    s2_search_results: List[SemanticScholarPaperData]
    s2_search_arguments: Optional[Dict]
    original_data: Dict
    error: Optional[str]


# Semantic Scholar Snippet Search Data


class SentenceAnnotation(BaseModel):
    start: int
    end: int
    matchedPaperCorpusId: Optional[str] = None


class Annotations(BaseModel):
    sentences: List[SentenceAnnotation]
    refMentions: Optional[List[SentenceAnnotation]] = None


class SnippetOffset(BaseModel):
    start: int
    end: int


class SnippetInfo(BaseModel):
    text: str
    snippetKind: str
    section: str
    snippetOffset: SnippetOffset
    annotations: Annotations


class OpenAccessInfo(BaseModel):
    license: str
    status: str
    disclaimer: str


class PaperInfo(BaseModel):
    corpusId: str
    title: str
    authors: List[str]
    openAccessInfo: OpenAccessInfo


class PaperSnippet(BaseModel):
    snippet: SnippetInfo
    score: float
    paper: PaperInfo


# Crawl4AI API Data
class Crawl4aiApiResult(BaseModel):
    """Result structure returned by Crawl4AI Docker API."""

    url: str
    success: bool
    markdown: str
    fit_markdown: Optional[str] = None
    html: Optional[str] = None
    error: Optional[str] = None
