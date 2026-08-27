from dr_agent.citations import (
    build_snippet_index,
    collect_sources,
    format_answer_with_references,
    format_bibliography,
    make_snippet_id,
    parse_citations,
    parse_snippets_from_trace,
    resolve_citations,
    strip_citation_tags,
)
from dr_agent.tool_interface.data_types import Document, DocumentToolOutput, ToolOutput


def make_document(index: int) -> Document:
    return Document(
        title=f"Paper {index}",
        url=f"https://example.org/{index}",
        snippet=f"Finding number {index}.",
    )


def make_search_output(call_id: str, count: int = 2) -> DocumentToolOutput:
    return DocumentToolOutput(
        tool_name="snippet_search",
        output="",
        called=True,
        call_id=call_id,
        documents=[make_document(i) for i in range(count)],
    )


class TestParseCitations:
    def test_parses_single_quoted_id(self):
        citations = parse_citations('<cite id="a1-0">Solar grew fast.</cite>')

        assert len(citations) == 1
        assert citations[0].ids == ["a1-0"]
        assert citations[0].text == "Solar grew fast."

    def test_accepts_the_attribute_spellings_the_prompts_use(self):
        # The shipped prompts show both `id=` and `ids=`, quoted and bare.
        text = (
            '<cite ids="a1-0,a1-1">first</cite> '
            "<cite id=a1-2>second</cite> "
            "<cite ids='a1-3'>third</cite>"
        )

        citations = parse_citations(text)

        assert [citation.ids for citation in citations] == [
            ["a1-0", "a1-1"],
            ["a1-2"],
            ["a1-3"],
        ]
        assert [citation.text for citation in citations] == [
            "first",
            "second",
            "third",
        ]

    def test_splits_ids_on_commas_and_whitespace(self):
        citations = parse_citations('<cite ids="a1-0, a1-1 a1-2">claim</cite>')

        assert citations[0].ids == ["a1-0", "a1-1", "a1-2"]

    def test_spans_may_cover_multiple_lines(self):
        citations = parse_citations('<cite id="a1-0">first line\nsecond line</cite>')

        assert citations[0].text == "first line\nsecond line"

    def test_reports_unclosed_tag_from_truncated_generation(self):
        citations = parse_citations('done <cite id="a1-0">cut off')

        assert len(citations) == 1
        assert citations[0].ids == ["a1-0"]
        assert citations[0].text == ""

    def test_unclosed_tags_can_be_ignored(self):
        assert parse_citations('<cite id="a1-0">cut off', include_unclosed=False) == []

    def test_offsets_point_at_the_whole_tag(self):
        text = 'lead in <cite id="a1-0">claim</cite> tail'

        citation = parse_citations(text)[0]

        assert text[citation.start : citation.end] == '<cite id="a1-0">claim</cite>'

    def test_ignores_text_without_citations(self):
        assert parse_citations("no citations here") == []
        assert parse_citations("") == []


class TestBuildSnippetIndex:
    def test_indexes_documents_by_call_id_and_position(self):
        result = {"tool_calls": [make_search_output("a1", count=2)]}

        index = build_snippet_index(result)

        assert set(index) == {"a1-0", "a1-1"}
        assert index["a1-1"].url == "https://example.org/1"

    def test_accepts_a_serialized_result(self):
        result = {"tool_calls": [make_search_output("a1").model_dump()]}

        index = build_snippet_index(result)

        assert index["a1-0"].title == "Paper 0"

    def test_accepts_a_bare_list_of_tool_calls(self):
        index = build_snippet_index([make_search_output("a1"), make_search_output("b2")])

        assert set(index) == {"a1-0", "a1-1", "b2-0", "b2-1"}

    def test_includes_documents_retrieved_by_sub_agents(self):
        browse_agent_result = {"tool_calls": [make_search_output("inner", count=1)]}
        outer = {
            "tool_calls": [
                make_search_output("outer", count=1),
                ToolOutput(
                    tool_name="browse_agent",
                    output="summary",
                    called=True,
                    call_id="agent-call",
                    raw_output=browse_agent_result,
                ),
            ]
        }

        index = build_snippet_index(outer)

        assert set(index) == {"outer-0", "inner-0"}

    def test_nested_documents_can_be_skipped(self):
        outer = {
            "tool_calls": [
                ToolOutput(
                    tool_name="browse_agent",
                    output="summary",
                    called=True,
                    call_id="agent-call",
                    raw_output={"tool_calls": [make_search_output("inner", count=1)]},
                )
            ]
        }

        assert build_snippet_index(outer, include_nested=False) == {}

    def test_falls_back_to_document_id_without_a_call_id(self):
        output = DocumentToolOutput(
            tool_name="snippet_search",
            output="",
            called=True,
            documents=[make_document(0)],
        )

        index = build_snippet_index({"tool_calls": [output]})

        assert list(index) == [output.documents[0].id]

    def test_tool_calls_without_documents_are_skipped(self):
        result = {"tool_calls": [ToolOutput(tool_name="noop", output="", called=True)]}

        assert build_snippet_index(result) == {}


class TestParseSnippetsFromTrace:
    def test_recovers_documents_from_a_saved_trace(self):
        trace = (
            "<snippet id=a1-0>\n"
            "Title: Solar Report\n"
            "URL: https://example.org/solar\n"
            "Snippet: Capacity doubled.\n"
            "</snippet>\n"
            "<webpage id=b2-0>\n"
            "Title: Wind Report\n"
            "URL: https://example.org/wind\n"
            "Snippet: Offshore capacity rose.\n"
            "</webpage>"
        )

        index = parse_snippets_from_trace(trace)

        assert index["a1-0"].url == "https://example.org/solar"
        assert index["a1-0"].title == "Solar Report"
        assert index["b2-0"].snippet == "Offshore capacity rose."

    def test_keeps_multi_line_snippet_bodies(self):
        trace = (
            "<snippet id=a1-0>\n"
            "Title: T\n"
            "URL: https://example.org/t\n"
            "Snippet: first\nsecond\n"
            "</snippet>"
        )

        assert parse_snippets_from_trace(trace)["a1-0"].snippet == "first\nsecond"

    def test_ids_line_up_with_the_ids_tools_emit(self):
        document = make_document(0)
        trace = f"<snippet id={make_snippet_id('a1', 0)}>\n{document.stringify()}\n</snippet>"

        index = parse_snippets_from_trace(trace)

        assert index["a1-0"].url == document.url

    def test_empty_trace_gives_an_empty_index(self):
        assert parse_snippets_from_trace("") == {}


class TestResolveCitations:
    def test_resolves_ids_to_documents(self):
        index = build_snippet_index({"tool_calls": [make_search_output("a1")]})

        citations = resolve_citations('<cite ids="a1-0,a1-1">claim</cite>', index)

        assert citations[0].urls == [
            "https://example.org/0",
            "https://example.org/1",
        ]
        assert citations[0].is_resolved

    def test_reports_ids_that_are_not_in_the_index(self):
        index = build_snippet_index({"tool_calls": [make_search_output("a1", count=1)]})

        citation = resolve_citations('<cite ids="a1-0,made-up">claim</cite>', index)[0]

        assert citation.unresolved_ids == ["made-up"]
        assert not citation.is_resolved
        assert [doc.url for doc in citation.documents] == ["https://example.org/0"]

    def test_urls_are_deduplicated_within_a_citation(self):
        document = make_document(0)
        output = DocumentToolOutput(
            tool_name="snippet_search",
            output="",
            called=True,
            call_id="a1",
            documents=[document, document.model_copy()],
        )

        citation = resolve_citations(
            '<cite ids="a1-0,a1-1">claim</cite>',
            build_snippet_index({"tool_calls": [output]}),
        )[0]

        assert citation.urls == ["https://example.org/0"]


class TestSourcesAndRendering:
    def test_collect_sources_deduplicates_in_citation_order(self):
        index = build_snippet_index({"tool_calls": [make_search_output("a1")]})
        citations = resolve_citations(
            '<cite id="a1-1">one</cite> <cite ids="a1-0,a1-1">two</cite>', index
        )

        assert [doc.url for doc in collect_sources(citations)] == [
            "https://example.org/1",
            "https://example.org/0",
        ]

    def test_format_bibliography_numbers_from_one(self):
        bibliography = format_bibliography([make_document(0), make_document(1)])

        assert bibliography == (
            "[1] Paper 0. https://example.org/0\n"
            "[2] Paper 1. https://example.org/1"
        )

    def test_answer_gets_inline_markers_and_a_reference_list(self):
        index = build_snippet_index({"tool_calls": [make_search_output("a1")]})

        rendered = format_answer_with_references(
            'Solar grew. <cite id="a1-0">Capacity doubled.</cite>', index
        )

        assert rendered == (
            "Solar grew. Capacity doubled.[1]\n\n"
            "References\n"
            "[1] Paper 0. https://example.org/0"
        )

    def test_unknown_ids_are_marked_rather_than_dropped(self):
        index = build_snippet_index({"tool_calls": [make_search_output("a1", count=1)]})

        rendered = format_answer_with_references(
            '<cite id="a1-0">real</cite> <cite id="ghost">invented</cite>', index
        )

        assert rendered.startswith("real[1] invented[?]")

    def test_answer_without_citations_is_returned_unchanged(self):
        assert format_answer_with_references("plain answer", {}) == "plain answer"

    def test_strip_citation_tags_keeps_the_prose(self):
        text = 'Solar grew. <cite ids="a1-0,a1-1">Capacity doubled.</cite> Done.'

        assert strip_citation_tags(text) == "Solar grew. Capacity doubled. Done."

    def test_strip_citation_tags_handles_a_truncated_tag(self):
        assert strip_citation_tags('Solar grew. <cite id="a1-0">cut') == (
            "Solar grew. cut"
        )
