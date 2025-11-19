import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import dotenv

from dr_agent.agent_interface import BaseAgent
from dr_agent.client import DocumentToolOutput, LLMToolClient, ToolOutput
from dr_agent.shared_prompts import UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS
from dr_agent.tool_interface.chained_tool import ChainedTool
from dr_agent.tool_interface.mcp_tools import (
    BaseTool,
    Crawl4AIBrowseTool,
    SemanticScholarSnippetSearchTool,
    SerperBrowseTool,
    SerperSearchTool,
    JinaBrowseTool,
)
from dr_agent.workflow import BaseWorkflow, BaseWorkflowConfiguration

# Make sure the .env file is in the root directory of the project rl-rag-mcp/.env
dotenv.load_dotenv(Path(__file__).parent.parent.parent / ".env")


@dataclass
class WebPageReaderAgentV2(BaseAgent):
    question: Optional[str] = None
    prompt = """
We are searching on the internet for the following question:
{question}
Here is some webpage scraped from the internet:
{document}
Can you clean the raw webpage text and convert it into a more readable format? You should remove all the unnecessary information and keep the main content of the page. Please produce the output in the format of "Cleaned webpage text:\n[you text here]".
""".strip()

    def preprocess_input(self, documents: Union[str, Any]) -> Dict[str, str]:
        # Accept either a raw string or a ToolOutput-like object with an `output` attribute
        assert self.question is not None, "Question is not set"

        if isinstance(documents, DocumentToolOutput):
            # print("using DocumentToolOutput")
            doc_str = "\n".join(
                [
                    document.simple_stringify()[: 32000 * 4 // len(documents.documents)]
                    for document in documents.documents
                ]
            )
        elif hasattr(documents, "output"):
            doc_str = documents.output
        else:
            doc_str = documents if isinstance(documents, str) else str(documents)
        input_params = {"question": self.question, "document": doc_str}
        # print(input_params)
        return input_params

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "Cleaned webpage text:" in output_string:
            output_string = output_string.split("Cleaned webpage text:")[1].strip()

        return output_string


@dataclass
class SearchAgent(BaseAgent):
    def prompt(
        self,
        question: str,
        dataset_name: Optional[str] = None,
    ) -> str:

        PROMPT = UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS["v20250907"]
        if dataset_name in [
            "2wiki",
            "simpleqa",
            "browsecomp",
            "bc_synthetic_depth_one_v2_verified",
            "bc_synthetic_varied_depth_o3_verified",
            "webwalker",
            "hle",
        ]:
            instruction_field_name = "exact_answer"
        elif dataset_name in ["sqav2"]:
            instruction_field_name = "long_form"
        elif dataset_name in ["healthbench", "deep_research_bench", "researchqa"]:
            instruction_field_name = "short_form"
        elif "sft-mix" in dataset_name:
            if "short_form" in dataset_name:
                instruction_field_name = "exact_answer"
            elif "long_form" in dataset_name:
                instruction_field_name = "long_form"  # or "short_form"?
            else:
                raise ValueError(
                    f"Unclear which instruction field name to use for the sft mix dataset: {dataset_name}"
                )
        else:
            if "short_form" in str(dataset_name):
                instruction_field_name = "exact_answer"
            elif "long_form" in str(dataset_name):
                instruction_field_name = "long_form"
            else:
                print("set additional instructions none")
                instruction_field_name = None

        return [
            {
                "role": "system",
                "content": PROMPT["system_prompt"],
            },
            {
                "role": "user",
                "content": (
                    question
                    + "\n\n"
                    + PROMPT["additional_instructions"][instruction_field_name]
                    if instruction_field_name is not None
                    else question
                ),
            },
        ]

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "<answer>" in output_string:
            output_string = (
                output_string.split("<answer>")[1].split("</answer>")[0].strip()
            )

        # Replace the "\boxed{" with "\\boxed{"
        output_string = output_string.replace("\boxed{", "\\boxed{")

        if "\\boxed{" in output_string:
            output_string = output_string.split("\\boxed{")[1].split("}")[0].strip()

        return output_string


@dataclass
class AnswerAgent(BaseAgent):

    def prompt(self, question: str, history: str, dataset_name: str) -> str:

        PROMPT = UNIFIED_TOOL_CALLING_STRUCTURED_PROMPTS["v20250907"]
        if dataset_name in [
            "2wiki",
            "simpleqa",
            "browsecomp",
            "bc_synthetic_depth_one_v2_verified",
            "bc_synthetic_varied_depth_o3_verified",
            "webwalker",
        ]:
            instruction_field_name = "exact_answer"
        elif dataset_name in ["sqav2"]:
            instruction_field_name = "long_form"
        elif dataset_name in ["healthbench", "deep_research_bench", "researchqa"]:
            instruction_field_name = "short_form"
        else:
            raise ValueError(f"Invalid dataset name: {dataset_name}")

        return [
            {
                "role": "system",
                "content": PROMPT["system_prompt"],
            },
            {
                "role": "user",
                "content": question
                + "\n\n"
                + PROMPT["additional_instructions"][instruction_field_name],
            },
            {
                "role": "assistant",
                "content": history,
            },
            {
                "role": "user",
                "content": "Now please generate an answer based on the search results by far.",
            },
        ]

    def postprocess_output(self, result: Dict[str, Any]) -> str:
        output_string = result.generated_text
        if "</think>" in output_string:
            output_string = "".join(output_string.split("</think>")[1:]).strip()

        if "<answer>" in output_string:
            output_string = (
                output_string.split("<answer>")[1].split("</answer>")[0].strip()
            )

        # Replace the "\boxed{" with "\\boxed{"
        output_string = output_string.replace("\boxed{", "\\boxed{")

        if "\\boxed{" in output_string:
            output_string = output_string.split("\\boxed{")[1].split("}")[0].strip()

        return output_string


class NoBrowseTool(BaseTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def __call__(self, *args, **kwargs):
        return DocumentToolOutput(
            output="Browse tool is not available at this time. Please try other tools.",
            called=True,
            timeout=False,
            runtime=0.0,
            error=None,
            call_id=self._generate_call_id(),
            raw_output=None,
            documents=[],
            tool_name="no_browse",
        )

    def _format_output(self, output: ToolOutput) -> str:
        return output.output

    def _generate_tool_schema(self):
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "URL to browse"}},
            "required": ["url"],
        }


class AutoReasonSearchWorkflow(BaseWorkflow):
    _default_configuration_path = os.path.join(
        os.path.dirname(__file__), "auto_search.yaml"
    )

    class Configuration(BaseWorkflowConfiguration):

        tool_parser: str

        search_tool_name: str = "serper"

        # Separate generation client (SFT model)
        search_agent_base_url: Optional[str] = None
        search_agent_model_name: str = (
            "dr-tulu/DR-Tulu-8B"
        )
        search_agent_tokenizer_name: str = "Qwen/Qwen3-8B"
        search_agent_api_key: str = "dummy-key"
        search_agent_max_tokens: int = 32000
        search_agent_temperature: float = 0.7
        search_agent_max_tool_calls: int = 10

        use_browse_agent: bool = False
        browse_agent_base_url: Optional[str] = None
        browse_agent_model_name: str = "Qwen/Qwen3-8B"
        browse_agent_tokenizer_name: str = "Qwen/Qwen3-8B"
        browse_agent_api_key: str = "dummy-key"
        browse_agent_max_tokens: int = 32000
        browse_agent_temperature: float = 0.3

        # Search configuration
        number_documents_to_search: int = 10
        search_timeout: int = 60

        # Browse configuration
        browse_tool_name: Optional[str] = "crawl4ai"
        browse_timeout: int = 60
        browse_max_pages_to_fetch: int = 10
        browse_context_char_length: int = 6000
        crawl4ai_use_docker_version: bool = False
        crawl4ai_use_ai2_config: bool = False

    def setup_components(
        self,
        mcp_transport_type: Optional[str] = "StreamableHttpTransport",
        mcp_executable: Optional[str] = None,
        mcp_port: Optional[int] = 8000,
    ) -> None:
        cfg = self.configuration
        assert cfg is not None
        # print(cfg)

        # Search and browse tools (MCP-backed) with unified tool parser
        if cfg.search_tool_name == "serper":
            self.search_tool = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",  # <- to test this v20250824 model, we need to set the tool name in a hacky way.
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.search_tool_name == "s2":
            self.search_tool = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SerperSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.search_tool_name == "s2-only":
            self.search_tool = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="snippet_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )

            self.search_tool2 = SemanticScholarSnippetSearchTool(
                tool_parser=cfg.tool_parser,
                number_documents_to_search=cfg.number_documents_to_search,
                timeout=cfg.search_timeout,
                name="google_search",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        else:
            raise ValueError(f"Invalid search tool name: {cfg.search_tool_name}")

        if cfg.browse_tool_name == "serper":
            self.browse_tool = SerperBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.browse_tool_name == "crawl4ai":
            self.browse_tool = Crawl4AIBrowseTool(
                tool_parser=cfg.tool_parser,
                max_pages_to_fetch=cfg.browse_max_pages_to_fetch,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
                context_chars=cfg.browse_context_char_length,
                use_docker_version=cfg.crawl4ai_use_docker_version,
                use_ai2_config=cfg.crawl4ai_use_ai2_config,
            )
        elif cfg.browse_tool_name == "jina":
            self.browse_tool = JinaBrowseTool(
                tool_parser=cfg.tool_parser,
                timeout=cfg.browse_timeout,
                name="browse_webpage",
                transport_type=mcp_transport_type,
                mcp_executable=mcp_executable,
                mcp_port=mcp_port,
            )
        elif cfg.browse_tool_name is None:
            self.browse_tool = NoBrowseTool(
                tool_parser=cfg.tool_parser,
                name="browse_webpage",
            )
        else:
            raise ValueError(f"Invalid browse tool name: {cfg.browse_tool_name}")
        print("Using browse tool: ", self.browse_tool)

        if cfg.use_browse_agent:
            with LLMToolClient(
                model_name=cfg.browse_agent_model_name,
                tokenizer_name=cfg.browse_agent_tokenizer_name,
                base_url=cfg.browse_agent_base_url,
                api_key=cfg.browse_agent_api_key,
            ) as client:
                self.browse_agent = WebPageReaderAgentV2(client=client).as_tool(
                    max_tokens=cfg.browse_agent_max_tokens,
                    temperature=cfg.browse_agent_temperature,
                )
                self.composed_browse_tool = ChainedTool(
                    [self.browse_tool, self.browse_agent],
                    name="browse_webpage",
                    tool_parser=cfg.tool_parser,
                    output_formatting="last",
                )
        else:
            self.composed_browse_tool = self.browse_tool

        with LLMToolClient(
            model_name=cfg.search_agent_model_name,
            tokenizer_name=cfg.search_agent_tokenizer_name,
            base_url=cfg.search_agent_base_url,
            api_key=cfg.search_agent_api_key,
        ) as client:
            self.search_agent = SearchAgent(
                client=client,
                tools=[self.search_tool, self.search_tool2, self.composed_browse_tool],
            )
            self.answer_agent = AnswerAgent(
                client=client,
            )

    async def __call__(
        self,
        problem: str,
        dataset_name: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        cfg = self.configuration
        assert cfg is not None

        # import litellm

        # litellm._turn_on_debug()

        # Set the question for the browse agent
        # TODO: This is a bit hectic and hacky, but it works for now
        # The problem: it uses a bad way to enable the runtime dynamics
        if isinstance(self.composed_browse_tool, ChainedTool):
            browse_tool = self.composed_browse_tool.tools[0]
            browse_tool.bm25_query = problem
            browse_agent = self.composed_browse_tool.tools[-1]
            browse_agent.agent.question = problem
        else:
            browse_tool = self.composed_browse_tool
            browse_tool.bm25_query = problem

        results = await self.search_agent(
            question=problem,
            dataset_name=dataset_name,
            max_tokens=cfg.search_agent_max_tokens,
            temperature=cfg.search_agent_temperature,
            max_tool_calls=cfg.search_agent_max_tool_calls,
            verbose=verbose,
        )

        browsed_links = []
        searched_links = []
        total_tool_calls = 0
        failed_tool_calls = 0
        failed_tool_call_errors = []
        for tool_output in results.tool_calls:
            total_tool_calls += 1
            if tool_output.error != "":
                failed_tool_calls += 1
                failed_tool_call_errors.append(tool_output.error)

            if tool_output.tool_name in ["snippet_search", "google_search"]:
                searched_links.extend(
                    [document.url for document in tool_output.documents]
                )

            if tool_output.tool_name == "browse_webpage":
                if isinstance(self.composed_browse_tool, ChainedTool):
                    if tool_output.raw_output is None:
                        continue
                    if chained_tool_outputs := tool_output.raw_output.get(
                        "tool_outputs"
                    ):
                        for document in chained_tool_outputs[0].documents:
                            if document.url:
                                browsed_links.append(document.url)
                else:
                    if hasattr(tool_output, "documents"):
                        for document in tool_output.documents:
                            if document.url:
                                browsed_links.append(document.url)
                    else:
                        print(
                            f"Warning: browse_webpage tool output has no documents: {tool_output}"
                        )

        browsed_links = list(set(browsed_links))
        searched_links = list(set(searched_links))

        if "<answer>" in results.generated_text:
            return {
                "final_response": self.search_agent.postprocess_output(results),
                "full_traces": results,
                "browsed_links": browsed_links,
                "searched_links": searched_links,
                "total_tool_calls": total_tool_calls,
                "total_failed_tool_calls": failed_tool_calls,
                "failed_tool_call_errors": failed_tool_call_errors,
            }

        answer = await self.answer_agent(
            question=problem,
            history=results.generated_text,
            dataset_name=dataset_name,
            additional_instructions="Now please generate an based on the search results by far:",
            generation_prefix="<answer>",
            max_tokens=cfg.search_agent_max_tokens,
            temperature=cfg.search_agent_temperature,
            verbose=verbose,
        )

        if verbose:
            print(results)  # noqa: T201

        answer.tool_calls = [results.model_dump()]

        return {
            "final_response": self.answer_agent.postprocess_output(answer),
            "full_traces": answer,
            "browsed_links": browsed_links,
            "searched_links": searched_links,
        }


if __name__ == "__main__":
    AutoReasonSearchWorkflow.app()
