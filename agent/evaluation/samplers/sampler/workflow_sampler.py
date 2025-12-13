import sys
import asyncio
from pathlib import Path
import re

# Add rl-rag-mcp root to path so workflows can be imported
rl_rag_mcp_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(rl_rag_mcp_root))

from .._types import MessageList, SamplerBase, SamplerResponse
from workflows.baselines_legacy.sft_search_generate import AutoReasonSearchWorkflow
from workflows.baselines_legacy.naive_rag import NaiveRagWorkflow
from workflows.baselines_legacy.react_agent import ReActWorkflow


class WorkflowSampler(SamplerBase):
    def __init__(self, run_mode: str, mcp_config: str, use_answer_only: bool = False):
        if run_mode == "naive_rag":
            self.workflow = NaiveRagWorkflow(configuration=mcp_config)
        elif run_mode == "auto_reason_search":
            self.workflow = AutoReasonSearchWorkflow(configuration=mcp_config)
        elif run_mode == "react_agent":
            self.workflow = ReActWorkflow(configuration=mcp_config)
        else:
            raise ValueError(f"Invalid run mode: {run_mode}")
        self.use_answer_only = use_answer_only
    
    def _pack_message(self, content: str, role: str):
        return {"content": content, "role": role}

    def _prepare_workflow_input(self, message_list: MessageList):
        messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in message_list])
        return messages
    
    def _extract_answer_string(self, response: str):
        # Try to extract answer string between <answer> and </answer> or in \boxed{}
        # If not found, return the full response
        # If there are multiple answer strings, return the last one

        # Try to extract answer string between <answer> and </answer> tags
        answer_match = re.findall(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            return answer_match[-1].strip()
        
        # Try to extract answer string from \boxed{} format
        boxed_match = re.findall(r'\\boxed\{([^}]*)\}', response)
        if boxed_match:
            return boxed_match[-1].strip()
        
        # Try to extract answer string between "Exact Answer:" and "\nConfidence:"
        exact_answer_match = re.findall(r'Exact Answer:(.*?)\nConfidence:', response, re.DOTALL)
        if exact_answer_match:
            return exact_answer_match[-1].strip()
        
        # Otherwise, take the string after </think> and only return the last 4k characters
        think_match = re.findall(r'</think>(.*)', response, re.DOTALL)
        if think_match:
            return think_match[-1].strip()[-4096:]
        
        return response[-4096:]
    
    def __call__(self, message_list: MessageList) -> SamplerResponse:
        messages_str = self._prepare_workflow_input(message_list)
        
        # Handle async workflows by running them with asyncio
        response = asyncio.run(self.workflow(question=messages_str))
        
        if self.use_answer_only:
            response_text = self._extract_answer_string(response.generated_text)
        else:
            response_text = response.generated_text
        
        return SamplerResponse(
            response_text=response_text,
            response_metadata={"usage": None},
            actual_queried_message_list=message_list,
            full_traces=response.model_dump(),
        )
