#!/usr/bin/env python3
"""
ResearchQA evaluation following the standard evaluation pattern.
"""

import json
import os
import tempfile
from typing import List, Dict, Optional

from samplers._types import Eval, SamplerBase, SingleEvalResult
from samplers import common
from .researchqa_loader import load_researchqa_data
from .compute_coverage import compute_coverage


class ResearchQAEval(Eval):
    def __init__(
        self,
        data_path: str,
        grader_model: SamplerBase | None = None,
        num_examples: int | None = None,
        n_threads: int = 10,
        generation_only: bool = False,
        enable_checkpoint: bool = False,
    ):
        # Load ResearchQA data
        self.items = load_researchqa_data(data_path)
        
        # Apply num_examples limit if specified
        if num_examples and num_examples < len(self.items):
            self.items = self.items[:num_examples]
        
        self.data_path = data_path
        self.grader_model = grader_model
        self.n_threads = n_threads
        self.generation_only = generation_only
        self.enable_checkpoint = enable_checkpoint

    def generate(self, sampler: SamplerBase, eval_save_dir: str = "eval_results") -> List[Dict]:
        """Generate responses for all ResearchQA items"""
        def generate_single(item):

            # ai queries only
            

            prompt_messages = [
                sampler._pack_message(content=item.query, role="user")
            ]
            sampler_response = sampler(prompt_messages)
            
            # Only save JSON-serializable data to avoid checkpoint issues
            return {
                "item_id": item.id,
                "query": item.query,
                "field": item.field,
                "response_text": sampler_response.response_text,
                "actual_queried_prompt_messages": sampler_response.actual_queried_message_list or [],
                "full_traces": sampler_response.full_traces,
                "rubric": [{"rubric_item": r.rubric_item, "type": r.type} for r in item.rubric]
            }

        # Temporarily disable checkpointing to avoid serialization issues
        # TODO: Fix checkpoint serialization for complex multiprocessing scenarios
        if self.enable_checkpoint:
            # Include num_examples in checkpoint filename for proper cache separation
            num_examples_str = f"_n{len(self.items)}"
            checkpoint_path = f"{eval_save_dir}/researchqa_generation{num_examples_str}.checkpoint.json"
            generation_data = common.map_with_progress_checkpoint(
                generate_single,
                self.items,
                checkpoint_path,
                num_threads=self.n_threads,
                checkpoint_interval=max(10, self.n_threads)
            )
        else:
            generation_data = common.map_with_progress(
                generate_single,
                self.items,
                num_threads=self.n_threads
            )
        
        return generation_data

    def evaluate(self, generation_data: List[Dict], eval_save_dir: str = "eval_results") -> List[SingleEvalResult]:
        """Evaluate ResearchQA responses using coverage computation"""
        if self.grader_model is None:
            raise ValueError("grader_model is required for ResearchQA evaluation")
        
        # Temporarily disable generation_only mode for evaluation
        original_generation_only = self.generation_only
        self.generation_only = False
        
        # Create temporary files for coverage computation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_responses:
            # Convert generation_data to the format expected by compute_coverage
            responses = {}
            for gen_data in generation_data:
                responses[gen_data["row"]["id"]] = {
                    "answer": gen_data["response_text"],
                    "query": gen_data["row"]["problem"],
                    # "field": gen_data["field"],
                    # "full_traces": gen_data["full_traces"]
                }
            
            json.dump(responses, temp_responses, indent=2)
            temp_responses_path = temp_responses.name

        # Create output path but don't create the file - let compute_coverage create it
        temp_output_fd, temp_output_path = tempfile.mkstemp(suffix='.json')
        os.close(temp_output_fd)  # Close the file descriptor
        os.unlink(temp_output_path)  # Remove the empty file - let compute_coverage create it

        # Extract model name from grader_model sampler
        model_name = getattr(self.grader_model, 'model', 'gpt-4-turbo')
        
        # Run coverage evaluation
        print(f"[DEBUG] Running compute_coverage with model: {model_name}")
        compute_coverage(
            data_path=self.data_path,
            response_map_path=temp_responses_path,
            output_path=temp_output_path,
            batch_size=8,
            model=model_name,
            model_provider="azure" if os.environ.get("AZURE_OPENAI_ENDPOINT") else "openai"
        )
        
        # Load and validate results
        if not os.path.exists(temp_output_path) or os.path.getsize(temp_output_path) == 0:
            raise ValueError("Coverage computation failed or produced empty output")
        
        with open(temp_output_path, 'r') as f:
            results = json.load(f)
        
        if not results:
            raise ValueError("Coverage computation returned empty results")
        
        # Convert to SingleEvalResult format
        single_results = []
        for gen_data in generation_data:
            item_id = gen_data["row"]["id"]
            result_data = results.get(item_id, {})
            
            coverage_score = result_data.get("coverage_score", 0)
            
            # Create conversation - handle both old and new key names
            message_list = (
                gen_data.get("actual_queried_prompt_messages") or 
                gen_data.get("actual_queried_message_list") or 
                [{"role": "user", "content": gen_data["row"]["problem"]}]
            )
            convo = message_list + [
                {"role": "assistant", "content": gen_data["response_text"]}
            ]
            
            single_results.append(SingleEvalResult(
                id=str(item_id),
                html=None,
                score=coverage_score,
                convo=convo,
                metrics={"coverage_score": coverage_score},
                example_level_metadata={
                    # "full_traces": gen_data["full_traces"],
                    # "field": gen_data["field"],
                    "item_id": item_id,
                    "query": gen_data["row"]["problem"]
                }
            ))
        
        # Clean up temporary files
        if os.path.exists(temp_responses_path):
            os.unlink(temp_responses_path)
        if os.path.exists(temp_output_path):
            os.unlink(temp_output_path)
        
        # Restore original generation_only state
        self.generation_only = original_generation_only
        
        return single_results

    def __call__(self, sampler: SamplerBase, eval_save_dir: str = "eval_results"):
        """Legacy method for backward compatibility - orchestrates generate -> evaluate -> aggregate"""
        # Generate responses
        generation_data = self.generate(sampler, eval_save_dir=eval_save_dir)
        
        # Handle generation_only mode vs evaluation mode
        if self.generation_only:
            # For generation_only, return placeholder results
            results = []
            for gen_data in generation_data:
                # Create conversation - handle both old and new key names
                message_list = (
                    gen_data.get("actual_queried_prompt_messages") or 
                    gen_data.get("actual_queried_message_list") or 
                    [{"role": "user", "content": gen_data.get("query", "")}]
                )
                convo = message_list + [
                    {"role": "assistant", "content": gen_data["response_text"]}
                ]
                
                results.append(SingleEvalResult(
                    score=0.0,
                    metrics={"generation_only": True},
                    html=None,
                    convo=convo,
                    example_level_metadata={
                        "item_id": gen_data["row"]["id"],
                        "query": gen_data["row"]["problem"],
                        # "field": gen_data["row"]["field"],
                        # "full_traces": gen_data["full_traces"],
                    },
                ))
        else:
            # Full evaluation
            results = self.evaluate(generation_data, eval_save_dir=eval_save_dir)
        
        # Use common aggregation
        return common.aggregate_results(results) 