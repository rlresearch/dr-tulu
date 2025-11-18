import asyncio
import inspect
import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Type,
    TypeVar,
    Union,
)

import litellm
import typer
import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_async

from .evaluation_utils.data_types import DatasetConfig

T = TypeVar("T")
R = TypeVar("R")


def _parse_overrides(overrides_str: Optional[str]) -> Dict[str, Any]:
    """Parse override string in format 'param1=value1,param2=value2' into dict."""
    if not overrides_str:
        return {}

    overrides = {}
    for pair in overrides_str.split(","):
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to appropriate type
        if value.lower() == "true":
            overrides[key] = True
        elif value.lower() == "false":
            overrides[key] = False
        elif value.lower() in ["none", "null"]:
            overrides[key] = None
        elif value.isdigit():
            overrides[key] = int(value)
        else:
            try:
                overrides[key] = float(value)
            except ValueError:
                overrides[key] = value

    return overrides


class DotDict(dict):
    """
    A dictionary that allows both dictionary-style access and attribute-style (dot) access.

    Example:
        config = DotDict({"a": 1, "b": {"c": 2}})
        config["a"]  # 1
        config.a     # 1
        config.b.c   # 2
    """

    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

        # Convert nested dictionaries to DotDict
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)

    def __getattr__(self, name):
        """
        Allow attribute-style access for dictionary keys.
        """
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            ) from e


def load_config(config_path: str) -> DotDict:
    """
    Load configuration from a YAML file with environment variable interpolation support.

    Environment variables can be used with ${oc.env:VAR_NAME} syntax.
    Optional defaults can be provided: ${oc.env:VAR_NAME,default_value}

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        DotDict containing the configuration with environment variables resolved
    """
    omega_conf = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(omega_conf, resolve=True)
    return DotDict(config_dict or {})


def save_config(config: Union[Dict[str, Any], BaseModel], path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Dictionary or Pydantic model representing the configuration
        path: File path to save YAML
    """
    if isinstance(config, BaseModel):
        data: Dict[str, Any] = config.model_dump(exclude_none=True)
    else:
        data = dict(config or {})

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


class BaseWorkflowConfiguration(BaseModel):
    """Base configuration model for workflows. Subclasses should declare fields with defaults."""


class BaseWorkflow(ABC):
    """
    Base class for defining workflows with configuration support and parallel execution capabilities.

    Subclasses can define an inner `Configuration` class extending `BaseWorkflowConfiguration`.
    """

    Configuration: Optional[Type[BaseWorkflowConfiguration]] = BaseWorkflowConfiguration

    __logger__ = logging.getLogger(__name__)
    __logger__.setLevel(logging.INFO)

    @property
    @abstractmethod
    def _default_configuration_path(self) -> Optional[str]:
        """Default configuration file path for this workflow. Should be overridden by subclasses."""
        pass

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)

        if cls.Configuration is None or not issubclass(
            cls.Configuration, BaseWorkflowConfiguration
        ):
            cls.__logger__.warning(
                f"[{cls.__name__}] No configuration class defined. Consider adding a nested Configuration subclass."
            )

        return instance

    def __init__(
        self,
        configuration: Optional[
            Union[BaseWorkflowConfiguration, Dict[str, Any], str]
        ] = None,
        **overrides: Any,
    ) -> None:
        """
        Initialize the workflow, building a typed configuration from various inputs.

        Args:
            configuration: One of
                - Pydantic configuration instance
                - Dictionary of configuration values
                - String path to a YAML file OR a YAML content string
            overrides: Keyword overrides to apply on top of the configuration
        """
        self.configuration = self._build_configuration(configuration, **overrides)
        self.setup_components()

    # ---- Configuration helpers ----

    def _build_configuration(
        self,
        configuration: Optional[Union[BaseWorkflowConfiguration, Dict[str, Any], str]],
        **overrides: Any,
    ) -> Optional[BaseWorkflowConfiguration]:
        """
        Normalize and instantiate the typed configuration for this workflow.
        """
        if self.Configuration is None:
            return None

        if isinstance(configuration, BaseWorkflowConfiguration):
            base_data: Dict[str, Any] = configuration.model_dump(exclude_none=True)
        elif isinstance(configuration, dict):
            base_data = dict(configuration)
        elif isinstance(configuration, str):
            # Treat as file path if it exists; otherwise parse as YAML content
            if os.path.exists(configuration):
                base_data = load_config(configuration)
            else:
                omega_conf = OmegaConf.create(configuration)
                parsed = OmegaConf.to_container(omega_conf, resolve=True) or {}
                base_data = dict(parsed)
        elif configuration is None:
            base_data = {}
        else:
            raise TypeError(
                "configuration must be a BaseWorkflowConfiguration, dict, str path/YAML, or None"
            )

        # Apply overrides (ignore Nones to avoid wiping defaults)
        override_data = {k: v for k, v in overrides.items()}
        merged: Dict[str, Any] = {**base_data, **override_data}

        try:
            return self.Configuration(**merged)  # type: ignore[arg-type]
        except Exception as e:
            raise ValueError(f"Failed to construct configuration: {e}") from e

    def config(
        self,
        configuration: Optional[
            Union[BaseWorkflowConfiguration, Dict[str, Any], str]
        ] = None,
        *,
        reinitialize_components: bool = True,
        **overrides: Any,
    ) -> "BaseWorkflow":
        """
        Update configuration at runtime. Optionally reinitialize components.

        The `configuration` can be a YAML path/string, a dict, or a typed model.
        """
        current = (
            self.configuration.model_dump(exclude_none=True)
            if isinstance(self.configuration, BaseModel)
            else {}
        )

        if isinstance(configuration, BaseWorkflowConfiguration):
            incoming: Dict[str, Any] = configuration.model_dump(exclude_none=True)
        elif isinstance(configuration, dict):
            incoming = dict(configuration)
        elif isinstance(configuration, str):
            if os.path.exists(configuration):
                incoming = dict(load_config(configuration))
            else:
                omega_conf = OmegaConf.create(configuration)
                incoming = dict(OmegaConf.to_container(omega_conf, resolve=True) or {})
        elif configuration is None:
            incoming = {}
        else:
            raise TypeError(
                "configuration must be a BaseWorkflowConfiguration, dict, str path/YAML, or None"
            )

        # Merge and rebuild
        merged = {
            **current,
            **incoming,
            **{k: v for k, v in overrides.items() if v is not None},
        }
        self.configuration = self.Configuration(**merged)  # type: ignore[arg-type]

        if reinitialize_components:
            self.setup_components()
        return self

    def save_configuration(self, path: str) -> None:
        """Persist current configuration to a YAML file."""
        if self.configuration is None:
            raise ValueError("No configuration to save.")
        save_config(self.configuration, path)

    def configuration_dict(self) -> Dict[str, Any]:
        """Return the configuration as a plain dictionary."""
        if self.configuration is None:
            return {}
        return self.configuration.model_dump(exclude_none=True)

    def _parse_args(self, item: T, **kwargs):
        """Parse arguments from item based on its type, prioritizing item values."""
        if isinstance(item, dict):
            merged_kwargs = {**kwargs, **item}
            return (), merged_kwargs
        else:
            raise ValueError(f"Unsupported item type: {type(item)}")
        # elif isinstance(item, (list, tuple)):
        #     return item, kwargs
        # else:
        #     return (item,), kwargs

    @property
    def logger(self) -> logging.Logger:
        return self.__logger__

    async def _execute_with_semaphore(
        self, semaphore: asyncio.Semaphore, item: T, **kwargs
    ) -> R:
        """Execute the workflow's __call__ method with semaphore control."""
        async with semaphore:
            parsed_args, parsed_kwargs = self._parse_args(item, **kwargs)
            return await self.__call__(*parsed_args, **parsed_kwargs)

    async def map(
        self,
        items: Iterable[T],
        *,
        max_concurrent_tasks: Optional[int] = 1,
        progress_desc: Optional[str] = None,
        return_exceptions: bool = False,
        **kwargs,
    ) -> List[Union[R, Exception]]:
        """
        Execute the workflow in parallel across multiple items with semaphore control and progress tracking.

        Args:
            items: Iterable of items to process
            max_concurrent_tasks: Override for max concurrent tasks (uses instance default if None)
            enable_progress_bar: Whether to show progress bar (default: True)
            progress_desc: Description for progress bar (default: auto-generated)
            return_exceptions: If True, exceptions are returned instead of raised
            **kwargs: Additional keyword arguments to pass to __call__

        Returns:
            List of results from __call__ method, in the same order as input items
        """

        # When max_concurrent_tasks=1, reduce to simple python sequential execution
        if max_concurrent_tasks == 1:
            results = []
            for item in tqdm(items, desc=progress_desc):
                parsed_args, parsed_kwargs = self._parse_args(item, **kwargs)
                result = await self.__call__(*parsed_args, **parsed_kwargs)
                results.append(result)

            return results

        semaphore = asyncio.Semaphore(max_concurrent_tasks)
        tasks = [
            self._execute_with_semaphore(semaphore, item, **kwargs) for item in items
        ]

        # Execute with optional progress bar
        results = await tqdm_async.gather(*tasks)
        # TODO: tqdm_async won't allow return_exceptions=True
        # https://github.com/tqdm/tqdm/issues/1286

        return results

    # ---- Lifecycle hooks ----

    @abstractmethod
    def setup_components(self) -> None:
        """Create or refresh underlying components based on configuration."""
        raise NotImplementedError

    @abstractmethod
    async def __call__(self, *args, **kwargs) -> Any:
        """Execute the workflow."""
        raise NotImplementedError

    # ---- generate methods ----
    def _shard_dataset_for_worker(self, dataset: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """
        Shard dataset for multi-worker processing.
        
        Args:
            dataset: Full dataset to shard
            **kwargs: Should contain num_total_workers and worker_index if sharding is needed
            
        Returns:
            Sharded dataset for this worker
        """
        num_total_workers = kwargs.get('num_total_workers')
        worker_index = kwargs.get('worker_index')
        
        if num_total_workers is not None and worker_index is not None:
            # Shard the dataset for this worker
            total_examples = len(dataset)
            examples_per_worker = total_examples // num_total_workers
            remainder = total_examples % num_total_workers
            
            # Calculate start and end indices for this worker
            start_idx = worker_index * examples_per_worker + min(worker_index, remainder)
            end_idx = start_idx + examples_per_worker + (1 if worker_index < remainder else 0)
            
            dataset = dataset[start_idx:end_idx]
            
            self.logger.warning(
                f"Worker {worker_index}/{num_total_workers}: Processing examples {start_idx}-{end_idx-1} "
                f"({len(dataset)} examples out of {total_examples} total)"
            )
        
        return dataset

    async def generate_dataset(
        self,
        dataset_config: Union[Dict[str, Any], "DatasetConfig"],
        max_concurrent_tasks: Optional[int] = 5,
        output_file: Optional[Union[str, "Path"]] = None,
        include_original_data: bool = True,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for evaluation datasets in parallel.

        Args:
            dataset_config: DatasetConfig object or dict with dataset configuration
            max_concurrent_tasks: Max concurrent processing tasks
            output_file: Optional path to save results as JSON
            local_data_dir: Override local data directory
            **kwargs: Additional arguments passed to workflow execution

        Returns:
            List of results with original data and generated responses
        """

        from .evaluation_utils.load_eval_data import load_eval_dataset

        # Load dataset using configuration
        dataset = load_eval_dataset(dataset_config)

        # Handle worker-based data sharding
        dataset = self._shard_dataset_for_worker(dataset, **kwargs)

        # Run some preprocessing on the dataset - filter fields based on __call__ signature
        call_sig = inspect.signature(self.__call__)
        call_params = set(call_sig.parameters.keys())
        dataset_keys = set(dataset[0].keys())

        self.logger.warning(
            f"Call parameters: {call_params}; Dataset keys: {dataset_keys}. "
            f"Removing {dataset_keys - call_params} fields."
        )
        dataset_to_process = [
            {k: v for k, v in item.items() if k in call_params} for item in dataset
        ]

        self.logger.warning(
            f"Loaded {len(dataset)} examples from {dataset_config['name']}"
        )

        # Filter out worker-specific parameters before passing to workflow
        workflow_kwargs = {k: v for k, v in kwargs.items() if k not in ['num_total_workers', 'worker_index']}
        
        # Process in parallel using the workflow's map function
        results = await self.map(
            dataset_to_process,
            max_concurrent_tasks=max_concurrent_tasks,
            progress_desc=f"Generating {dataset_config['name']}",
            **workflow_kwargs,
        )

        # Combine original data with results
        output = []
        for i, (example, result) in enumerate(zip(dataset, results)):
            result = result.copy()

            final_response = result.pop("final_response")
            full_traces = result.pop("full_traces")

            eval_output = {
                "example_id": example["id"],
                "problem": example["problem"],
                "final_response": final_response,
                "full_traces": full_traces.model_dump(),
            }
            # If there is additional output data, add it to the eval_output
            if len(result):
                eval_output["additional_output_data"] = result

            if include_original_data:
                eval_output["original_data"] = example
            output.append(eval_output)

        # Optionally save to file
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w") as f:
                json.dump(output, f, indent=2)
            self.logger.info(f"Saved {len(output)} results to {output_path}")

        return output

    async def generate_dataset_batch(
        self,
        dataset_config: Union[Dict[str, Any], "DatasetConfig"],
        batch_size: int = 50,
        max_concurrent_tasks: Optional[int] = 5,
        output_file: Optional[Union[str, "Path"]] = None,
        include_original_data: bool = True,
        keep_instance_ids: Optional[Set[str]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for evaluation datasets in batches with resumption support.

        Args:
            dataset_config: DatasetConfig object or dict with dataset configuration
            batch_size: Number of examples to process per batch
            max_concurrent_tasks: Max concurrent processing tasks
            output_file: Optional path to save results as JSONL
            include_original_data: Whether to include original dataset data
            keep_instance_ids: Optional set of IDs to keep for processing (for rejection sampling)
            **kwargs: Additional arguments passed to workflow execution

        Returns:
            List of results with original data and generated responses
        """
        from .evaluation_utils.load_eval_data import load_eval_dataset

        # Load dataset
        dataset = load_eval_dataset(dataset_config)

        # Handle worker-based data sharding
        dataset = self._shard_dataset_for_worker(dataset, **kwargs)

        if keep_instance_ids is not None:
            current_dataset_size = len(dataset)
            dataset = [item for item in dataset if item["id"] in keep_instance_ids]
            self.logger.warning(
                f"Filtering dataset from {current_dataset_size} examples to {len(dataset)} examples based on keep_instance_ids"
            )

        # Check existing results if output file exists
        existing_ids = set()
        if output_file and os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    if line.strip():
                        existing_result = json.loads(line)
                        existing_ids.add(existing_result.get("example_id"))

        # Filter out already processed examples
        remaining_dataset = [item for item in dataset if item["id"] not in existing_ids]

        if not remaining_dataset or len(remaining_dataset) == 0:
            self.logger.info("All examples already processed")
            return []

        # Add the dataset name to the remaining dataset
        if "dataset_name" not in remaining_dataset[0]:
            for item in remaining_dataset:
                item["dataset_name"] = dataset_config["name"]

        self.logger.warning(
            f"Found {len(existing_ids)} existing results, processing {len(remaining_dataset)} remaining examples"
        )

        # Filter dataset fields based on __call__ signature
        call_sig = inspect.signature(self.__call__)
        call_params = set(call_sig.parameters.keys())
        dataset_keys = set(dataset[0].keys())

        self.logger.warning(
            f"Call parameters: {call_params}; Dataset keys: {dataset_keys}. "
            f"Removing {dataset_keys - call_params} fields."
        )

        # Process in batches
        output_path = Path(output_file) if output_file else None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        all_results = []
        for i in range(0, len(remaining_dataset), batch_size):
            batch = remaining_dataset[i : i + batch_size]
            batch_to_process = [
                {k: v for k, v in item.items() if k in call_params} for item in batch
            ]

            self.logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(remaining_dataset) + batch_size - 1)//batch_size}"
            )

            try:
                # Filter out worker-specific parameters before passing to workflow
                workflow_kwargs = {k: v for k, v in kwargs.items() if k not in ['num_total_workers', 'worker_index']}
                
                # Process batch
                batch_results = await self.map(
                    batch_to_process,
                    max_concurrent_tasks=max_concurrent_tasks,
                    progress_desc=f"Batch {i//batch_size + 1}",
                    **workflow_kwargs,
                )

                # Format batch results
                formatted_results = []
                for example, result in zip(batch, batch_results):
                    result = result.copy()
                    final_response = result.pop("final_response")
                    full_traces = result.pop("full_traces")

                    eval_output = {
                        "example_id": example["id"],
                        "problem": example["problem"],
                        "final_response": final_response,
                        "full_traces": full_traces.model_dump(),
                    }

                    if len(result):
                        eval_output["additional_output_data"] = result

                    if include_original_data:
                        eval_output["original_data"] = example

                    formatted_results.append(eval_output)

                all_results.extend(formatted_results)

                # Append batch to file
                if output_path:
                    with open(output_path, "a") as f:
                        for result in formatted_results:
                            f.write(json.dumps(result) + "\n")

            except Exception as e:
                self.logger.error(
                    f"Error processing batch {i//batch_size + 1}: {e}. Continuing with next batch."
                )

        # Sort final file by original dataset order if output file was provided
        if output_path and os.path.exists(output_path):
            id_to_order = {item["id"]: i for i, item in enumerate(dataset)}

            # Read all results
            all_file_results = []
            with open(output_path, "r") as f:
                for line in f:
                    if line.strip():
                        all_file_results.append(json.loads(line))

            # Sort by original order
            all_file_results.sort(
                key=lambda x: id_to_order.get(x["example_id"], float("inf"))
            )

            # Rewrite sorted results
            with open(output_path, "w") as f:
                for result in all_file_results:
                    f.write(json.dumps(result) + "\n")

            self.logger.info(
                f"Sorted and saved {len(all_file_results)} total results to {output_path}"
            )

        return all_results

    @classmethod
    def app(cls):
        """Create a typer app with commands for this workflow."""
        app = typer.Typer()

        @app.command()
        def debug(
            config_file: Optional[str] = typer.Option(
                None,
                "--config",
                help="Configuration file path for the class",
            ),
        ):
            if config_file is None:
                config_file = cls._default_configuration_path
                cls.__logger__.warning(
                    f"Using default configuration file: {config_file}"
                )

            workflow = cls(configuration=config_file)
            workflow.setup_components()
            typer.echo("Setup complete")

        @app.command()
        def generate_dataset(
            dataset_name: str = typer.Argument(
                ...,
                help="Dataset name (e.g., 'simpleqa', 'browsecomp')",
            ),
            num_examples: Optional[str] = typer.Option(
                None,
                "--num-examples",
                "-n",
                help="Number of examples to process (or 'ablation')",
            ),
            subset: Optional[str] = typer.Option(
                None,
                "--subset",
                "-s",
                help="Dataset subset to use",
            ),
            max_concurrent_tasks: int = typer.Option(
                5,
                "--max-concurrent",
                "-c",
                help="Maximum concurrent tasks",
            ),
            batch_size: int = typer.Option(
                20,
                "--batch-size",
                "-b",
                help="Batch size for processing (when using cache)",
            ),
            use_cache: bool = typer.Option(
                False,
                "--use-cache",
                help="Load from existing cache and use batch processing",
            ),
            output_file: Optional[str] = typer.Option(
                None,
                "--output",
                "-o",
                help="Output file path",
            ),
            config_file: Optional[str] = typer.Option(
                None,
                "--config",
                help="Configuration file path for the class",
            ),
            verbose: bool = typer.Option(
                False,
                "--verbose",
                "-v",
                help="Verbose output",
            ),
            config_overrides: Optional[str] = typer.Option(
                None,
                "--config-overrides",
                help="Override configuration parameters in format 'param1=value1,param2=value2'",
            ),
            num_total_workers: Optional[int] = typer.Option(
                None,
                "--num_total_workers",
                help="Total number of workers for parallel evaluation",
            ),
            worker_index: Optional[int] = typer.Option(
                None,
                "--worker_index",
                help="Index of current worker (0-based)",
            ),
        ):
            """Generate responses for an evaluation dataset."""

            if config_file is None:
                config_file = cls._default_configuration_path
                cls.__logger__.warning(
                    f"Using default configuration file: {config_file}"
                )

            if not verbose:
                # fmt: off
                logging.getLogger('mcp.client.streamable_http').setLevel(logging.WARNING)
                logging.getLogger('LiteLLM').setLevel(logging.WARNING)
                # litellm._logging._disable_debugging()
                litellm.turn_off_message_logging=True
                # fmt: on

            # Check if num_examples is an integer or 'ablation'
            # The reason is that current typer doesn't support Union[int, str]
            if num_examples not in ["ablation", "final_run", None]:
                try:
                    num_examples = int(num_examples)
                except ValueError:
                    raise ValueError("num_examples must be an integer or 'ablation'")

            # Validate worker parameters
            if (num_total_workers is None) != (worker_index is None):
                raise ValueError("Both num_total_workers and worker_index must be provided together or not at all")
            
            if num_total_workers is not None:
                if num_total_workers <= 0:
                    raise ValueError("num_total_workers must be positive")
                if worker_index < 0 or worker_index >= num_total_workers:
                    raise ValueError(f"worker_index must be between 0 and {num_total_workers - 1}")

            dataset_config = {
                "name": dataset_name,
                "num_examples": num_examples,
                "subset": subset,
            }

            parsed_overrides = _parse_overrides(config_overrides)
            if parsed_overrides:
                cls.__logger__.warning(
                    f"Overriding the config with: {parsed_overrides}"
                )

            workflow = cls(configuration=config_file, **parsed_overrides)

            # Handle worker-specific output file paths and result merging
            if num_total_workers is not None:
                # Create worker-specific output file path
                if output_file:
                    from pathlib import Path
                    output_path = Path(output_file)
                    worker_output_file = str(output_path.parent / f"{output_path.stem}_worker_{worker_index}{output_path.suffix}")
                else:
                    worker_output_file = None
                
                # Pass worker parameters to the dataset generation
                if use_cache:
                    results = asyncio.get_event_loop().run_until_complete(
                        workflow.generate_dataset_batch(
                            dataset_config=dataset_config,
                            max_concurrent_tasks=max_concurrent_tasks,
                            batch_size=batch_size,
                            output_file=worker_output_file,
                            verbose=verbose,
                            num_total_workers=num_total_workers,
                            worker_index=worker_index,
                        )
                    )
                else:
                    results = asyncio.get_event_loop().run_until_complete(
                        workflow.generate_dataset(
                            dataset_config=dataset_config,
                            max_concurrent_tasks=max_concurrent_tasks,
                            output_file=worker_output_file,
                            verbose=verbose,
                            num_total_workers=num_total_workers,
                            worker_index=worker_index,
                        )
                    )
                
                # Try to merge results if all workers are done
                if output_file:
                    workflow._try_merge_worker_results(output_file, num_total_workers)
            else:
                # Original single-worker behavior
                if use_cache:
                    results = asyncio.get_event_loop().run_until_complete(
                        workflow.generate_dataset_batch(
                            dataset_config=dataset_config,
                            max_concurrent_tasks=max_concurrent_tasks,
                            batch_size=batch_size,
                            output_file=output_file,
                            verbose=verbose,
                        )
                    )
                else:
                    results = asyncio.get_event_loop().run_until_complete(
                        workflow.generate_dataset(
                            dataset_config=dataset_config,
                            max_concurrent_tasks=max_concurrent_tasks,
                            output_file=output_file,
                            verbose=verbose,
                        )
                    )

            typer.echo(f"Generated {len(results)} responses")
            if output_file:
                if num_total_workers is not None:
                    typer.echo(f"Worker {worker_index} results saved to worker-specific file")
                    # Check if final merged file exists
                    if os.path.exists(output_file):
                        typer.echo(f"Final merged results available at {output_file}")
                else:
                    typer.echo(f"Results saved to {output_file}")

        @app.command()
        def collect_rejection_sampling_data(
            output_file: str = typer.Argument(
                ...,
                help="Output file path for collected results",
            ),
            max_iterations: int = typer.Option(
                10,
                "--max-iterations",
                help="Maximum number of iterations to check (default: 10)",
            ),
            verbose: bool = typer.Option(
                False,
                "--verbose",
                "-v",
                help="Verbose output",
            ),
        ):
            """Collect final results from rejection sampling iterations."""

            if not output_file:
                raise ValueError("Output file must be specified")

            # Parse the output file path to get base name
            output_path = Path(output_file)
            base_name = str(output_path).replace(".jsonl", "")
            parent_dir = output_path.parent

            # Track all successfully processed IDs and their results
            successful_results = {}  # id -> result dict
            iteration_stats = []

            # Process each iteration
            for iteration in range(max_iterations):
                iter_file = f"{base_name}-iter-{iteration}.jsonl"
                eval_file = f"{base_name}-iter-{iteration}_eval_results.json"

                if not os.path.exists(iter_file):
                    if verbose:
                        cls.__logger__.warning(
                            f"Iteration {iteration} file not found: {iter_file}"
                        )
                    break

                if not os.path.exists(eval_file):
                    cls.__logger__.warning(
                        f"Eval file not found for iteration {iteration}: {eval_file}"
                    )
                    continue

                # Load eval results to find successful IDs
                with open(eval_file, "r") as f:
                    eval_data = json.load(f)

                successful_ids_this_iter = set()
                for result in eval_data.get("per_example_results", []):
                    if result.get("score", 0) == 1.0:
                        successful_ids_this_iter.add(result["id"])

                # Load the corresponding results from jsonl
                iter_results = {}
                with open(iter_file, "r") as f:
                    for line in f:
                        if line.strip():
                            result = json.loads(line)
                            example_id = result.get("example_id")
                            if example_id in successful_ids_this_iter:
                                # Only add if not already successful in a previous iteration
                                if example_id not in successful_results:
                                    successful_results[example_id] = result
                                    iter_results[example_id] = result

                iteration_stats.append(
                    {
                        "iteration": iteration,
                        "total_in_file": len(eval_data.get("per_example_results", [])),
                        "successful": len(successful_ids_this_iter),
                        "newly_successful": len(iter_results),
                    }
                )

                if verbose:
                    print(
                        f"Iteration {iteration}: {len(successful_ids_this_iter)} successful, "
                        f"{len(iter_results)} newly successful"
                    )

            # Report statistics
            print("Rejection Sampling Summary:")
            print("-" * 40)
            for stats in iteration_stats:
                print(
                    f"Iteration {stats['iteration']}: "
                    f"{stats['successful']}/{stats['total_in_file']} successful "
                    f"({stats['newly_successful']} new)"
                )
            print("-" * 40)
            print(f"Total successful examples: {len(successful_results)}")

            # Check if we need to get the original order from iteration 0
            iter0_file = f"{base_name}-iter-0.jsonl"
            id_order = {}
            if os.path.exists(iter0_file):
                with open(iter0_file, "r") as f:
                    for idx, line in enumerate(f):
                        if line.strip():
                            result = json.loads(line)
                            id_order[result.get("example_id")] = idx

            # Sort results by original order if possible
            final_results = list(successful_results.values())
            if id_order:
                final_results.sort(
                    key=lambda x: id_order.get(x.get("example_id"), float("inf"))
                )

            # Write final collected results
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                for result in final_results:
                    f.write(json.dumps(result) + "\n")

            print(f"Collected {len(final_results)} successful results")
            print(f"Results saved to {output_file}")

            # Print final success rate
            if iteration_stats and iteration_stats[0]["total_in_file"] > 0:
                success_rate = (
                    len(successful_results) / iteration_stats[0]["total_in_file"] * 100
                )
                print(f"Overall success rate: {success_rate:.1f}%")

        @app.command()
        def rejection_sampling(
            dataset_name: str = typer.Argument(
                ...,
                help="Dataset name (e.g., 'simpleqa', 'browsecomp')",
            ),
            iteration: int = typer.Option(
                ...,
                help="Iteration number for rejection sampling",
            ),
            output_file: str = typer.Option(
                ...,
                "--output",
                "-o",
                help="Base output file path (will be appended with -iter-N)",
            ),
            num_examples: Optional[str] = typer.Option(
                None,
                "--num-examples",
                "-n",
                help="Number of examples to process (or 'ablation')",
            ),
            subset: Optional[str] = typer.Option(
                None,
                "--subset",
                "-s",
                help="Dataset subset to use",
            ),
            max_concurrent_tasks: int = typer.Option(
                5,
                "--max-concurrent",
                "-c",
                help="Maximum concurrent tasks",
            ),
            batch_size: int = typer.Option(
                20,
                "--batch-size",
                "-b",
                help="Batch size for processing",
            ),
            config_file: Optional[str] = typer.Option(
                None,
                "--config",
                help="Configuration file path for the class",
            ),
            verbose: bool = typer.Option(
                False,
                "--verbose",
                "-v",
                help="Verbose output",
            ),
            config_overrides: Optional[str] = typer.Option(
                None,
                "--config-overrides",
                help="Override configuration parameters in format 'param1=value1,param2=value2'",
            ),
            threshold: float = typer.Option(
                1.0,
                "--threshold",
                "-t",
                help="Score threshold for rejection sampling (examples with score < threshold will be regenerated)",
            ),
        ):
            """Run rejection sampling for an evaluation dataset."""

            if config_file is None:
                config_file = cls._default_configuration_path
                cls.__logger__.warning(
                    f"Using default configuration file: {config_file}"
                )

            if not verbose:
                # fmt: off
                logging.getLogger('mcp.client.streamable_http').setLevel(logging.WARNING)
                logging.getLogger('LiteLLM').setLevel(logging.WARNING)
                litellm.turn_off_message_logging=True
                # fmt: on

            # Check if num_examples is an integer or 'ablation'
            if num_examples not in ["ablation", "final_run", None]:
                try:
                    num_examples = int(num_examples)
                except ValueError:
                    raise ValueError(
                        "num_examples must be an integer or 'ablation' or 'final_run'"
                    )

            dataset_config = {
                "name": dataset_name,
                "num_examples": num_examples,
                "subset": subset,
            }

            parsed_overrides = _parse_overrides(config_overrides)
            if parsed_overrides:
                cls.__logger__.warning(
                    f"Overriding the config with: {parsed_overrides}"
                )

            workflow = cls(configuration=config_file, **parsed_overrides)

            # Construct output file path with iteration
            base_name = output_file.replace(".jsonl", "")
            if f"iter-{iteration}" in base_name:
                base_name = base_name[: -len(f"-iter-{iteration}")]
                current_output_file = output_file
            else:
                current_output_file = f"{base_name}-iter-{iteration}.jsonl"

            cls.__logger__.warning(f"Base name: {base_name}")
            cls.__logger__.warning(f"Output file: {current_output_file}")

            # For iterations > 0, load previous eval results and filter
            keep_instance_ids = None
            if iteration > 0:
                # Construct previous eval file path
                prev_eval_file = f"{base_name}-iter-{iteration-1}_eval_results.json"

                if not os.path.exists(prev_eval_file):
                    raise FileNotFoundError(
                        f"Previous evaluation file not found: {prev_eval_file}. "
                        f"Make sure iteration {iteration-1} has been evaluated."
                    )

                # Load previous eval results
                with open(prev_eval_file, "r") as f:
                    eval_data = json.load(f)

                # Extract IDs where score != threshold
                keep_instance_ids = set()
                for result in eval_data.get("per_example_results", []):
                    if result.get("score", 0) < threshold:
                        keep_instance_ids.add(result["id"])

                cls.__logger__.warning(
                    f"Loaded {len(eval_data.get('per_example_results', []))} results from previous iteration. "
                    f"Found {len(keep_instance_ids)} examples to regenerate (score < {threshold})."
                )

                if not keep_instance_ids:
                    cls.__logger__.warning(
                        f"All examples have score >= {threshold}. No regeneration needed."
                    )
                    return

            # Run generation with batch processing (always use cache for rejection sampling)
            results = asyncio.get_event_loop().run_until_complete(
                workflow.generate_dataset_batch(
                    dataset_config=dataset_config,
                    max_concurrent_tasks=max_concurrent_tasks,
                    batch_size=batch_size,
                    output_file=current_output_file,
                    verbose=verbose,
                    keep_instance_ids=keep_instance_ids,
                )
            )

            typer.echo(f"Generated {len(results)} responses for iteration {iteration}")
            typer.echo(f"Results saved to {current_output_file}")

        return app()

    def _try_merge_worker_results(self, final_output_file: str, num_total_workers: int) -> bool:
        """
        Try to merge worker results into a final file if all workers are complete.
        
        Args:
            final_output_file: Path to the final merged output file
            num_total_workers: Total number of workers expected
            
        Returns:
            True if merge was successful, False otherwise
        """
        from pathlib import Path
        import json
        import os
        
        output_path = Path(final_output_file)
        
        # Check if all worker files exist
        worker_files = []
        for worker_idx in range(num_total_workers):
            worker_file = str(output_path.parent / f"{output_path.stem}_worker_{worker_idx}{output_path.suffix}")
            if os.path.exists(worker_file):
                worker_files.append(worker_file)
            else:
                # Not all workers are done yet
                return False
        
        # All worker files exist, merge them
        try:
            all_results = []
            for worker_file in worker_files:
                with open(worker_file, 'r') as f:
                    if worker_file.endswith('.jsonl'):
                        # JSONL format
                        for line in f:
                            if line.strip():
                                all_results.append(json.loads(line))
                    else:
                        # JSON format
                        worker_results = json.load(f)
                        if isinstance(worker_results, list):
                            all_results.extend(worker_results)
                        else:
                            all_results.append(worker_results)
            
            # Sort results by example_id for consistency
            all_results.sort(key=lambda x: x.get('example_id', ''))
            
            # Write merged results
            with open(final_output_file, 'w') as f:
                if final_output_file.endswith('.jsonl'):
                    for result in all_results:
                        f.write(json.dumps(result) + '\n')
                else:
                    json.dump(all_results, f, indent=2)
            
            self.logger.info(f"Successfully merged {len(all_results)} results from {num_total_workers} workers to {final_output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to merge worker results: {e}")
            return False
