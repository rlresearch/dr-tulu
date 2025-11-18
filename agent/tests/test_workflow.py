import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import Mock, patch

import pytest
import yaml
from pydantic import BaseModel

from dr_agent.workflow import (
    BaseWorkflow,
    BaseWorkflowConfiguration,
    DotDict,
    load_config,
    save_config,
)


class TestConfiguration(BaseWorkflowConfiguration):
    """Test configuration for mock workflow"""

    name: str = "test_workflow"
    max_retries: int = 3
    timeout: float = 30.0


class MockWorkflow(BaseWorkflow):
    """Mock concrete workflow for testing"""

    Configuration = TestConfiguration

    @property
    def _default_configuration_path(self) -> Optional[str]:
        return None

    def setup_components(self) -> None:
        """Mock setup"""
        pass

    async def __call__(self, item: str, **kwargs) -> str:
        """Mock execution that returns processed item"""
        return f"processed: {item}"


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary"""
    return {
        "name": "test_workflow",
        "max_retries": 5,
        "timeout": 45.0,
        "nested": {"key1": "value1", "key2": {"deep_key": "deep_value"}},
    }


@pytest.fixture
def temp_config_file(sample_config_dict):
    """Create temporary config file"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(sample_config_dict, f)
        yield f.name
    Path(f.name).unlink(missing_ok=True)


class TestDotDict:
    """Test DotDict functionality"""

    def test_dict_access(self, sample_config_dict):
        """Test dictionary-style access"""
        dot_dict = DotDict(sample_config_dict)
        assert dot_dict["name"] == "test_workflow"
        assert dot_dict["max_retries"] == 5

    def test_dot_access(self, sample_config_dict):
        """Test attribute-style dot access"""
        dot_dict = DotDict(sample_config_dict)
        assert dot_dict.name == "test_workflow"
        assert dot_dict.max_retries == 5

    def test_nested_access(self, sample_config_dict):
        """Test nested dictionary access"""
        dot_dict = DotDict(sample_config_dict)
        assert dot_dict.nested.key1 == "value1"
        assert dot_dict.nested.key2.deep_key == "deep_value"

    def test_missing_attribute_error(self):
        """Test that missing attributes raise AttributeError"""
        dot_dict = DotDict({"a": 1})
        with pytest.raises(AttributeError):
            _ = dot_dict.missing_key


class TestConfigOperations:
    """Test configuration loading and saving"""

    def test_load_config(self, temp_config_file):
        """Test loading config from YAML file"""
        config = load_config(temp_config_file)
        assert isinstance(config, DotDict)
        assert config.name == "test_workflow"
        assert config.nested.key1 == "value1"

    def test_save_config_dict(self, sample_config_dict):
        """Test saving dictionary config to file"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        save_config(sample_config_dict, temp_path)

        # Verify file was created and contains correct data
        with open(temp_path, "r") as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data["name"] == "test_workflow"

        Path(temp_path).unlink(missing_ok=True)

    def test_save_config_pydantic(self):
        """Test saving Pydantic model config to file"""
        config = TestConfiguration(name="pydantic_test", max_retries=7)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        save_config(config, temp_path)

        # Verify file was created and contains correct data
        with open(temp_path, "r") as f:
            loaded_data = yaml.safe_load(f)
        assert loaded_data["name"] == "pydantic_test"
        assert loaded_data["max_retries"] == 7

        Path(temp_path).unlink(missing_ok=True)


class TestBaseWorkflow:
    """Test BaseWorkflow functionality"""

    def test_init_with_dict_config(self, sample_config_dict):
        """Test workflow initialization with dictionary config"""
        workflow = MockWorkflow(configuration=sample_config_dict)
        assert workflow.configuration.name == "test_workflow"
        assert workflow.configuration.max_retries == 5

    def test_init_with_pydantic_config(self):
        """Test workflow initialization with Pydantic config"""
        config = TestConfiguration(name="pydantic_workflow")
        workflow = MockWorkflow(configuration=config)
        assert workflow.configuration.name == "pydantic_workflow"

    def test_init_with_file_path(self, temp_config_file):
        """Test workflow initialization with file path"""
        workflow = MockWorkflow(configuration=temp_config_file)
        assert workflow.configuration.name == "test_workflow"

    def test_init_with_yaml_string(self):
        """Test workflow initialization with YAML string"""
        yaml_string = """
name: yaml_string_test
max_retries: 2
"""
        workflow = MockWorkflow(configuration=yaml_string)
        assert workflow.configuration.name == "yaml_string_test"
        assert workflow.configuration.max_retries == 2

    def test_config_overrides(self):
        """Test configuration overrides"""
        base_config = {"name": "base", "max_retries": 3}
        workflow = MockWorkflow(
            configuration=base_config, name="overridden", timeout=60.0
        )
        assert workflow.configuration.name == "overridden"
        assert workflow.configuration.max_retries == 3
        assert workflow.configuration.timeout == 60.0

    def test_config_update(self):
        """Test runtime configuration updates"""
        workflow = MockWorkflow(configuration={"name": "initial"})
        assert workflow.configuration.name == "initial"

        workflow.config({"name": "updated", "max_retries": 10})
        assert workflow.configuration.name == "updated"
        assert workflow.configuration.max_retries == 10

    def test_configuration_dict(self):
        """Test configuration dictionary export"""
        config = {"name": "test", "max_retries": 5}
        workflow = MockWorkflow(configuration=config)
        config_dict = workflow.configuration_dict()
        assert config_dict["name"] == "test"
        assert config_dict["max_retries"] == 5

    def test_save_configuration(self):
        """Test saving workflow configuration"""
        workflow = MockWorkflow(configuration={"name": "save_test"})

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        workflow.save_configuration(temp_path)

        # Verify saved config
        with open(temp_path, "r") as f:
            saved_data = yaml.safe_load(f)
        assert saved_data["name"] == "save_test"

        Path(temp_path).unlink(missing_ok=True)


class TestWorkflowExecution:
    """Test workflow execution methods"""

    @pytest.mark.asyncio
    async def test_single_item_execution(self):
        """Test executing workflow on single item"""
        workflow = MockWorkflow()
        result = await workflow("test_item")
        assert result == "processed: test_item"

    @pytest.mark.asyncio
    async def test_map_sequential_execution(self):
        """Test map execution with max_concurrent_tasks=1"""
        workflow = MockWorkflow()
        items = ["item1", "item2", "item3"]

        # This should error in this version - _parse_args doesn't handle non-dict items
        with pytest.raises(ValueError, match="Unsupported item type"):
            results = await workflow.map(items, max_concurrent_tasks=1)

    @pytest.mark.asyncio
    async def test_map_parallel_execution(self):
        """Test map execution with parallel processing"""
        workflow = MockWorkflow()
        items = [{"item": "item1"}, {"item": "item2"}]

        results = await workflow.map(items, max_concurrent_tasks=2)

        assert len(results) == 2
        assert "processed: item1" == results[0]
        assert "processed: item2" == results[1]

    @pytest.mark.asyncio
    async def test_map_empty_items(self):
        """Test map execution with empty items list"""
        workflow = MockWorkflow()
        results = await workflow.map([])
        assert results == []

    def test_parse_args_dict(self):
        """Test argument parsing for dictionary items"""
        workflow = MockWorkflow()
        args, kwargs = workflow._parse_args({"key": "value"}, extra="param")
        assert args == ()
        assert kwargs == {"key": "value", "extra": "param"}

    # TODO: Add tests for non-dictionary items
    # def test_parse_args_non_dict(self):
    #     """Test argument parsing for non-dictionary items"""
    #     workflow = MockWorkflow()
    #     args, kwargs = workflow._parse_args("string_item", extra="param")
    #     assert args == ("string_item",)
    #     assert kwargs == {"extra": "param"}


# TODO: Add tests for workflow CLI app
# class TestWorkflowApp:
#     """Test workflow CLI app functionality"""

#     def test_app_creation(self):
#         """Test that workflow app can be created"""
#         app = MockWorkflow.app
#         assert app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
