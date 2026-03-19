import time
import os
from typing import Any

import openai
from openai import OpenAI, AzureOpenAI

from .._types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float | None = 0.5,
        max_tokens: int | None = 4096,
        reasoning_effort: str | None = None,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        if os.environ.get("AZURE_OPENAI_ENDPOINT"):
            self.client = AzureOpenAI(
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("OPENAI_API_VERSION"),
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            )
        else:
            self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                completion_kwargs = {
                    "model": self.model,
                    "messages": message_list,
                }
                if self.max_tokens is not None:
                    completion_kwargs["max_tokens"] = self.max_tokens
                if self.temperature is not None:
                    completion_kwargs["temperature"] = self.temperature
                if self.reasoning_effort is not None:
                    completion_kwargs["reasoning_effort"] = self.reasoning_effort

                response = self.client.chat.completions.create(
                    **completion_kwargs,
                )
                content = response.choices[0].message.content
                if content is None or not str(content).strip():
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                    full_traces=response.model_dump() if hasattr(response, 'model_dump') else None,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception
