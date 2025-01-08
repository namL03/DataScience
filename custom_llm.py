from typing import Any, Dict, List, Optional, Generator
from langchain.llms.base import LLM
import requests
from pydantic import Field
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import json

class CustomLLM(LLM):
    """
    A custom LLM class for integrating with your API model.
    """

    base_url: str = Field(..., description="Base URL of the API endpoint.")
    model: str = Field(..., description="The model name to be used in the API.")
    headers: Dict[str, str] = Field(default_factory=lambda: {"Content-Type": "application/json"})

    def __init__(
        self,
        base_url: str,
        model: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """
        Initialize the CustomLLM class.

        :param base_url: Base URL of the API endpoint 
        :param model: The model name to be used in the API
        :param headers: Optional headers for the API requests.
        :param kwargs: Additional parameters for customization.
        """
        super().__init__(base_url=base_url, model=model, headers=headers, **kwargs)

    @property
    def _llm_type(self) -> str:
        return "custom_api_llm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        """
        Make a call to the API with the provided prompt and return the streaming response.

        :param prompt: The input text prompt.
        :param stop: Optional stop sequences for the response.
        :param kwargs: Additional parameters for the API call (e.g., max_tokens).
        :return: A generator yielding chunks of the response text.
        """
        data = {
            "temperature": 0.75,
            "messages": [{"role": "user", "content": prompt}],
            "model": self.model,
            "stream": True,  # Enable streaming
        }

        try:
            with requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json=data,
                stream=True  # Enable response streaming
            ) as response:
                response.raise_for_status()

                # Process the stream of responses
                for chunk in response.iter_lines(decode_unicode=True):
                    if chunk:  # Skip keep-alive newlines
                        # Parse the chunk as JSON
                        chunk_data = chunk.strip()
                        try:
                            # Strip the "data:" prefix if present
                            if chunk_data.startswith("data:"):
                                chunk_data = chunk_data[len("data:"):].strip()

                            # Skip the `[DONE]` chunk
                            if chunk_data == "[DONE]":
                                break

                            # Parse the chunk as JSON
                            json_chunk = json.loads(chunk_data)
                            content = json_chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")

                            if content:
                                # Notify the callback manager if provided
                                if run_manager:
                                    run_manager.on_llm_new_token(content)

                                # Yield the content chunk
                                yield content

                        except json.JSONDecodeError as parse_error:
                            raise ValueError(f"Failed to parse chunk as JSON: {chunk_data}") from parse_error

        except requests.exceptions.RequestException as e:
            raise ValueError(f"API request error: {e}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Unexpected API response format: {e}")


# Example usage
BASE_URL = "http://10.124.68.81:10000"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"

custom_llm = CustomLLM(base_url=BASE_URL, model=MODEL_NAME, headers=HEADERS)

# prompt = "Bạn có biết gì về ortools không?"
# for token in custom_llm._call(prompt):
#     print(token, end="")  # Stream tokens to the console in real time

