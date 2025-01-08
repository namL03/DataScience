from typing import Any, Dict, List, Optional, Generator
from langchain.llms.base import LLM
import requests
from pydantic import Field
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
import json
from langchain.prompts import PromptTemplate
import logging
from custom_llm import CustomLLM
from langchain.chains.conversation.memory import ConversationBufferMemory

logging.basicConfig(level=logging.INFO)

BASE_URL = "http://10.124.68.81:10000"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8"


custom_llm = CustomLLM(base_url=BASE_URL, model=MODEL_NAME, headers=HEADERS)

template = """Question: {question}
Answer the question in Vietnamese
Answer:
"""
prompt = PromptTemplate.from_template(template)

class ChatBot():
    def __init__(self, llm=custom_llm, default_prompt=prompt):
        self.llm = llm
        self.default_prompt = default_prompt
        self.chain = default_prompt | llm.bind(skip_prompt=True)

    def generate_answer_stream(self, question: str) -> Generator[str, None, None]:
        logging.info(f"Generating streaming answer for question: {question}")
        for chunk in self.chain.stream({"question": question}):
            if isinstance(chunk, str):
                yield chunk
            elif isinstance(chunk, dict) and "text" in chunk:
                yield chunk["text"]
            else:
                logging.warning(f"Unexpected chunk type: {type(chunk)}")
                yield ""

chatbot = ChatBot()

