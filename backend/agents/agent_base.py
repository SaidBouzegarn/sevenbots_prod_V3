from typing import List, Optional, Dict, Any, Union, Callable, Sequence, TypedDict, Annotated, Literal, Type
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain.schema.runnable import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore
from PIL import Image
import io
import os
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_fireworks import ChatFireworks
from langchain_community.chat_models import ChatOllama
from datetime import datetime
import json
from pathlib import Path

set_llm_cache(InMemoryCache())

# Update the path resolution to be relative to the current file
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
llm_models_path = project_root / "app" / "Data" / "llm_models.json"

# Load the LLM models from the JSON file
try:
    with open(llm_models_path, "r") as f:
        LLM_MODELS = json.load(f)
except FileNotFoundError:
    # Provide default model lists if file is not found
    LLM_MODELS = {
        "OPENAI_MODELS": ["gpt-4", "gpt-3.5-turbo"],
        "MISTRAL_MODELS": ["mistral-tiny", "mistral-small", "mistral-medium"],
        "COHERE_MODELS": ["command", "command-light"],
        "GROQ_MODELS": ["mixtral-8x7b-32768", "llama2-70b-4096"],
        "VERTEXAI_MODELS": ["gemini-pro"],
        "OLLAMA_MODELS": ["llama2", "mistral"],
        "NVIDIA_MODELS": ["mixtral-8x7b"],
        "ANTHROPIC_MODELS": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
        "FIREWORKS_MODELS": ["llama-v2-7b", "llama-v2-13b"]
    }
    print(f"Warning: Could not find {llm_models_path}. Using default model lists.")

# Use the imported model lists
OPENAI_MODELS = LLM_MODELS["OPENAI_MODELS"]
MISTRAL_MODELS = LLM_MODELS["MISTRAL_MODELS"]
COHERE_MODELS = LLM_MODELS["COHERE_MODELS"]
GROQ_MODELS = LLM_MODELS["GROQ_MODELS"]
VERTEXAI_MODELS = LLM_MODELS["VERTEXAI_MODELS"]
OLLAMA_MODELS = LLM_MODELS["OLLAMA_MODELS"]
NVIDIA_MODELS = LLM_MODELS["NVIDIA_MODELS"]
ANTHROPIC_MODELS = LLM_MODELS["ANTHROPIC_MODELS"]
FIREWORKS_MODELS = LLM_MODELS["FIREWORKS_MODELS"]

class BaseAgent:
    def __init__(
        self,
        name: str,
        llm: str,
        llm_params: Dict[str, Any],
        assistant_llm: str,
        assistant_llm_params: Dict[str, Any],
        tools: List[BaseTool] = None,
        system_message: Optional[str] = None,
        debug: bool = False,
        **kwargs: Any
    ):
        self.name = name
        self.tools = tools
        self.llm = self._construct_llm(llm, llm_params)
        self.assistant_llm = self._construct_llm(assistant_llm, assistant_llm_params)
        self.system_message = system_message
        self.debug = debug
        self.kwargs = kwargs

    def _construct_llm(self, llm_name: str, llm_params: Dict[str, Any], tools: List[BaseTool] = None) -> BaseLanguageModel:
        """Construct the appropriate LLM based on the input string and parameters."""
        if llm_name in OPENAI_MODELS:
            llm = ChatOpenAI(model_name=llm_name, **llm_params)
        elif llm_name in MISTRAL_MODELS:
            llm = ChatMistralAI(model=llm_name, **llm_params)
        elif llm_name in COHERE_MODELS:
            llm = ChatCohere(model=llm_name, **llm_params)
        elif llm_name in GROQ_MODELS:
            llm = ChatGroq(model=llm_name, **llm_params)
        elif llm_name in OLLAMA_MODELS:
            llm = ChatOllama(model=llm_name, **llm_params)
        elif llm_name in NVIDIA_MODELS:
            llm = ChatNVIDIA(model=llm_name, **llm_params)
        elif llm_name in ANTHROPIC_MODELS:
            llm = ChatAnthropic(model=llm_name, **llm_params)
        elif llm_name in FIREWORKS_MODELS:
            llm = ChatFireworks(model=llm_name, **llm_params)
        else:
            raise ValueError(f"Unsupported model: {llm_name}")
        
        if tools:
            return llm.bind_tools(self.tools)

        return llm

    # Old version of create_message not working properly
    def create_message_old_version(self, content, agent_name: str = None) -> AIMessage:
        """Dynamically create a new message class for a specific agent."""
        if agent_name is None:
            agent_name = self.name
        message = type(f"{agent_name}Message", (BaseMessage,), {
            "content": content,
            "__doc__": f"A message from the {agent_name} Agent.",
            "agent_name": agent_name,
            "timestamp": datetime.now().isoformat()
        })
        return message


    def _create_graph(self) -> StateGraph:
        raise NotImplementedError("Subclasses must implement this method")

    def run(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state = self.state_schema(**inputs)
        default_config = {
            "thread_id": "default_thread",
            "checkpoint_ns": "default_namespace",
            "checkpoint_id": "default_checkpoint"
        }
        if config:
            default_config.update(config)
        return self.graph.invoke(state, default_config)

    def stream(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        state = self.state_schema(**inputs)
        default_config = {
            "thread_id": "default_thread",
            "checkpoint_ns": "default_namespace",
            "checkpoint_id": "default_checkpoint"
        }
        if config:
            default_config.update(config)
        return self.graph.stream(state, default_config)

    def arun(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        state = self.state_schema(**inputs)
        default_config = {
            "thread_id": "default_thread",
            "checkpoint_ns": "default_namespace",
            "checkpoint_id": "default_checkpoint"
        }
        if config:
            default_config.update(config)
        return self.graph.ainvoke(state, default_config)

    def astream(self, inputs: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        state = self.state_schema(**inputs)
        default_config = {
            "thread_id": "default_thread",
            "checkpoint_ns": "default_namespace",
            "checkpoint_id": "default_checkpoint"
        }
        if config:
            default_config.update(config)
        return self.graph.astream(state, default_config)

    @staticmethod
    def get_latest_message_list(list1: List[AIMessage], list2: List[AIMessage]) -> List[AIMessage]:
        """
        Compare two lists of messages and return the list with the most recent message.
        If both lists are empty, return an empty list.
        If one list is empty and the other is not, return the non-empty list.
        """
        if not list1 and not list2:
            return []
        if not list1:
            return list2
        if not list2:
            return list1
        
        last_msg1 = list1[-1]
        last_msg2 = list2[-1]
        
        time1 = datetime.fromisoformat(last_msg1.timestamp)
        time2 = datetime.fromisoformat(last_msg2.timestamp)
        
        return list1 if time1 > time2 else list2
