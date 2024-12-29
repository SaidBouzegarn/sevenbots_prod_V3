from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List
from tenacity import retry, stop_after_attempt
import os
import asyncio
import logging
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_fireworks import ChatFireworks
from langchain_community.chat_models import ChatOllama
from langchain_core.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import List, Optional, Dict, Any, Union, Callable, Sequence, TypedDict, Annotated, Literal, Type
from langchain_core.language_models import BaseLanguageModel

import tiktoken
from .llm_models import (
    OPENAI_MODELS,
    MISTRAL_MODELS,
    COHERE_MODELS,
    GROQ_MODELS,
    OLLAMA_MODELS,
    NVIDIA_MODELS,
    ANTHROPIC_MODELS,
    FIREWORKS_MODELS,
)

import tiktoken

logger = logging.getLogger(__name__)
# Initialize logger

#### LLM Models ####
#cach this function
#@lru_cache(maxsize=10)
async def get_llm(llm_name: str = "gpt-4o-mini", llm_params: Dict[str, Any] = {"temperature": 0.1618}, tools: List[BaseTool] = []) -> BaseLanguageModel:
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
        return llm.bind_tools(tools)

    return llm


############ Select likely URLs ############    

class URLListResponse(BaseModel):
    likely_urls : List[str] 

# Cached tokenizer to improve performance
@lru_cache(maxsize=10)
def get_encoding(model: str):
    """Cached function to get tokenizer encoding."""
    return tiktoken.encoding_for_model(model)

async def async_truncate_prompt(
    prompt: str, 
    max_tokens: int = 70000, 
    model: str = "gpt-4o",
    truncation_strategy: str = 'end'
) -> str:
    """
    Asynchronously truncate a prompt with different truncation strategies.
    
    Args:
        prompt (str): The input prompt to truncate
        max_tokens (int): Maximum number of tokens allowed (default: 70000)
        model (str): The model to use for tokenization (default: "gpt-4o")
        truncation_strategy (str): How to truncate ('start', 'end', 'middle')
    
    Returns:
        str: Truncated prompt
    """
    try:
        # Use cached encoding
        encoding = get_encoding(model)
        
        # Encode the prompt
        tokens = encoding.encode(prompt)
        
        # If tokens are within limit, return original prompt
        if len(tokens) <= max_tokens:
            return prompt
        
        # Truncate based on strategy
        if truncation_strategy == 'start':
            truncated_tokens = tokens[-max_tokens:]
        elif truncation_strategy == 'end':
            truncated_tokens = tokens[:max_tokens]
        else:  # middle strategy (default)
            mid = len(tokens) // 2
            start = mid - (max_tokens // 2)
            end = start + max_tokens
            truncated_tokens = tokens[max(0, start):end]
        
        # Decode back to string
        truncated_prompt = encoding.decode(truncated_tokens)
        
        # Log truncation details
        logger.warning(
            f"Prompt truncated: "
            f"Original tokens: {len(tokens)}, "
            f"Truncated tokens: {len(truncated_tokens)}, "
            f"Strategy: {truncation_strategy}"
        )
        
        return truncated_prompt
    
    except Exception as e:
        logger.error(f"Error in async prompt truncation: {str(e)}")
        return prompt  # Fallback to original prompt

@retry(stop=stop_after_attempt(3))
async def select_likely_URLS(prompt, llm_name: str = "gpt-4o-mini", llm_config: Dict[str, Any] = {"temperature": 0.1618}):
    """Detect good to go links from bad links."""
    try:
        logger.info("Starting URL selection process")
        
        # Asynchronously truncate prompt
        truncated_prompt = await async_truncate_prompt(
            prompt, 
            max_tokens=70000, 
            model="gpt-4o-mini",
        )
        
        # Construct LLM using get_llm with configuration
        llm = await get_llm(llm_name, llm_config)
        structured_llm = llm.with_structured_output(URLListResponse)
        
        logger.debug(f"Processing truncated prompt: {truncated_prompt[:100]}...")
        
        response = structured_llm.invoke([
            {"role": "system", "content": "you are an assistant that is tasked with selecting a list of URLs that meet the criteria for likely news articles the most and are not suspected bot traps nor user related nor categories webpages."},
            {"role": "user", "content": truncated_prompt}
        ])

        logger.info(f"Successfully selected {len(response.likely_urls)} URLs")
        return response
        
    except Exception as e:
        logger.error(f"URL selection failed: {str(e)}", exc_info=True)
        raise

############ Detect login url  ############   

class FormFieldLoginUrl(BaseModel):
    sucess : bool
    login_url: str
    comment: str
@retry(stop=stop_after_attempt(3))
async def detect_login_url(prompt, llm_name: str = "gpt-4o-mini", llm_config: Dict[str, Any] = {"temperature": 0.1618}):
    try:
        logger.info("Starting login URL detection")
        
        # Truncate prompt
        truncated_prompt = await async_truncate_prompt(
            prompt, 
            max_tokens=70000, 
            model="gpt-4o",
        )
        
        # Construct LLM using get_llm with configuration
        llm = await get_llm(llm_name, llm_config)
        structured_llm = llm.with_structured_output(FormFieldLoginUrl)

        response = structured_llm.invoke([
            {"role": "system", "content": "Detect login url in the a list of urls"},
            {"role": "user", "content": truncated_prompt}
        ])
        logger.info(f"looking for login url in {truncated_prompt}")
        logger.info(f"Login URL detected: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Login URL detection failed: {str(e)}", exc_info=True)
        raise

############ Detect css selectors  ############   

class FormFieldInfoCredentials(BaseModel):
    sucess : bool
    username_selector: str
    password_selector: str
    submit_button_selector: str
    comment: str

@retry(stop=stop_after_attempt(3))
async def detect_selectors(prompt, llm_name: str = "gpt-4o-mini", llm_config: Dict[str, Any] = {"temperature": 0.1618}):
    try:
        logger.info("Starting CSS selector detection")
        
        # Truncate prompt
        truncated_prompt = await async_truncate_prompt(
            prompt, 
            max_tokens=70000, 
            model="gpt-4o",
        )
        
        # Construct LLM using get_llm with configuration
        llm = await get_llm(llm_name, llm_config)
        structured_llm = llm.with_structured_output(FormFieldInfoCredentials)

        response = structured_llm.invoke([
            {"role": "system", "content": "Detect username field, password fields, and submit button css selectors in the cleaned HTML"},
            {"role": "user", "content": truncated_prompt}
        ])
        logger.info(f"looking for css selectors in {truncated_prompt}")
        logger.info(f"CSS selectors detected successfully : {response}")
        logger.debug(f"Detected selectors: {response}")
        return response
        
    except Exception as e:
        logger.error(f"Selector detection failed: {str(e)}", exc_info=True)
        raise

############ Classify page and Extract news article ############   

class FormFieldNewsArticleExtractor(BaseModel):
    classification: bool
    title: str
    author: str
    body: str
    date_published: str
    comment: str

@retry(stop=stop_after_attempt(3))
async def classify_and_extract_news_article(prompt, llm_name: str = "gpt-4o-mini", llm_config: Dict[str, Any] = {"temperature": 0.1618}):
    try:
        logger.info("Starting article classification and extraction")
        
        # Truncate prompt
        truncated_prompt = await async_truncate_prompt(
            prompt, 
            max_tokens=70000, 
            model="gpt-4o-mini",
        )
        
        # Construct LLM using get_llm with configuration
        llm = await get_llm(llm_name, llm_config)
        structured_llm = llm.with_structured_output(FormFieldNewsArticleExtractor)

        response = structured_llm.invoke([
            {"role": "system", "content": "you are an assistant that is tasked with classifying a webpage as either a 'full Article webpage' or 'Not an Article webpage' based on its cleaned HTML content and extracting the full article content when it is an article webpage."},
            {"role": "user", "content": truncated_prompt}
        ])

        
        if response.classification:
            logger.info(f"Article extracted successfully: {response.title}")
            logger.debug(f"Article details: {response}")
        else:
            logger.info("Page classified as non-article")
            
        return response
        
    except Exception as e:
        logger.error(f"Article extraction failed: {str(e)}", exc_info=True)
        raise

############ Filter Relevant Links ############   

class RelevantLinksResponse(BaseModel):
    relevant_urls: List[str]

@retry(stop=stop_after_attempt(3))
async def filter_relevant_links(prompt, llm_name: str = "gpt-4o-mini", llm_config: Dict[str, Any] = {"temperature": 0.1618}):
    """
    Filter URLs to identify those relevant to healthcare or medical technology.
    
    Args:
        urls (List[str]): List of URLs to evaluate
    
    Returns:
        RelevantLinksResponse: List of relevant URLs
    """
    try:
        logger.info("Starting relevant links filtering process")
        
        # Prepare the prompt by formatting URLs
        
        # Truncate prompt if needed
        truncated_prompt = await async_truncate_prompt(
            prompt, 
            max_tokens=70000, 
            model="gpt-4o-mini",
        )
        
        # Construct LLM using get_llm with configuration
        llm = await get_llm(llm_name, llm_config)
        structured_llm = llm.with_structured_output(RelevantLinksResponse)
        
        response = structured_llm.invoke([
            {"role": "system", "content": "You are tasked with evaluating a list of URLs to identify those relevant to user interest."},
            {"role": "user", "content": truncated_prompt}
        ])

        logger.info(f"Successfully filtered {len(response.relevant_urls)} relevant URLs")
        return response
        
    except Exception as e:
        logger.error(f"Relevant links filtering failed: {str(e)}", exc_info=True)
        raise

############ Check Article Relevance ############   

class ArticleRelevanceResponse(BaseModel):
    is_relevant: bool
    relevance_reason: str

@retry(stop=stop_after_attempt(3))
async def is_article_relevant(prompt, llm_name: str = "gpt-4o-mini", llm_config: Dict[str, Any] = {"temperature": 0.1618}):
    """
    Determine if an article is relevant to healthcare or medical technology.
    
    Args:
        article (str): Full text of the article to evaluate
    
    Returns:
        ArticleRelevanceResponse: Relevance assessment
    """
    try:
        logger.info("Starting article relevance assessment")
        
        # Truncate article if needed
        truncated_prompt = await async_truncate_prompt(
            prompt, 
            max_tokens=70000, 
            model="gpt-4o-mini",
        )
        
        # Construct LLM using get_llm with configuration
        llm = await get_llm(llm_name, llm_config)
        structured_llm = llm.with_structured_output(ArticleRelevanceResponse)

        response = structured_llm.invoke([
            {"role": "system", "content": "Evaluate the relevance of an article to user interest."},
            {"role": "user", "content": truncated_prompt}
        ])

        logger.info(f"Article relevance assessment complete: {response.is_relevant}")
        return response
        
    except Exception as e:
        logger.error(f"Article relevance assessment failed: {str(e)}", exc_info=True)
        raise

