from abc import ABC, abstractmethod
import requests
import time
from openai import OpenAI
import google.generativeai as genai
import os
import logging

logger = logging.getLogger("clipify.ai_providers")

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    def get_response(self, prompt, retry_count=3):
        """Get response from AI provider"""
        pass

class GoogleGeminiProvider(AIProvider):
    """Google Gemini provider implementation (Gemini API)"""

    AVAILABLE_MODELS = {
        "default": "gemini-1.5-flash",
        "gemini-1.5-flash": "gemini-1.5-flash",
        "gemini-1.5-pro": "gemini-1.5-pro",
    }

    def __init__(self, api_key, model="default", max_tokens=2048, temperature=0.7):
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        self.model = self.AVAILABLE_MODELS.get(model, model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache = {}
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model)

    def get_response(self, prompt, retry_count=3):
        """Get response from Gemini with retry & caching; return OpenAI-like dict."""
        if prompt in self.cache:
            logger.info("LLM cache hit: provider=gemini model=%s", self.model)
            return self.cache[prompt]

        last_err = None
        for _ in range(retry_count):
            try:
                logger.info("LLM request: provider=gemini model=%s prompt_chars=%s", self.model, len(prompt or ""))
                resp = self.client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    },
                )
                text = (resp.text or "").strip()
                if not text:
                    raise ValueError("Invalid AI response")

                result = {"choices": [{"message": {"content": text}}]}
                self.cache[prompt] = result
                logger.info("LLM response: provider=gemini model=%s content_chars=%s", self.model, len(text))
                return result

            except Exception as e:
                last_err = e
                time.sleep(1)

        raise last_err if last_err else RuntimeError("Unknown error in GoogleGeminiProvider")

class HyperbolicAI(AIProvider):
    """Hyperbolic AI provider implementation"""
    
    AVAILABLE_MODELS = {
        "deepseek-v3": "deepseek-ai/DeepSeek-V3",
        "deepseek-v2": "deepseek-ai/DeepSeek-V2",
        "default": "deepseek-ai/DeepSeek-V3"
    }
    
    def __init__(self, api_key, model="default", max_tokens=5012, temperature=0.7):
        self.url = "https://api.hyperbolic.xyz/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache = {}

    def get_response(self, prompt, retry_count=3):
        """Get response from Hyperbolic AI with retry mechanism and caching"""
        if prompt in self.cache:
            return self.cache[prompt]
            
        for attempt in range(retry_count):
            try:
                data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": 0.9
                }
                
                response = requests.post(self.url, headers=self.headers, json=data)
                result = response.json()
                
                if 'choices' in result:
                    self.cache[prompt] = result
                    return result
                    
            except Exception as e:
                if attempt == retry_count - 1:
                    raise e
                time.sleep(1)
                
        return None

class OpenAIProvider(AIProvider):
    """OpenAI provider implementation (OpenAI SDK v1.x)"""

    AVAILABLE_MODELS = {
        "gpt-4": "gpt-4",
        "gpt-3.5-turbo": "gpt-3.5-turbo",
        "gpt-4-turbo": "gpt-4o-mini",  # modern lightweight 4-class model
        "default": "gpt-4o-mini"
    }

    def __init__(self, api_key, model="default", max_tokens=2048, temperature=0.7):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        # Normalize model name: allow keys from AVAILABLE_MODELS or a raw model string
        resolved_model = self.AVAILABLE_MODELS.get(model, model)

        self.client = OpenAI(api_key=api_key)
        self.model = resolved_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache = {}

    def get_response(self, prompt, retry_count=3):
        """Get response from OpenAI with retry mechanism and caching."""
        if prompt in self.cache:
            return self.cache[prompt]

        last_err = None
        for _ in range(retry_count):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

                # Safely read content
                content = (
                    resp.choices[0].message.content.strip()
                    if resp and resp.choices and resp.choices[0].message and resp.choices[0].message.content
                    else ""
                )

                if not content:
                    raise ValueError("Invalid AI response")

                # Convert to the structure the rest of the app expects
                result = {
                    "choices": [{
                        "message": {
                            "content": content
                        }
                    }]
                }

                self.cache[prompt] = result
                return result

            except Exception as e:
                last_err = e
                time.sleep(1)

        # If all retries failed, re-raise the last error
        raise last_err if last_err else RuntimeError("Unknown error in OpenAIProvider")

class AnthropicProvider(AIProvider):
    """Anthropic (Claude) provider implementation"""
    
    AVAILABLE_MODELS = {
        "claude-3-opus": "claude-3-opus-20240229",
        "claude-3-sonnet": "claude-3-sonnet-20240229",
        "claude-3-haiku": "claude-3-haiku-20240307",
        "default": "claude-3-sonnet-20240229"
    }
    
    def __init__(self, api_key, model="default", max_tokens=2048, temperature=0.7):
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache = {}

    def get_response(self, prompt, retry_count=3):
        """Get response from Claude with retry mechanism and caching"""
        if prompt in self.cache:
            return self.cache[prompt]
            
        for attempt in range(retry_count):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                # Convert Anthropic response format to match Hyperbolic format
                result = {
                    "choices": [{
                        "message": {
                            "content": response.content[0].text
                        }
                    }]
                }
                
                self.cache[prompt] = result
                return result
                
            except Exception as e:
                if attempt == retry_count - 1:
                    raise e
                time.sleep(1)
                
        return None

class OpenRouterProvider(AIProvider):
    """OpenRouter provider implementation - compatible with OpenAI API"""
    
    AVAILABLE_MODELS = {
        "default": "auto",
        "auto": "auto",
        "auto_balanced": "auto_balanced",
        "auto_free": "auto_free",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4.1": "openai/gpt-4.1",
        "o3": "openai/o3",
        "o4-mini": "openai/o4-mini",
        "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
        "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "deepseek-chat": "deepseek/deepseek-chat",
        "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
        "qwen-2.5-72b": "qwen/qwen-2.5-72b-instruct",
    }

    # Best premium models ranked by reasoning/content quality for clip strategy.
    AUTO_BEST_MODELS = [
        "anthropic/claude-sonnet-4",
        "openai/o3",
        "openai/gpt-4.1",
        "google/gemini-2.5-pro",
        "openai/gpt-4o",
    ]
    
    # Best free models ranked by capability (used when model="auto_free")
    AUTO_BEST_FREE_MODELS = [
        "qwen/qwen-2.5-72b-instruct",  # Best free: large, capable
        "deepseek/deepseek-chat",       # Good free option
        "meta-llama/llama-3.1-70b-instruct",
    ]

    AUTO_BALANCED_MODELS = [
        "openai/gpt-4o",
        "anthropic/claude-3.7-sonnet",
        "deepseek/deepseek-chat",
        "openai/gpt-4o-mini",
    ]
    
    BASE_URL = "https://openrouter.ai/api/v1"

    @staticmethod
    def _is_model_not_found_error(err: Exception) -> bool:
        msg = str(err or "").lower()
        status = getattr(err, "status_code", None)
        return (
            status == 404
            or "no endpoints found" in msg
            or "error code: 404" in msg
            or "not found" in msg
        )
    
    def __init__(self, api_key, model="default", max_tokens=2048, temperature=0.7):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set.")

        model_in = (model or "default").strip().lower()
        self.auto_mode = None
        if model_in in ("default", "auto", "auto (best quality)"):
            self.auto_mode = "quality"
            model = self.AUTO_BEST_MODELS[0]
        elif model_in in ("auto_balanced", "auto (balanced)"):
            self.auto_mode = "balanced"
            model = self.AUTO_BALANCED_MODELS[0]
        elif model_in in ("auto_free", "auto (best free)"):
            self.auto_mode = "free"
            model = self.AUTO_BEST_FREE_MODELS[0]
        
        resolved_model = self.AVAILABLE_MODELS.get(model, model)
        
        import logging
        logging.getLogger("clipify").info("OpenRouterProvider: requested model=%s, resolved=%s", model, resolved_model)
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/ai-cfia/clipify",
                "X-Title": "Clipify"
            }
        )
        self.model = resolved_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache = {}
    
    def get_response(self, prompt, retry_count=3):
        """Get response from OpenRouter with retry mechanism and caching."""
        if prompt in self.cache:
            logger.info("LLM cache hit: provider=openrouter model=%s", self.model)
            return self.cache[prompt]
        
        # Try requested model first. If unavailable on this account/region,
        # fail over across premium then free-capable alternatives.
        if self.auto_mode == "quality":
            fallback_pool = self.AUTO_BEST_MODELS + self.AUTO_BEST_FREE_MODELS
        elif self.auto_mode == "balanced":
            fallback_pool = self.AUTO_BALANCED_MODELS + self.AUTO_BEST_FREE_MODELS
        elif self.auto_mode == "free":
            fallback_pool = self.AUTO_BEST_FREE_MODELS + ["openai/gpt-4o-mini"]
        else:
            fallback_pool = self.AUTO_BEST_MODELS + self.AUTO_BALANCED_MODELS + self.AUTO_BEST_FREE_MODELS + ["openai/gpt-4o-mini"]

        model_candidates = [self.model] + [m for m in fallback_pool if m != self.model]

        last_err = None
        for model_name in model_candidates:
            for _ in range(retry_count):
                try:
                    logger.info("LLM request: provider=openrouter model=%s prompt_chars=%s", model_name, len(prompt or ""))
                    resp = self.client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                    )

                    content = (
                        resp.choices[0].message.content.strip()
                        if resp and resp.choices and resp.choices[0].message and resp.choices[0].message.content
                        else ""
                    )

                    if not content:
                        raise ValueError("Invalid AI response")

                    result = {
                        "choices": [{
                            "message": {
                                "content": content
                            }
                        }],
                        "_llm_meta": {
                            "provider": "openrouter",
                            "model_requested": self.model,
                            "model_used": model_name,
                            "auto_mode": self.auto_mode,
                        },
                    }

                    usage = getattr(resp, "usage", None)
                    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
                    logger.info("LLM response: provider=openrouter model=%s total_tokens=%s content_chars=%s", model_name, total_tokens, len(content))

                    self.model = model_name
                    self.cache[prompt] = result
                    return result

                except Exception as e:
                    last_err = e
                    if self._is_model_not_found_error(e):
                        logger.warning("OpenRouter model unavailable, trying fallback: model=%s error=%s", model_name, e)
                        break
                    time.sleep(1)

        raise last_err if last_err else RuntimeError("Unknown error in OpenRouterProvider")


class OllamaProvider(AIProvider):
    """Ollama local AI provider implementation"""
    
    AVAILABLE_MODELS = {
        "llama2": "llama2",
        "mistral": "mistral",
        "codellama": "codellama",
        "default": "llama2"
    }
    
    def __init__(self, api_key, model="default", max_tokens=2048, temperature=0.7):
        """Initialize Ollama provider
        
        Note: api_key is ignored since Ollama runs locally
        """
        self.url = "http://localhost:11434/api/chat"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.cache = {}

    def get_response(self, prompt, retry_count=3):
        """Get response from Ollama with retry mechanism and caching"""
        if prompt in self.cache:
            return self.cache[prompt]
            
        for attempt in range(retry_count):
            try:
                data = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens
                    }
                }
                
                response = requests.post(self.url, json=data)
                response_json = response.json()
                
                # Convert Ollama response format to match Hyperbolic format
                result = {
                    "choices": [{
                        "message": {
                            "content": response_json.get("message", {}).get("content", "")
                        }
                    }]
                }
                
                self.cache[prompt] = result
                return result
                
            except Exception as e:
                if attempt == retry_count - 1:
                    raise e
                time.sleep(1)
                
        return None

def get_ai_provider(
    provider_name: str, 
    api_key: str, 
    model: str = "default",
    max_tokens: int = None,
    temperature: float = None
) -> AIProvider:
    """
    Factory function to get AI provider instance
    
    Args:
        provider_name: Name of the AI provider
        api_key: API key for the provider (not needed for Ollama)
        model: Model name to use (provider-specific)
        max_tokens: Maximum number of tokens in response (optional)
        temperature: Temperature for response generation (optional)
    """
    providers = {
        "hyperbolic": (HyperbolicAI, 5012, 0.7),
        "openai": (OpenAIProvider, 5048, 0.7),
        "openrouter": (OpenRouterProvider, 5048, 0.7),
        "anthropic": (AnthropicProvider, 5048, 0.7),
        "ollama": (OllamaProvider, 2048, 0.7),
        "google": (GoogleGeminiProvider, 2048, 0.7),
        "gemini": (GoogleGeminiProvider, 2048, 0.7),
    }

    provider_info = providers.get(provider_name.lower())
    if not provider_info:
        raise ValueError(f"Unknown AI provider: {provider_name}. Available providers: {', '.join(providers.keys())}")
    
    provider_class, default_max_tokens, default_temp = provider_info

    logger.info(
        "Creating AI provider instance: provider=%s model=%s max_tokens=%s temperature=%s",
        provider_name,
        model,
        max_tokens if max_tokens is not None else default_max_tokens,
        temperature if temperature is not None else default_temp,
    )
    
    return provider_class(
        api_key, 
        model,
        max_tokens=max_tokens if max_tokens is not None else default_max_tokens,
        temperature=temperature if temperature is not None else default_temp
    ) 