import os
import logging
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ChatCompletionGenerator:
    def __init__(self, temperature: float = 0.33, api_key: Optional[str] = None,
                 model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.client = OpenAI(api_key=self.api_key)

    def generate_completion(self, messages: List[dict],
                            temperature: Optional[float] = None,
                            model: Optional[str] = None) -> str:
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        logger.info("Generating completion with model: %s", model)

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4000,
            temperature=temperature,
        )
        logger.info("Completion successful")
        return response.choices[0].message.content
