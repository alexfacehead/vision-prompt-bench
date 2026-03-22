import argparse
import os
import logging

from src.utils.constants_for_prompt_enhancement import (
    SYSTEM_MESSAGE_OPTIMIZER,
    STATIC_USER_QUESTION_INPUT,
    LLM_RESPONSE_FOR_CONTEXT,
    USER_INPUT_FOR_ENHANCEMENT,
)
from src.completions.completion_generator import ChatCompletionGenerator
from src.utils.helpers import update_message_with_new_prompt

logger = logging.getLogger(__name__)


def enhance_prompt(prompt: str, model: str = "gpt-4o-mini",
                   api_key: str = None) -> str:
    generator = ChatCompletionGenerator(
        temperature=0.1, api_key=api_key, model=model
    )

    user_input = update_message_with_new_prompt(USER_INPUT_FOR_ENHANCEMENT, prompt)

    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE_OPTIMIZER},
        {"role": "user", "content": STATIC_USER_QUESTION_INPUT},
        {"role": "assistant", "content": LLM_RESPONSE_FOR_CONTEXT},
        {"role": "user", "content": user_input},
    ]

    enhanced = generator.generate_completion(messages)
    logger.info("Prompt enhanced successfully")
    return enhanced


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Enhance an image generation prompt using AI."
    )
    parser.add_argument("prompt", type=str, help="The base prompt to enhance.")
    args = parser.parse_args()

    result = enhance_prompt(args.prompt)
    print(result)
