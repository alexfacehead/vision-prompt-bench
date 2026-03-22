import base64
import os
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class ImageGenerator:
    def __init__(self, api_key: Optional[str] = None,
                 model: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("IMAGE_MODEL", "gpt-image-1")
        self.client = OpenAI(api_key=self.api_key)

    def generate_image(self, prompt: str, size: str = "1024x1024",
                       quality: str = "high", n: int = 1):
        logger.info("Generating %d image(s) with model: %s", n, self.model)
        result = self.client.images.generate(
            model=self.model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=n,
        )
        return result

    def generate_and_save(self, prompt: str, output_dir: str, prefix: str,
                          size: str = "1024x1024", quality: str = "high",
                          n: int = 1) -> list[str]:
        result = self.generate_image(prompt, size, quality, n)
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []

        for i, image_data in enumerate(result.data):
            filename = f"{prefix}_{i}.png"
            path = os.path.join(output_dir, filename)

            if hasattr(image_data, "b64_json") and image_data.b64_json:
                img_bytes = base64.b64decode(image_data.b64_json)
                with open(path, "wb") as f:
                    f.write(img_bytes)
            elif hasattr(image_data, "url") and image_data.url:
                import httpx
                resp = httpx.get(image_data.url)
                with open(path, "wb") as f:
                    f.write(resp.content)
            else:
                logger.warning("No image data in response for index %d", i)
                continue

            saved_paths.append(path)
            logger.info("Saved image: %s", path)

        return saved_paths
