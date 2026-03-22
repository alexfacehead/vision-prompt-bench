import json
import os
import logging
from datetime import datetime

from src.easy_prompt_enhancer.prompt_enhancer import enhance_prompt
from src.image_generation.image_generator import ImageGenerator
from src.metrics.metric_calculations import (
    compute_single_image_quality,
    compute_similarity_metrics,
)
from src.evaluation_metrics.statistical_analysis import (
    run_statistical_analysis,
    format_report,
)

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, images_per_prompt: int = 3, image_model: str = None,
                 chat_model: str = None, image_size: str = "1024x1024",
                 image_quality: str = "high", output_dir: str = "output"):
        self.images_per_prompt = images_per_prompt
        self.image_size = image_size
        self.image_quality = image_quality
        self.chat_model = chat_model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
        self.output_dir = output_dir
        self._image_model = image_model
        self._image_generator = None

    @property
    def image_generator(self):
        if self._image_generator is None:
            self._image_generator = ImageGenerator(model=self._image_model)
        return self._image_generator

    def run(self, prompts: list[str]) -> dict:
        """
        Full pipeline: for each prompt, enhance it, generate images for both
        base and enhanced versions, compute quality metrics, and run
        statistical analysis.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"run_{timestamp}")

        all_base_quality = []
        all_improved_quality = []
        all_similarity = {}
        prompt_results = []

        for i, base_prompt in enumerate(prompts):
            logger.info("Processing prompt %d/%d: %s", i + 1, len(prompts),
                        base_prompt[:60])

            # Enhance the prompt
            enhanced_prompt = enhance_prompt(base_prompt, model=self.chat_model)
            logger.info("Enhanced prompt: %s", enhanced_prompt[:80])

            # Save prompts
            prompt_dir = os.path.join(run_dir, "prompts")
            os.makedirs(prompt_dir, exist_ok=True)
            with open(os.path.join(prompt_dir, f"prompt_{i}_base.txt"), "w") as f:
                f.write(base_prompt)
            with open(os.path.join(prompt_dir, f"prompt_{i}_enhanced.txt"), "w") as f:
                f.write(enhanced_prompt)

            # Generate images
            image_dir = os.path.join(run_dir, "images", f"prompt_{i}")

            base_paths = self.image_generator.generate_and_save(
                base_prompt, image_dir, prefix="base",
                size=self.image_size, quality=self.image_quality,
                n=self.images_per_prompt,
            )
            improved_paths = self.image_generator.generate_and_save(
                enhanced_prompt, image_dir, prefix="improved",
                size=self.image_size, quality=self.image_quality,
                n=self.images_per_prompt,
            )

            # Compute no-reference quality metrics for each image
            base_quality = [compute_single_image_quality(p) for p in base_paths]
            improved_quality = [compute_single_image_quality(p) for p in improved_paths]

            all_base_quality.extend(base_quality)
            all_improved_quality.extend(improved_quality)

            # Compute similarity between paired base/improved images
            n_pairs = min(len(base_paths), len(improved_paths))
            for j in range(n_pairs):
                sim = compute_similarity_metrics(base_paths[j], improved_paths[j])
                for key, val in sim.items():
                    all_similarity.setdefault(key, []).append(val)

            prompt_results.append({
                "prompt_index": i,
                "base_prompt": base_prompt,
                "enhanced_prompt": enhanced_prompt,
                "base_quality": base_quality,
                "improved_quality": improved_quality,
                "n_base_images": len(base_paths),
                "n_improved_images": len(improved_paths),
            })

        # Run statistical analysis
        analysis = run_statistical_analysis(all_base_quality, all_improved_quality)
        report = format_report(analysis, all_similarity)

        # Save results
        metrics_dir = os.path.join(run_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        results = {
            "timestamp": timestamp,
            "n_prompts": len(prompts),
            "images_per_prompt": self.images_per_prompt,
            "image_model": self.image_generator.model,
            "chat_model": self.chat_model,
            "prompt_results": prompt_results,
            "analysis": [_serialize(a) for a in analysis],
            "similarity": {k: v for k, v in all_similarity.items()},
        }

        with open(os.path.join(metrics_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

        with open(os.path.join(run_dir, "report.txt"), "w") as f:
            f.write(report)

        return {"report": report, "analysis": analysis, "run_dir": run_dir}

    def run_from_directories(self, base_dir: str, improved_dir: str) -> dict:
        """
        Evaluate pre-existing images in two directories.
        Computes quality metrics and statistical analysis without generating images.
        """
        base_images = sorted([
            os.path.join(base_dir, f) for f in os.listdir(base_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])
        improved_images = sorted([
            os.path.join(improved_dir, f) for f in os.listdir(improved_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        ])

        if not base_images or not improved_images:
            logger.error("No images found in one or both directories")
            return {"report": "Error: No images found", "analysis": []}

        logger.info("Found %d base and %d improved images",
                     len(base_images), len(improved_images))

        base_quality = [compute_single_image_quality(p) for p in base_images]
        improved_quality = [compute_single_image_quality(p) for p in improved_images]

        # Similarity metrics for paired images
        all_similarity = {}
        n_pairs = min(len(base_images), len(improved_images))
        for j in range(n_pairs):
            sim = compute_similarity_metrics(base_images[j], improved_images[j])
            for key, val in sim.items():
                all_similarity.setdefault(key, []).append(val)

        analysis = run_statistical_analysis(base_quality, improved_quality)
        report = format_report(analysis, all_similarity)

        return {"report": report, "analysis": analysis}


def _serialize(obj):
    """Make analysis result JSON-serializable."""
    result = {}
    for k, v in obj.items():
        if isinstance(v, tuple):
            result[k] = list(v)
        else:
            result[k] = v
    return result
