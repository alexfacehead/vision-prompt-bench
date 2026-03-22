import argparse
import logging
import os

from dotenv import load_dotenv
from src.pipeline import Pipeline

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate whether AI-enhanced prompts produce measurably "
                    "better images using statistical analysis."
    )

    # Image generation mode
    parser.add_argument(
        "--prompts", nargs="+",
        help="Base prompts to evaluate (generates images via API)."
    )
    parser.add_argument(
        "--images-per-prompt", type=int,
        default=int(os.getenv("IMAGES_PER_PROMPT", "3")),
        help="Number of images to generate per prompt variant (default: 3)."
    )

    # Pre-existing images mode
    parser.add_argument(
        "--base-dir",
        help="Directory of pre-generated base images (skips image generation)."
    )
    parser.add_argument(
        "--improved-dir",
        help="Directory of pre-generated improved images (skips image generation)."
    )

    # Configuration
    parser.add_argument("--output-dir", default="output",
                        help="Output directory for results (default: output/).")
    parser.add_argument("--image-model", default=None,
                        help="Image generation model (default: gpt-image-1).")
    parser.add_argument("--chat-model", default=None,
                        help="Chat model for prompt enhancement (default: gpt-4o-mini).")
    parser.add_argument("--image-size", default="1024x1024")
    parser.add_argument("--image-quality", default="high")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable verbose logging.")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pipeline = Pipeline(
        images_per_prompt=args.images_per_prompt,
        image_model=args.image_model,
        chat_model=args.chat_model,
        image_size=args.image_size,
        image_quality=args.image_quality,
        output_dir=args.output_dir,
    )

    if args.base_dir and args.improved_dir:
        # Evaluate pre-existing images
        results = pipeline.run_from_directories(args.base_dir, args.improved_dir)
    elif args.prompts:
        # Full pipeline: enhance, generate, evaluate
        results = pipeline.run(args.prompts)
    else:
        parser.error("Provide either --prompts or both --base-dir and --improved-dir.")
        return

    print(results["report"])

    if "run_dir" in results:
        print(f"\nFull results saved to: {results['run_dir']}")


if __name__ == "__main__":
    main()
