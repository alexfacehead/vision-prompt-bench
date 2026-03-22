# Prompt Optimization Evaluation Framework

A framework for measuring whether AI-enhanced prompts produce statistically better images from OpenAI's image generation models (GPT Image 1 / 1.5).

## What It Does

1. Takes your base prompts and enhances them using AI (tree-of-thought expert prompting)
2. Generates multiple images for both the base and enhanced prompts
3. Computes no-reference image quality metrics (BRISQUE, entropy, colorfulness)
4. Runs paired statistical tests (Wilcoxon signed-rank) to determine if improvements are significant
5. Reports effect sizes (Cohen's d), confidence intervals, and p-values

## Key Insight

Unlike naive approaches that compare images using similarity metrics (SSIM, MSE), this framework correctly separates **quality** from **similarity**:

- **No-reference quality metrics** measure how good each image is independently — these determine if the enhanced prompt actually produced better images
- **Similarity metrics** (SSIM, PSNR, histogram correlation, VMAF) are reported separately to show how much the images changed — they do NOT measure quality

## Quickstart

```bash
# Clone and setup
git clone <repo-url>
cd dalle-3-optimization-framework
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Copy and fill in your API key
cp .env.template .env

# Run with prompts (generates images via API)
python main.py --prompts "a hyper realistic bengal cat with green eyes"

# Run with multiple prompts for better statistical power
python main.py --prompts \
  "a hyper realistic bengal cat with green eyes" \
  "a serene mountain landscape at sunset" \
  "a steampunk clockwork robot in a workshop" \
  --images-per-prompt 5

# Or evaluate pre-existing images
python main.py --base-dir path/to/base/images --improved-dir path/to/improved/images
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--prompts` | Base prompts to evaluate | — |
| `--images-per-prompt` | Images generated per prompt variant | 3 |
| `--base-dir` | Directory of pre-generated base images | — |
| `--improved-dir` | Directory of pre-generated improved images | — |
| `--output-dir` | Where to save results | `output/` |
| `--image-model` | Image generation model | `gpt-image-1` |
| `--chat-model` | Chat model for prompt enhancement | `gpt-4o-mini` |
| `--image-size` | Generated image dimensions | `1024x1024` |
| `--image-quality` | Image quality tier | `high` |
| `-v` | Verbose logging | off |

## Configuration

Copy `.env.template` to `.env` and set your values:

```
OPENAI_API_KEY=your_key_here
CHAT_MODEL=gpt-4o-mini
IMAGE_MODEL=gpt-image-1
IMAGES_PER_PROMPT=3
FFMPEG_PATH=  # Optional: for VMAF support
```

## Output

Each run creates a structured output directory:

```
output/run_2026-03-22_143000/
  prompts/          # Base and enhanced prompts
  images/           # Generated images
  metrics/          # Raw metric data (JSON)
  report.txt        # Statistical analysis report
```

## Metrics

### Quality Metrics (No-Reference)
- **BRISQUE** — Blind image quality assessment (lower = better)
- **Entropy** — Information content / detail richness (higher = better)
- **Colorfulness** — Color variety and vibrancy (higher = better)

### Similarity Metrics (Reference-Based)
- **SSIM** — Structural similarity (grayscale + color)
- **PSNR** — Peak signal-to-noise ratio
- **Histogram Correlation** — Color distribution similarity (per-channel)
- **VMAF** — Perceptual quality via FFmpeg (optional, requires libvmaf)

### Statistical Tests
- **Wilcoxon signed-rank test** — Non-parametric paired test (requires >= 6 samples)
- **Cohen's d** — Effect size magnitude
- **95% confidence intervals** — On mean quality differences

## Optional: VMAF Support

VMAF requires FFmpeg compiled with libvmaf. This is optional — all other metrics work without it.

```bash
# Build libvmaf (requires Python 3.7+, meson, ninja)
git clone https://github.com/Netflix/vmaf.git
cd vmaf/libvmaf
meson build --buildtype release
ninja -vC build

# Build FFmpeg with VMAF
git clone https://github.com/FFmpeg/FFmpeg.git
cd FFmpeg
./configure --enable-libvmaf
make -j4

# Set path in .env
FFMPEG_PATH=/path/to/your/ffmpeg
```

## Prompt Enhancement

The prompt enhancer can also be used standalone:

```bash
python -m src.easy_prompt_enhancer.prompt_enhancer "a beautiful sunset over the ocean"
```

## Known Issues

**pybrisque import error**: After installing, you may see `ModuleNotFoundError: No module named 'svmutil'`. Fix by editing the installed package:

```bash
# Find the file
python -c "import brisque; print(brisque.__file__)"
# Edit brisque/brisque.py: change these imports:
#   import svmutil              →  from libsvm import svmutil
#   from svmutil import ...     →  from libsvm.svmutil import ...
```

## Requirements

- Python 3.9+
- OpenAI API key with access to GPT Image 1 and a chat model
