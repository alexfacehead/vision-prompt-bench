import math
import shutil
import subprocess
import os
import logging

import cv2
import numpy as np
from PIL import Image
from brisque import BRISQUE
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

logger = logging.getLogger(__name__)

brisq = BRISQUE()


# --- No-Reference Quality Metrics ---
# These measure absolute quality of a single image.

def calculate_brisque(image_path: str) -> float:
    return brisq.get_score(image_path)


def calculate_entropy(img_np: np.ndarray) -> float:
    hist = cv2.calcHist([img_np], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    return float(entropy(hist))


def calculate_colorfulness(img_np: np.ndarray) -> float:
    if len(img_np.shape) < 3 or img_np.shape[2] < 3:
        return 0.0
    if img_np.shape[2] > 3:
        img_np = img_np[:, :, :3]
    (B, G, R) = cv2.split(img_np.astype("float"))
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = math.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) + \
                   0.3 * (np.mean(rg) + np.mean(yb))
    return colorfulness


# --- Reference-Based Similarity Metrics ---
# These measure how similar two images are (NOT quality).

def calculate_ssim(image1_np: np.ndarray, image2_np: np.ndarray,
                   multichannel: bool = False) -> float:
    if multichannel and len(image1_np.shape) == 3:
        return ssim(image1_np, image2_np, channel_axis=-1)
    return ssim(image1_np, image2_np)


def calculate_psnr(image1_np: np.ndarray, image2_np: np.ndarray) -> float:
    mse_value = mse(image1_np, image2_np)
    if mse_value == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse_value))


def calculate_histogram_correlation(img1_np: np.ndarray,
                                    img2_np: np.ndarray) -> float:
    if len(img1_np.shape) == 3:
        # Compute per-channel and average
        correlations = []
        for ch in range(img1_np.shape[2]):
            h1 = cv2.calcHist([img1_np], [ch], None, [256], [0, 256])
            h2 = cv2.calcHist([img2_np], [ch], None, [256], [0, 256])
            correlations.append(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
        return float(np.mean(correlations))
    else:
        h1 = cv2.calcHist([img1_np], [0], None, [256], [0, 256])
        h2 = cv2.calcHist([img2_np], [0], None, [256], [0, 256])
        return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def calculate_vmaf(reference_image: str, distorted_image: str) -> float:
    ffmpeg_path = os.getenv("FFMPEG_PATH") or shutil.which("ffmpeg")
    if not ffmpeg_path:
        logger.warning("FFmpeg not found. Skipping VMAF. Set FFMPEG_PATH or install FFmpeg.")
        return None

    command = [
        ffmpeg_path,
        "-i", reference_image,
        "-i", distorted_image,
        "-filter_complex", "libvmaf",
        "-an", "-f", "null", "-",
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        for line in result.stderr.split("\n"):
            if "VMAF score" in line:
                return float(line.split()[-1])
        logger.warning("VMAF score not found in FFmpeg output")
        return None
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("VMAF calculation failed: %s", e)
        return None


# --- Image Comparison ---

def compute_single_image_quality(image_path: str) -> dict:
    """Compute no-reference quality metrics for a single image."""
    img_color = np.array(Image.open(image_path))
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)

    return {
        "brisque": calculate_brisque(image_path),
        "entropy": calculate_entropy(img_gray),
        "colorfulness": calculate_colorfulness(img_color),
    }


def compute_similarity_metrics(image1_path: str, image2_path: str) -> dict:
    """Compute reference-based similarity metrics between two images."""
    img1_color = np.array(Image.open(image1_path))
    img2_color = np.array(Image.open(image2_path))
    img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_RGB2GRAY)
    img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_RGB2GRAY)

    result = {
        "ssim_gray": calculate_ssim(img1_gray, img2_gray),
        "ssim_color": calculate_ssim(img1_color, img2_color, multichannel=True),
        "psnr": calculate_psnr(img1_gray, img2_gray),
        "histogram_correlation": calculate_histogram_correlation(
            img1_color, img2_color
        ),
    }

    # VMAF is optional (requires FFmpeg with libvmaf)
    vmaf = calculate_vmaf(image1_path, image2_path)  # correct arg order: ref, distorted
    if vmaf is not None:
        result["vmaf"] = vmaf

    return result
