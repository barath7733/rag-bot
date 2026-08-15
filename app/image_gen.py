"""
AI image generation.

Uses the free Pollinations.ai image API (https://pollinations.ai),
which requires no API key — a plain HTTPS request with a URL-encoded
prompt returns a generated image. This keeps the "get it working
today" bar low; if you later want higher-quality or commercially
licensed output, swap this module for a provider like Stability AI
or Together AI (both take an API key the same way Groq/Pinecone do
here — see the README for notes).
"""

from __future__ import annotations

import logging
import urllib.parse
import uuid

import httpx

from app.config import get_settings

logger = logging.getLogger("rag_chatbot.image_gen")

POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt"


class ImageGenerationError(Exception):
    """Raised for any user-facing image generation failure."""


def build_image_url(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """
    Build a direct, shareable image URL for the given prompt. The
    frontend can render this URL straight into an <img> tag — the
    image is generated on Pollinations' side on first fetch.
    """
    if not prompt or not prompt.strip():
        raise ImageGenerationError("Please provide a description of the image you want to generate.")

    settings = get_settings()
    encoded_prompt = urllib.parse.quote(prompt.strip())
    # A random seed keeps repeated identical prompts from being cached to the same image.
    seed = uuid.uuid4().int % 1_000_000
    return (
        f"{POLLINATIONS_BASE_URL}/{encoded_prompt}"
        f"?width={width}&height={height}&seed={seed}&nologo=true&model={settings.image_gen_model}"
    )


def generate_image(prompt: str, width: int = 1024, height: int = 1024) -> str:
    """
    Generate an image for the prompt and return its URL, after
    verifying the generation actually succeeds (Pollinations returns
    a real error status for disallowed/invalid prompts rather than a
    valid image).
    """
    url = build_image_url(prompt, width=width, height=height)

    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Image generation failed with status %s", exc.response.status_code)
        raise ImageGenerationError(
            "Image generation failed. Try rephrasing your prompt."
        ) from exc
    except httpx.HTTPError as exc:
        logger.error("Image generation request error: %s", exc)
        raise ImageGenerationError(f"Image generation request failed: {exc}") from exc

    return url
