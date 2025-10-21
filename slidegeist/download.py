"""Video download from URLs using yt-dlp."""

import logging
import re
import tempfile
from pathlib import Path
from typing import Literal

import yt_dlp

logger = logging.getLogger(__name__)

BrowserType = Literal["firefox", "safari", "chrome", "chromium", "edge", "opera", "brave"]


def translate_url(url: str) -> str:
    """Translate URLs to formats compatible with yt-dlp extractors.

    Args:
        url: Original URL.

    Returns:
        Translated URL if translation needed, otherwise original URL.

    Examples:
        TU Graz portal URL -> paella URL:
        https://tube.tugraz.at/portal/watch/<UUID>
        -> https://tube.tugraz.at/paella/ui/watch.html?id=<UUID>
    """
    # TU Graz Tube: portal format -> paella format
    tugraz_portal_pattern = r"https?://tube\.tugraz\.at/portal/watch/([0-9a-fA-F-]+)"
    match = re.match(tugraz_portal_pattern, url)
    if match:
        video_id = match.group(1)
        translated = f"https://tube.tugraz.at/paella/ui/watch.html?id={video_id}"
        logger.info(f"Translated TU Graz URL: {url} -> {translated}")
        return translated

    return url


def download_video(
    url: str,
    output_dir: Path | None = None,
    cookies_from_browser: BrowserType | None = None
) -> Path:
    """Download video from URL using yt-dlp.

    Supports YouTube, Mediasite, TU Graz Tube, and many other platforms.

    Args:
        url: Video URL to download.
        output_dir: Directory to save video. If None, uses temp directory.
        cookies_from_browser: Browser to extract cookies from for authentication.
            Supports: firefox, safari, chrome, chromium, edge, opera, brave.

    Returns:
        Path to the downloaded video file.

    Raises:
        Exception: If download fails.

    Examples:
        # Public video
        video_path = download_video("https://youtube.com/watch?v=...")

        # Authenticated video using Firefox cookies
        video_path = download_video(
            "https://tube.tugraz.at/...",
            cookies_from_browser="firefox"
        )
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="slidegeist_"))
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Template for output filename: use video title, sanitized
    output_template = str(output_dir / "%(title)s.%(ext)s")

    # yt-dlp options
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "quiet": False,
        "no_warnings": False,
        "extract_flat": False,
        "merge_output_format": "mp4",
    }

    # Translate URL to yt-dlp-compatible format if needed
    url = translate_url(url)

    # Add browser cookies if specified
    if cookies_from_browser:
        logger.info(f"Using cookies from {cookies_from_browser} browser")
        ydl_opts["cookiesfrombrowser"] = (cookies_from_browser,)

    logger.info(f"Downloading video from: {url}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract info first to get the final filename
            info = ydl.extract_info(url, download=True)

            if info is None:
                raise Exception("Failed to extract video information")

            # Get the downloaded file path
            if "requested_downloads" in info and info["requested_downloads"]:
                downloaded_file = Path(info["requested_downloads"][0]["filepath"])
            else:
                # Fallback: construct filename from info
                sanitized_title = ydl.prepare_filename(info)
                downloaded_file = Path(sanitized_title)

            if not downloaded_file.exists():
                raise FileNotFoundError(f"Downloaded file not found: {downloaded_file}")

            logger.info(f"Downloaded video to: {downloaded_file}")
            return downloaded_file

    except Exception as e:
        logger.error(f"Failed to download video: {e}")
        raise


def is_url(input_str: str) -> bool:
    """Check if input string is a URL.

    Args:
        input_str: String to check.

    Returns:
        True if input looks like a URL, False otherwise.
    """
    return input_str.startswith(("http://", "https://", "www."))
