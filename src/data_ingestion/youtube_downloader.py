from pathlib import Path
import json
import re
import subprocess
from datetime import datetime
from typing import Optional, List, Tuple, Any
import asyncio
import logging

try:
    import yt_dlp
except ImportError:
    raise ImportError("yt-dlp is required. Install with: pip install -e '.[data]'")

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DownloadResult(BaseModel):
    success: bool
    file_path: Optional[Path] = None
    metadata: dict = {}
    error: Optional[str] = None


class YouTubeAudioDownloader:
    """Robust YouTube audio downloader integrated for the Swedish sentiment analysis project.

    Downloads best audio, converts to 16kHz mono WAV (optimal for Whisper ASR),
    saves rich metadata JSON, supports playlists.
    """

    def __init__(self, output_base: Path = Path("data/ingested/youtube")):
        self.output_base = Path(output_base)
        self.output_base.mkdir(parents=True, exist_ok=True)

    def sanitize_filename(self, filename: str, max_length: int = 120) -> str:
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        filename = re.sub(r'\s+', ' ', filename).strip()
        if len(filename) > max_length:
            filename = filename[:max_length].rsplit(' ', 1)[0]
        return filename

    def download(
        self,
        url: str,
        playlist: bool = False,
        convert_to_wav: bool = True,
        sample_rate: int = 16000,
        channels: int = 1,
    ) -> DownloadResult | List[DownloadResult]:
        """Synchronous download. Returns single result or list for playlists."""
        output_dir = self.output_base
        output_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = output_dir / "_temp_yt_dlp"
        temp_dir.mkdir(exist_ok=True)

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(temp_dir / "%(title)s [%(id)s].%(ext)s"),
            "noplaylist": not playlist,
            "quiet": False,
            "no_warnings": False,
            "extractaudio": True,
            "restrictfilenames": False,
            "windowsfilenames": False,
        }

        logger.info(f"📥 Downloading audio from: {url}")

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                # Handle playlist
                if playlist and "entries" in info:
                    entries = [e for e in info.get("entries", []) if e]
                    logger.info(f"Playlist with {len(entries)} videos – downloading sequentially...")
                    results = []
                    for entry in entries:
                        video_url = entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry.get('id')}"
                        res = self.download(
                            video_url,
                            playlist=False,
                            convert_to_wav=convert_to_wav,
                            sample_rate=sample_rate,
                            channels=channels,
                        )
                        if isinstance(res, DownloadResult):
                            results.append(res)
                    return results

                info_dict = info
                if "entries" in info_dict:
                    info_dict = info_dict["entries"][0]

                video_id = info_dict.get("id", "unknown")
                title = info_dict.get("title", f"Unknown_{video_id}")
                uploader = info_dict.get("uploader", info_dict.get("channel", ""))
                duration = info_dict.get("duration", 0)
                upload_date = info_dict.get("upload_date", "")

                downloaded = list(temp_dir.glob(f"*[{video_id}]*"))
                if not downloaded:
                    raise FileNotFoundError(f"Downloaded file not found for {video_id}")

                source_audio = downloaded[0]
                safe_title = self.sanitize_filename(title)
                base_name = f"{safe_title} [{video_id}]"

                if convert_to_wav:
                    wav_path = output_dir / f"{base_name}.wav"
                    logger.info("🔄 Converting to 16 kHz mono WAV with ffmpeg...")

                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", str(source_audio),
                        "-ar", str(sample_rate), "-ac", str(channels),
                        "-c:a", "pcm_s16le", "-loglevel", "error",
                        str(wav_path),
                    ]
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.warning(f"ffmpeg error: {result.stderr}")
                        final_path = output_dir / f"{base_name}{source_audio.suffix}"
                        source_audio.rename(final_path)
                        output_audio_path = final_path
                        actual_format = source_audio.suffix.lstrip(".")
                    else:
                        source_audio.unlink(missing_ok=True)
                        output_audio_path = wav_path
                        actual_format = "wav_16khz_mono"
                else:
                    final_path = output_dir / f"{base_name}{source_audio.suffix}"
                    source_audio.rename(final_path)
                    output_audio_path = final_path
                    actual_format = source_audio.suffix.lstrip(".")

                metadata = {
                    "source_url": url,
                    "youtube_id": video_id,
                    "title": title,
                    "uploader": uploader,
                    "duration_seconds": duration,
                    "upload_date": upload_date,
                    "downloaded_at": datetime.now().isoformat(),
                    "format": actual_format,
                    "sample_rate_hz": sample_rate if convert_to_wav else None,
                    "channels": channels if convert_to_wav else None,
                    "file_path": str(output_audio_path.relative_to(output_dir)),
                    "project": "Automatisk-sentimentanalys",
                    "purpose": "test_data_for_asr_and_sentiment",
                    "language": "sv",
                }

                meta_path = output_dir / f"{base_name}.json"
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                logger.info(f"✅ Done: {output_audio_path.name} | Metadata: {meta_path.name} | Duration: {duration}s")

                return DownloadResult(
                    success=True,
                    file_path=output_audio_path,
                    metadata=metadata,
                )

        except yt_dlp.utils.DownloadError as e:
            logger.error(f"yt-dlp download error: {e}")
            error = str(e)
        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg error: {e}")
            error = str(e)
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            error = str(e)

        # Cleanup temp on error
        for f in temp_dir.glob("*"):
            f.unlink(missing_ok=True)

        return DownloadResult(success=False, error=error)

    async def adownload(self, url: str, **kwargs) -> DownloadResult | List[DownloadResult]:
        """Async wrapper using thread pool."""
        return await asyncio.to_thread(self.download, url, **kwargs)


# Backward compatibility functions (can be used from CLI or scripts)
def download_youtube_audio(url: str, output_dir: Path = Path("data/ingested/youtube"), **kwargs) -> Tuple[Optional[Path], Optional[dict]]:
    downloader = YouTubeAudioDownloader(output_base=output_dir)
    result = downloader.download(url, **kwargs)
    if isinstance(result, list):
        if result:
            first = result[0]
            return first.file_path, first.metadata
        return None, None
    return result.file_path if result.success else None, result.metadata if result.success else None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="YouTube Audio Downloader for Swedish Call Center Sentiment Analysis")
    parser.add_argument("url", help="YouTube URL")
    parser.add_argument("--playlist", action="store_true")
    parser.add_argument("--no-wav", action="store_true")
    args = parser.parse_args()

    downloader = YouTubeAudioDownloader()
    res = downloader.download(args.url, playlist=args.playlist, convert_to_wav=not args.no_wav)
    print(res)


if __name__ == "__main__":
    main()