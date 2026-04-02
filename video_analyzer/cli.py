import argparse
from pathlib import Path
import json
import logging
import shutil
import sys
import gc
import time
from typing import Optional
import torch
import torch.backends.mps

from .config import Config, get_client, get_model
from .frame import VideoProcessor
from .prompt import PromptLoader
from .analyzer import VideoAnalyzer
from .audio_processor import AudioProcessor, AudioTranscript
from .clients.ollama import OllamaClient
from .clients.generic_openai_api import GenericOpenAIAPIClient
from .clients.sensenova import SenseNovaClient

# Initialize logger at module level
logger = logging.getLogger(__name__)


def build_video_signature(video_path: Path) -> dict:
    """Build a stable signature used to validate checkpoint compatibility."""
    resolved = video_path.resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

def get_log_level(level_str: str) -> int:
    """Convert string log level to logging constant."""
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)

def cleanup_files(output_dir: Path):
    """Clean up temporary files and directories."""
    try:
        frames_dir = output_dir / "frames"
        if frames_dir.exists():
            shutil.rmtree(frames_dir)
            logger.debug(f"Cleaned up frames directory: {frames_dir}")
            
        audio_file = output_dir / "audio.wav"
        if audio_file.exists():
            audio_file.unlink()
            logger.debug(f"Cleaned up audio file: {audio_file}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def create_client(config: Config):
    """Create the appropriate client based on configuration."""
    client_type = config.get("clients", {}).get("default", "ollama")
    client_config = get_client(config)
    
    if client_type == "ollama":
        return OllamaClient(client_config["url"])
    elif client_type == "openai_api":
        return GenericOpenAIAPIClient(client_config["api_key"], client_config["api_url"])
    elif client_type == "sensenova":
        return SenseNovaClient(client_config["api_key"], client_config["api_url"])
    else:
        raise ValueError(f"Unknown client type: {client_type}")

def main():
    parser = argparse.ArgumentParser(description="Analyze video using Vision models")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--config", type=str, default="config",
                        help="Path to configuration directory")
    parser.add_argument("--output", type=str, help="Output directory for analysis results")
    parser.add_argument("--client", type=str, help="Client to use (ollama or openrouter)")
    parser.add_argument("--ollama-url", type=str, help="URL for the Ollama service")
    parser.add_argument("--api-key", type=str, help="API key for OpenAI-compatible service")
    parser.add_argument("--api-url", type=str, help="API URL for OpenAI-compatible API")
    parser.add_argument("--model", type=str, help="Name of the vision model to use")
    parser.add_argument("--duration", type=float, help="Duration in seconds to process")
    parser.add_argument("--keep-frames", action="store_true", help="Keep extracted frames after analysis")
    parser.add_argument("--whisper-model", type=str, help="Whisper model size (tiny, base, small, medium, large), or path to local Whisper model snapshot")
    parser.add_argument("--start-stage", type=int, default=1, help="Stage to start processing from (1-3)")
    parser.add_argument("--max-frames", type=int, default=sys.maxsize, help="Maximum number of frames to process")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")
    parser.add_argument("--prompt", type=str, default="",
                        help="Question to ask about the video")
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--temperature", type=float, help="Temperature for LLM generation")
    args = parser.parse_args()

    # Set up logging with specified level
    log_level = get_log_level(args.log_level)
    # Configure the root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Force reconfiguration of the root logger
    )
    # Ensure our module logger has the correct level
    logger.setLevel(log_level)

    # Load and update configuration
    config = Config(args.config)
    config.update_from_args(args)

    # Initialize components
    video_path = Path(args.video_path)
    output_dir = Path(config.get("output_dir"))
    video_signature = build_video_signature(video_path)
    client = create_client(config)
    model = get_model(config)
    prompt_loader = PromptLoader(config.get("prompt_dir"), config.get("prompts", []))
    
    try:
        transcript = None
        frames = []
        frame_analyses = []
        video_description = None
        
        # Stage 1: Frame and Audio Processing
        if args.start_stage <= 1:
            # Initialize audio processor and extract transcript, the AudioProcessor accept following parameters that can be set in config.json:
            # language (str): Language code for audio transcription (default: None)
            # whisper_model (str): Whisper model size or path (default: "medium")
            # device (str): Device to use for audio processing (default: "cpu")
            logger.debug("Initializing audio processing...")
            audio_processor = AudioProcessor(language=config.get("audio", {}).get("language", ""), 
                                             model_size_or_path=config.get("audio", {}).get("whisper_model", "medium"),
                                             device=config.get("audio", {}).get("device", "cpu"))
            
            logger.info("Extracting audio from video...")
            try:
                audio_path = audio_processor.extract_audio(video_path, output_dir)
            except Exception as e:
                logger.error(f"Error extracting audio: {e}")
                audio_path = None
            
            if audio_path is None:
                logger.debug("No audio found in video - skipping transcription")
                transcript = None
            else:
                logger.info("Transcribing audio...")
                transcript = audio_processor.transcribe(audio_path)
                if transcript is None:
                    logger.warning("Could not generate reliable transcript. Proceeding with video analysis only.")
                
                # --- 内存优化：显式回收音频处理器 (Whisper 模型) ---
                logger.info("音频转录完成，正在释放模型内存以备画面分析...")
                del audio_processor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("音频处理器内存已释放。")
            
            logger.info(f"正在从视频中提取关键帧画面，采样密度: {config.get('frames', {}).get('per_minute', 60)} 帧/分钟...")
            processor = VideoProcessor(
                video_path, 
                output_dir / "frames", 
                model
            )
            frames = processor.extract_keyframes(
                frames_per_minute=config.get("frames", {}).get("per_minute", 60),
                duration=config.get("duration"),
                max_frames=args.max_frames
            )
            logger.info(f"关键帧提取完成，共计 {len(frames)} 帧。")
            
        # Stage 2: Frame Analysis
        if args.start_stage <= 2:
            logger.info(f"Analyzing {len(frames)} frames...")
            analyzer = VideoAnalyzer(
                client, 
                model, 
                prompt_loader,
                config.get("clients", {}).get("temperature", 0.2),
                config.get("prompt", "")
            )
            
            # --- 断点续传逻辑 (Checkpoint Support) ---
            frame_analyses = []
            results_file = output_dir / "analysis.json"
            existing_results = {}
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                        metadata = data.get("metadata", {})
                        existing_signature = metadata.get("source_video")
                        if existing_signature == video_signature:
                            # 建立快速索引：timestamp -> analysis
                            for fa in data.get("frame_analyses", []):
                                if "timestamp" in fa:
                                    existing_results[f"{fa['timestamp']:.3f}"] = fa
                            logger.info(
                                "Detected existing analysis.json for the same video, found %d completed frames.",
                                len(existing_results),
                            )
                        else:
                            logger.warning(
                                "Ignoring existing analysis.json because it belongs to a different source video."
                            )
                except Exception as e:
                    logger.warning(f"Could not load existing analysis for checkpoint: {e}")

            total_frames = len(frames)
            for i, frame in enumerate(frames):
                ts_key = f"{frame.timestamp:.3f}"
                if ts_key in existing_results:
                    logger.info(f"Checkpoint: Skipping frame {i+1}/{total_frames} (already analyzed)")
                    analysis = existing_results[ts_key]
                    # 同步到 analyzer 内存以便维持上下文
                    analyzer.previous_analyses.append(analysis)
                    frame_analyses.append(analysis)
                    continue

                logger.info(f"Progress: Analyzing frame {i+1}/{total_frames} ({(i+1)/total_frames*100:.1f}%)")
                analysis = analyzer.analyze_frame(frame)
                # 记录时间戳以便后续索引
                analysis["timestamp"] = frame.timestamp
                frame_analyses.append(analysis)
                
                # --- 增量存盘 (Incremental Save) ---
                # 每跑一帧都存一下，防止后面又被 429
                temp_results = {
                    "metadata": {
                        "frames_processed": len(frame_analyses),
                        "status": "incomplete",
                        "source_video": video_signature,
                    },
                    "frame_analyses": frame_analyses
                }
                with open(results_file, "w") as f:
                    json.dump(temp_results, f, indent=2)
                
                # --- 优雅退避 (Rate Limiting) ---
                # 每帧分析后强制休息 1.5 秒，防止触发魔搭等 API 的 429 报错
                time.sleep(1.5)
            
            logger.info("Frame analysis complete.")
                
        # Stage 3: Video Reconstruction
        if args.start_stage <= 3:
            logger.info("Reconstructing video description...")
            video_description = analyzer.reconstruct_video(
                frame_analyses, frames, transcript
            )
        
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "metadata": {
                "client": config.get("clients", {}).get("default"),
                "model": model,
                "whisper_model": config.get("audio", {}).get("whisper_model"),
                "frames_per_minute": config.get("frames", {}).get("per_minute"),
                "duration_processed": config.get("duration"),
                "frames_extracted": len(frames),
                "frames_processed": min(len(frames), args.max_frames),
                "start_stage": args.start_stage,
                "audio_language": transcript.language if transcript else None,
                "transcription_successful": transcript is not None,
                "transcript_used_in_description": bool(transcript and transcript.text.strip()),
                "transcript_word_count": transcript.word_count if transcript else 0,
                "transcript_avg_word_probability": (
                    transcript.average_word_probability if transcript else None
                ),
                "source_video": video_signature,
            },
            "transcript": {
                "text": transcript.text if transcript else None,
                "segments": transcript.segments if transcript else None,
                "word_count": transcript.word_count if transcript else None,
                "average_word_probability": (
                    transcript.average_word_probability if transcript else None
                ),
                "speech_duration": transcript.speech_duration if transcript else None,
            } if transcript else None,
            "frame_analyses": frame_analyses,
            "video_description": video_description
        }
        
        with open(output_dir / "analysis.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info("\nTranscript:")
        if transcript:
            logger.info(transcript.text)
        else:
            logger.info("No reliable transcript available")
            
        if video_description:
            logger.info("\nVideo Description:")
            logger.info(video_description.get("response", "No description generated"))
        
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        
        logger.info(f"Analysis complete. Results saved to {output_dir / 'analysis.json'}")
            
    except Exception as e:
        logger.error(f"Error during video analysis: {e}")
        if not config.get("keep_frames"):
            cleanup_files(output_dir)
        raise

if __name__ == "__main__":
    main()
