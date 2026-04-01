from typing import List, Dict, Any, Optional
import logging
from .clients.llm_client import LLMClient
from .prompt import PromptLoader
from .frame import Frame
from .audio_processor import AudioTranscript

logger = logging.getLogger(__name__)

class VideoAnalyzer:
    def __init__(self, client: LLMClient, model: str, prompt_loader: PromptLoader, temperature: float, user_prompt: str = ""):
        """Initialize the VideoAnalyzer.
        
        Args:
            client: LLM client for making API calls
            model: Name of the model to use
            prompt_loader: Loader for prompt templates
            user_prompt: Optional user question about the video that will be injected into frame analysis
                        and video description prompts using the {prompt} token
        """
        self.client = client
        self.model = model
        self.prompt_loader = prompt_loader
        self.temperature = temperature
        self.user_prompt = user_prompt  # Store user's question about the video
        self._load_prompts()
        self.previous_analyses = []

    def _is_sensenova_client(self) -> bool:
        return self.client.__class__.__name__ == "SenseNovaClient"
        
    def _format_user_prompt(self) -> str:
        """Format the user's prompt by adding prefix if not empty."""
        if self.user_prompt:
            return f"我想了解：{self.user_prompt}"
        return ""

    def _load_prompts(self):
        """Load prompts from files."""
        self.frame_prompt = self.prompt_loader.get_by_index(0)  # Frame Analysis prompt
        self.video_prompt = self.prompt_loader.get_by_index(1)  # Video Reconstruction prompt

    def _format_previous_analyses(self) -> str:
        """Format previous frame analyses for inclusion in prompt."""
        if not self.previous_analyses:
            return ""
            
        formatted_analyses = []
        for i, analysis in enumerate(self.previous_analyses):
            formatted_analysis = (
                f"帧 {i}\n"
                f"{analysis.get('response', '不可用')}\n"
            )
            formatted_analyses.append(formatted_analysis)
            
        return "\n".join(formatted_analyses)

    def analyze_frame(self, frame: Frame) -> Dict[str, Any]:
        """Analyze a single frame using the LLM."""
        if self._is_sensenova_client():
            prompt = (
                "请使用中文如实描述当前视频帧。\n"
                "要求：\n"
                "1. 优先逐字抄录画面中可见的文字；如果没有文字，明确写“无可见文字”。\n"
                "2. 再描述人物外观、动作、表情。\n"
                "3. 再描述场景与背景。\n"
                "4. 禁止推测人物身份、剧情、动机、前因后果。\n"
                "5. 只输出客观描述，控制在2到4句话。\n"
            )
            if self.user_prompt:
                prompt += f"补充关注点：{self.user_prompt}\n"
            prompt += f"这是在 {frame.timestamp:.2f} 秒截取的第 {frame.number} 帧图片。"
            num_predict = 1024
        else:
            # Replace {PREVIOUS_FRAMES} token with formatted previous analyses
            prompt = self.frame_prompt.replace("{PREVIOUS_FRAMES}", self._format_previous_analyses())
            prompt = prompt.replace("{prompt}", self._format_user_prompt())
            prompt = f"{prompt}\n这是在 {frame.timestamp:.2f} 秒截取的第 {frame.number} 帧图片。\n请使用中文进行详细描述。"
            num_predict = 500
        
        try:
            response = self.client.generate(
                prompt=prompt,
                image_path=str(frame.path),
                model=self.model,
                temperature=self.temperature,
                num_predict=num_predict
            )
            if self._is_sensenova_client() and not response.get("response", "").strip():
                logger.warning(f"SenseNova returned empty content for frame {frame.number}, retrying with an even shorter prompt")
                retry_prompt = (
                    "请用中文输出三行：\n"
                    "画面文字：...\n"
                    "人物动作：...\n"
                    "场景背景：...\n"
                    "不要推测剧情和身份。"
                )
                response = self.client.generate(
                    prompt=retry_prompt,
                    image_path=str(frame.path),
                    model=self.model,
                    temperature=0.0,
                    num_predict=1024
                )
            logger.debug(f"Successfully analyzed frame {frame.number}")
            
            # Store the analysis for future frames
            analysis_result = {k: v for k, v in response.items() if k != "context"}
            self.previous_analyses.append(analysis_result)
            
            return analysis_result
        except Exception as e:
            logger.error(f"Error analyzing frame {frame.number}: {e}")
            error_result = {"response": f"分析帧 {frame.number} 时出错: {str(e)}"}
            self.previous_analyses.append(error_result)
            return error_result

    def reconstruct_video(self, frame_analyses: List[Dict[str, Any]], frames: List[Frame], 
                         transcript: Optional[AudioTranscript] = None) -> Dict[str, Any]:
        """Reconstruct video description from frame analyses and transcript."""
        frame_notes = []
        for i, (frame, analysis) in enumerate(zip(frames, frame_analyses)):
            frame_note = (
                f"帧 {i} ({frame.timestamp:.2f}s):\n"
                f"{analysis.get('response', '无分析内容')}"
            )
            frame_notes.append(frame_note)
        
        analysis_text = "\n\n".join(frame_notes)
        
        # Get first frame analysis
        first_frame_text = ""
        if frame_analyses and len(frame_analyses) > 0:
            first_frame_text = frame_analyses[0].get('response', '')
        
        # Include transcript information if available
        transcript_text = ""
        if transcript and transcript.text.strip():
            transcript_text = transcript.text
        
        prompt = self.video_prompt.replace("{prompt}", self._format_user_prompt())
        prompt = prompt.replace("{FRAME_NOTES}", analysis_text)
        prompt = prompt.replace("{FIRST_FRAME}", first_frame_text)
        prompt = prompt.replace("{TRANSCRIPT}", transcript_text)
        
        # 强制中文输出要求
        prompt = f"{prompt}\n请根据以上信息，使用中文对整个视频进行总结和叙述。"
        
        try:
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                temperature=self.temperature,
                num_predict=4096
            )
            logger.info("Successfully reconstructed video description")
            return {k: v for k, v in response.items() if k != "context"}
        except Exception as e:
            logger.error(f"Error reconstructing video: {e}")
            return {"response": f"Error reconstructing video: {str(e)}"}
