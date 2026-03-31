#!/usr/bin/env python3
import json
import os
from pathlib import Path
import tempfile

def find_latest_session(results_dir):
    """找到最近一次的 session 目录。"""
    if not Path(results_dir).exists():
        return None
    sessions = sorted(Path(results_dir).iterdir(), key=os.path.getmtime, reverse=True)
    for s in sessions:
        if (s / "analysis.json").exists():
            return s
    return None

def dump_notes():
    print("\n" + "📝" * 30)
    print("      Video Analyzer 原始笔记脱水导出器")
    print("      —— 直接提取 235B 模型生成的 100+ 帧数据 ——")
    print("📝" * 30 + "\n")

    # 1. 定位分析结果
    tmp_results = Path(tempfile.gettempdir()) / 'video-analyzer-ui' / 'results'
    latest_session = find_latest_session(tmp_results)
    
    if not latest_session:
        print("❌ 错误：未找到任何分析记录。")
        return

    json_path = latest_session / "analysis.json"
    print(f"🔍 正在从以下路径打捞数据:\n   {json_path}\n")

    # 2. 读取并解析
    with open(json_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"❌ 错误：读取 JSON 失败 - {e}")
            return

    frame_analyses = data.get("frame_analyses", [])
    if not frame_analyses:
        print("💡 提示：该文件中不含帧分析记录。")
        return

    # 3. 格式化并输出到 TXT/Markdown
    output_file = Path.cwd() / "FULL_DIRECTOR_NOTES.md"
    
    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"# 视频原始分析报告 (Qwen3-VL-235B 原始笔记)\n\n")
        out.write(f"> **Session ID**: {latest_session.name}\n")
        out.write(f"> **已成功提取总帧数**: {len(frame_analyses)}\n\n")
        out.write("---\n\n")

        for i, fa in enumerate(frame_analyses):
            ts = fa.get("timestamp", 0.0)
            response = fa.get("response", "分析记录丢失")
            
            # 美化输出
            out.write(f"### [第 {i} 帧] 时间戳: {ts:.2f}s\n")
            out.write(f"{response}\n\n")
            out.write("---\n\n")

    print(f"🎉 营救成功！")
    print(f"所有 {len(frame_analyses)} 帧的原始分析笔记已导出至：")
    print(f"👉 {output_file}\n")
    print("你可以直接打开该文件查看 235B 模型捕捉到的所有剧情细节！")

if __name__ == "__main__":
    dump_notes()
