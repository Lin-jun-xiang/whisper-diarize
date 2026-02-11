from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Callable

import torch
import whisperx
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

ProgressCallback = Callable[[str], None]


def process_audio(
    audio_file: str,
    output_file: str,
    hf_token: str,
    progress_cb: ProgressCallback | None = None,
) -> None:
    """Transcribe, align, diarize, and write speaker-labeled text output."""
    if not hf_token:
        raise ValueError("hf_token is required for diarization.")

    input_path = Path(audio_file)
    output_path = Path(output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    torch.serialization.add_safe_globals([DictConfig, ListConfig])

    device = "cuda"
    batch_size = 8
    compute_type = "float16"

    model = None
    model_a = None
    diarize_model = None
    audio = None
    result = None
    diarize_segments = None

    try:
        if progress_cb:
            progress_cb("載入轉錄模型")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type, language="zh")

        if progress_cb:
            progress_cb("語音轉錄中")
        audio = whisperx.load_audio(str(input_path))
        result = model.transcribe(audio, batch_size=batch_size)

        if progress_cb:
            progress_cb("文字時間對齊中")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False
        )

        if progress_cb:
            progress_cb("說話者分離中")
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=hf_token, device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        if progress_cb:
            progress_cb("寫入文字檔")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for segment in result["segments"]:
                start = segment["start"]
                end = segment["end"]
                speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
                text = segment["text"]
                f.write(f"[{start:0.2f} - {end:0.2f}] {speaker}: {text}\n")
    finally:
        # Release strong references first, then clear CUDA cache.
        del model
        del model_a
        del diarize_model
        del audio
        del result
        del diarize_segments
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


if __name__ == "__main__":
    token = os.getenv("HUGGINGFACE_TOKEN", "")
    if not token:
        raise RuntimeError("Please set HUGGINGFACE_TOKEN before running directly.")

    process_audio(
        audio_file="./sample.wav",
        output_file="./output/sample_text.txt",
        hf_token=token,
    )
