from __future__ import annotations

import io
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from src.transcriber import process_audio

app = FastAPI(title="Whisper Diarize Service", version="0.2.0")

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
JOB_CLEANUP_INTERVAL_SECONDS = 60
JOB_RESULT_TTL_SECONDS = 1800
JOB_FAILED_TTL_SECONDS = 900


@dataclass
class JobState:
    job_id: str
    status: str = "queued"  # queued | running | completed | failed
    message: str = "排隊中"
    total_files: int = 0
    done_files: int = 0
    current_file: str = ""
    error: str = ""
    zip_bytes: bytes | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


JOBS: dict[str, JobState] = {}
JOBS_LOCK = threading.Lock()
CLEANUP_STARTED = False
CLEANUP_LOCK = threading.Lock()


def _safe_name(name: str) -> str:
    return Path(name or "audio").name


def _validate_extension(filename: str) -> None:
    ext = Path(filename).suffix.lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {filename}")


def _set_job(job: JobState, **kwargs: Any) -> None:
    with job.lock:
        for key, value in kwargs.items():
            setattr(job, key, value)
        job.updated_at = time.time()
        if job.status in {"completed", "failed"} and job.finished_at is None:
            job.finished_at = job.updated_at


def _build_zip_bytes(output_files: list[Path]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zf:
        for out_file in output_files:
            zf.write(out_file, arcname=out_file.name)
    return buffer.getvalue()


def _run_job(job: JobState, temp_root: Path, input_files: list[Path], token: str) -> None:
    output_dir = temp_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files: list[Path] = []
    _set_job(job, status="running", message="準備開始", total_files=len(input_files), done_files=0)

    try:
        for index, input_path in enumerate(input_files, start=1):
            output_path = output_dir / f"{input_path.stem}_text.txt"

            def update_stage(stage: str) -> None:
                _set_job(
                    job,
                    current_file=input_path.name,
                    message=f"[{index}/{len(input_files)}] {input_path.name} - {stage}",
                )

            update_stage("初始化")
            process_audio(str(input_path), str(output_path), token, progress_cb=update_stage)
            output_files.append(output_path)
            _set_job(job, done_files=index, message=f"[{index}/{len(input_files)}] {input_path.name} - 完成")

        zip_bytes = _build_zip_bytes(output_files)
        _set_job(job, status="completed", message="全部完成，可下載結果", zip_bytes=zip_bytes)
    except Exception as exc:
        _set_job(job, status="failed", error=str(exc), message="處理失敗")
    finally:
        # No persistent storage: always clean temporary uploaded/result files.
        shutil.rmtree(temp_root, ignore_errors=True)


def _cleanup_jobs_loop() -> None:
    while True:
        time.sleep(JOB_CLEANUP_INTERVAL_SECONDS)
        now = time.time()
        expired_ids: list[str] = []

        with JOBS_LOCK:
            for job_id, job in JOBS.items():
                with job.lock:
                    if job.status == "completed" and job.finished_at:
                        if now - job.finished_at >= JOB_RESULT_TTL_SECONDS:
                            expired_ids.append(job_id)
                    elif job.status == "failed" and job.finished_at:
                        if now - job.finished_at >= JOB_FAILED_TTL_SECONDS:
                            expired_ids.append(job_id)

            for job_id in expired_ids:
                JOBS.pop(job_id, None)


@app.on_event("startup")
def _startup_cleanup_worker() -> None:
    global CLEANUP_STARTED
    with CLEANUP_LOCK:
        if CLEANUP_STARTED:
            return
        cleaner = threading.Thread(target=_cleanup_jobs_loop, daemon=True)
        cleaner.start()
        CLEANUP_STARTED = True


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang='zh-Hant'>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <title>Whisper Diarize</title>
  <style>
    :root {
      --bg-top: #f7fbff;
      --bg-bottom: #eef4f9;
      --ink: #0f2233;
      --muted: #556373;
      --line: #cedae6;
      --card: #ffffff;
      --accent: #0d9488;
      --accent-ink: #ffffff;
      --focus: #0f766e;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 15%, #d7f3ef 0, #d7f3ef 18%, transparent 40%),
        radial-gradient(circle at 88% 82%, #dbeafe 0, #dbeafe 15%, transparent 37%),
        linear-gradient(160deg, var(--bg-top), var(--bg-bottom));
      font-family: "Space Grotesk", "Noto Sans TC", "Segoe UI", sans-serif;
      display: grid;
      place-items: center;
      padding: 24px;
    }

    .card {
      width: min(820px, 100%);
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 28px;
      box-shadow: 0 20px 50px rgba(16, 24, 40, 0.10);
    }

    h1 {
      margin: 0 0 8px;
      font-size: clamp(24px, 3vw, 34px);
      letter-spacing: 0.02em;
    }

    .subtitle {
      margin: 0 0 24px;
      color: var(--muted);
      font-size: 15px;
    }

    .field { margin-bottom: 16px; }

    label {
      display: block;
      margin-bottom: 8px;
      font-weight: 600;
      font-size: 14px;
    }

    input[type='password'],
    input[type='file'] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 11px 12px;
      font: inherit;
      background: #fff;
    }

    input:focus-visible,
    button:focus-visible,
    .download:focus-visible {
      outline: 3px solid color-mix(in srgb, var(--focus) 25%, transparent);
      outline-offset: 1px;
      border-color: var(--focus);
    }

    .help {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }

    .actions {
      margin-top: 20px;
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }

    button,
    .download {
      border: 0;
      border-radius: 12px;
      padding: 11px 18px;
      background: linear-gradient(135deg, #0f766e, var(--accent));
      color: var(--accent-ink);
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      justify-content: center;
    }

    button:disabled { opacity: 0.6; cursor: wait; }

    .status {
      color: var(--muted);
      font-size: 14px;
      min-height: 20px;
      width: 100%;
    }

    .progress-wrap {
      margin-top: 12px;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 999px;
      overflow: hidden;
      height: 14px;
      background: #f0f4f8;
    }

    .progress-bar {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #0d9488, #06b6d4);
      transition: width 240ms ease;
    }

    .download { display: none; }

    @media (max-width: 640px) {
      .card { padding: 20px; border-radius: 14px; }
    }
  </style>
</head>
<body>
  <main class='card'>
    <h1>Whisper Diarize</h1>
    <p class='subtitle'>上傳多個音訊檔，系統會依序轉寫、顯示進度，完成後提供文字壓縮檔下載。</p>

    <form id='upload-form' enctype='multipart/form-data'>
      <div class='field'>
        <label for='hf_token'>Hugging Face Token</label>
        <input id='hf_token' type='password' name='hf_token' placeholder='可留空（若已設定 HUGGINGFACE_TOKEN）'>
      </div>

      <div class='field'>
        <label for='audio_files'>Audio Files</label>
        <input id='audio_files' type='file' name='audio_files' multiple required>
        <div class='help' id='file-help'>支援多檔上傳：wav, mp3, m4a, flac, ogg, aac</div>
      </div>

      <div class='actions'>
        <button id='submit-btn' type='submit'>開始轉寫</button>
        <a id='download-link' class='download' href='#'>下載結果 ZIP</a>
        <div class='status' id='status'>待命中</div>
      </div>
      <div class='progress-wrap'>
        <div id='progress-bar' class='progress-bar'></div>
      </div>
    </form>
  </main>

  <script>
    const form = document.getElementById('upload-form');
    const fileInput = document.getElementById('audio_files');
    const fileHelp = document.getElementById('file-help');
    const statusEl = document.getElementById('status');
    const submitBtn = document.getElementById('submit-btn');
    const downloadLink = document.getElementById('download-link');
    const progressBar = document.getElementById('progress-bar');

    let pollTimer = null;

    function setProgress(done, total) {
      const percent = total > 0 ? Math.round((done / total) * 100) : 0;
      progressBar.style.width = `${percent}%`;
    }

    async function pollJob(jobId) {
      pollTimer = setInterval(async () => {
        try {
          const resp = await fetch(`/jobs/${jobId}/status`, { cache: 'no-store' });
          if (!resp.ok) throw new Error('讀取進度失敗');
          const data = await resp.json();

          statusEl.textContent = data.message || data.status;
          setProgress(data.done_files || 0, data.total_files || 0);

          if (data.status === 'completed') {
            clearInterval(pollTimer);
            submitBtn.disabled = false;
            downloadLink.style.display = 'inline-flex';
            downloadLink.href = `/jobs/${jobId}/download`;
            statusEl.textContent = '已完成，請下載結果';
            setProgress(1, 1);
          }

          if (data.status === 'failed') {
            clearInterval(pollTimer);
            submitBtn.disabled = false;
            statusEl.textContent = `失敗：${data.error || '未知錯誤'}`;
          }
        } catch (err) {
          clearInterval(pollTimer);
          submitBtn.disabled = false;
          statusEl.textContent = `失敗：${err.message}`;
        }
      }, 1200);
    }

    fileInput.addEventListener('change', () => {
      const count = fileInput.files ? fileInput.files.length : 0;
      fileHelp.textContent = count > 0
        ? `已選擇 ${count} 個檔案`
        : '支援多檔上傳：wav, mp3, m4a, flac, ogg, aac';
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      if (pollTimer) clearInterval(pollTimer);

      submitBtn.disabled = true;
      downloadLink.style.display = 'none';
      progressBar.style.width = '0%';
      statusEl.textContent = '上傳中...';

      const formData = new FormData(form);

      try {
        const resp = await fetch('/jobs', {
          method: 'POST',
          body: formData,
        });

        const data = await resp.json();
        if (!resp.ok) {
          throw new Error(data.detail || '建立任務失敗');
        }

        statusEl.textContent = '任務已建立，開始處理...';
        await pollJob(data.job_id);
      } catch (err) {
        submitBtn.disabled = false;
        statusEl.textContent = `失敗：${err.message}`;
      }
    });
  </script>
</body>
</html>
"""


@app.post("/jobs")
async def create_job(
    audio_files: list[UploadFile] = File(...),
    hf_token: str = Form(default=""),
) -> dict[str, str]:
    token = hf_token.strip() or os.getenv("HUGGINGFACE_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="Missing Hugging Face token.")
    if not audio_files:
        raise HTTPException(status_code=400, detail="No audio files uploaded.")

    job_id = uuid.uuid4().hex
    job = JobState(job_id=job_id)
    with JOBS_LOCK:
        JOBS[job_id] = job

    temp_root = Path(tempfile.mkdtemp(prefix=f"whisper_diarize_{job_id}_"))
    input_dir = temp_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    input_files: list[Path] = []
    for upload in audio_files:
        src_name = _safe_name(upload.filename or "audio")
        _validate_extension(src_name)

        input_path = input_dir / src_name
        with input_path.open("wb") as f:
            shutil.copyfileobj(upload.file, f)
        input_files.append(input_path)
        await upload.close()

    worker = threading.Thread(target=_run_job, args=(job, temp_root, input_files, token), daemon=True)
    worker.start()

    return {"job_id": job_id}


@app.get("/jobs/{job_id}/status")
def job_status(job_id: str) -> dict[str, Any]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    with job.lock:
        return {
            "job_id": job.job_id,
            "status": job.status,
            "message": job.message,
            "total_files": job.total_files,
            "done_files": job.done_files,
            "current_file": job.current_file,
            "error": job.error,
        }


@app.get("/jobs/{job_id}/download")
def download_result(job_id: str) -> Response:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    with job.lock:
        if job.status != "completed" or not job.zip_bytes:
            raise HTTPException(status_code=409, detail="Job is not completed yet")
        payload = job.zip_bytes

    # Remove in-memory job artifact after issuing download response.
    with JOBS_LOCK:
        JOBS.pop(job_id, None)

    return Response(
        content=payload,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="transcripts_{job_id}.zip"'},
    )


def main() -> None:
    uvicorn.run("app:app", host="0.0.0.0", port=8134, reload=False)


if __name__ == "__main__":
    main()
