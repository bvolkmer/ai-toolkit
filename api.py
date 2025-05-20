from __future__ import annotations

import os
import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
import logging
import math
from pathlib import Path
import uuid
from fastapi.responses import FileResponse
import yaml
from collections import Counter
from copy import deepcopy
from enum import StrEnum, auto
from types import TracebackType
from typing import Iterable, Self
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta, timezone

import anycorn
import anyio
import anyio.to_thread
import rich.traceback
from anycorn.config import Config
from anyio.abc import ObjectReceiveStream, ObjectSendStream, TaskGroup
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from pydantic import BaseModel
from rich.logging import RichHandler

from toolkit.job import get_job


logger = logging.getLogger("api")

app = FastAPI()

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["DISABLE_TELEMETRY"] = "YES"


base_config = yaml.safe_load("""---
job: "extension"
config:
  name: "helena-test"
  process:
    - type: "sd_trainer"
      training_folder: "./output"
      sqlite_db_path: "./aitk_db.db"
      device: "cuda."
      trigger_word: null
      performance_log_every: 0
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
        lokr_full_rank: true
        lokr_factor: -1
        network_kwargs:
          ignore_if_contains: []
      save:
        dtype: "bf16"
        save_every: 0
        max_step_saves_to_keep: 4
        save_format: "diffusers"
        push_to_hub: false
      datasets:
        - folder_path: "./datasets/helena"
          mask_path: null
          mask_min_value: 0.1
          default_caption: ""
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          cache_latents_to_disk: false
          is_reg: false
          network_weight: 1
          resolution:
            - 512
            - 768
            - 1024
          controls: []
      train:
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        quantize: true
        quantize_te: true
        arch: "flux"
        low_vram: false
        model_kwargs: {}
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 960
        height: 1280
        prompts: []
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 4
        sample_steps: 25
        num_frames: 1
        fps: 1
meta:
  name: "[name]"
  version: "1.0"
""")


class Training(BaseModel, frozen=True):
    config: str = """---
batch_size: 1
bypass_guidance_embedding: false
steps: 30
gradient_accumulation: 1
train_unet: true
train_text_encoder: false
gradient_checkpointing: true
noise_scheduler: "flowmatch"
optimizer: "adamw8bit"
timestep_type: "sigmoid"
content_or_style: "balanced"
optimizer_params:
  weight_decay: 0.0001
unload_text_encoder: false
lr: 0.0001
ema_config:
  use_ema: false
  ema_decay: 0.99
dtype: "bf16"
diff_output_preservation: false
diff_output_preservation_multiplier: 1
diff_output_preservation_class: "person"
"""


class Status(StrEnum):
    QUEUED = auto()
    PREPARING = auto()
    RUNNING = auto()
    FINISHED = auto()
    FAILED = auto()


@asynccontextmanager
async def fifo_callback() -> AsyncGenerator[tuple[Path, AsyncGenerator[dict]]]:
    with TemporaryDirectory() as tempdir:
        fifo = Path(tempdir) / "progress.fifo"
        os.mkfifo(fifo)

        async def reader() -> AsyncGenerator[dict]:
            async with await anyio.open_file(fifo, "r") as f:
                async for line in f:
                    yield yaml.safe_load(line)

        yield fifo, reader()


class Job:
    id: str
    training: Training
    progress: float
    eta: datetime | None = None
    status: Status

    def __init__(self: Self, training: Training):
        self.id = str(uuid.uuid4())
        self.training = training
        self.progress = 0
        self.status = Status.QUEUED

    async def run(self, sem: anyio.Semaphore) -> None:
        self.status = Status.PREPARING
        logger.info(f"Job {self.id} started")
        try:
            config = deepcopy(base_config)
            config["config"]["name"] = self.id
            config["config"]["process"][0]["train"] = yaml.safe_load(
                self.training.config
            )
            async with fifo_callback() as (fifo, cb):
                config["config"]["process"][0]["progress_file"] = fifo.as_posix()
                job = get_job(config, name=self.id)

                async def _cb() -> None:
                    async for progress in cb:
                        self.status = Status.RUNNING
                        n = progress["n"]
                        total = progress["total"]
                        elapsed = progress["elapsed"]
                        perc = n / total * 100
                        eta = datetime.now(tz=timezone.utc) + timedelta(
                            seconds=elapsed * (total - n)
                        )
                        self.progress = perc
                        self.eta = eta
                        logger.info("Progress %d%% %d/%d eta: %s", perc, n, total, eta)

                async with anyio.create_task_group() as tg:
                    tg.start_soon(anyio.to_thread.run_sync, job.run)
                    tg.start_soon(_cb)
            await anyio.to_thread.run_sync(job.cleanup)
        except Exception:
            logger.exception("Error running job")
            self.status = Status.FAILED
        else:
            self.progress = 100.0
            self.status = Status.FINISHED
        finally:
            sem.release()
        logger.info(f"Job {self.id} {self.status.name.lower()}")


class JobControl:
    job_lock: anyio.Lock
    jobs: dict[str, Job]
    _tg_proto: TaskGroup
    tg: TaskGroup
    gpu_sem: anyio.Semaphore
    job_stream_receive: ObjectReceiveStream[str]
    job_stream_send: ObjectSendStream[str]

    def __init__(self: Self, concurrent_jobs: int = 1):
        self.job_lock = anyio.Lock()
        self.jobs = {}
        self.gpu_sem = anyio.Semaphore(concurrent_jobs)
        self.job_stream_send, self.job_stream_receive = (
            anyio.create_memory_object_stream[str](max_buffer_size=math.inf)
        )

    async def enqueue_job(self: Self, job: Job) -> None:
        async with self.job_lock:
            self.jobs[job.id] = job
            await self.job_stream_send.send(job.id)
        logger.info(f"Job {job.id} enqueued")

    async def get_jobs(self: Self) -> Iterable[Job]:
        async with self.job_lock:
            return self.jobs.values()

    async def get_job(self: Self, id: str) -> Job | None:
        async with self.job_lock:
            if id not in self.jobs:
                return None
            return deepcopy(self.jobs.get(id))

    async def _queue_worker(self: Self) -> None:
        async for id in self.job_stream_receive:
            await self.gpu_sem.acquire()
            async with self.job_lock:
                job = self.jobs[id]
            self.tg.start_soon(job.run, self.gpu_sem)

    async def __aenter__(self: Self):
        self._tg_proto = anyio.create_task_group()
        self.tg = await self._tg_proto.__aenter__()
        self.tg.start_soon(self._queue_worker)
        return self

    async def __aexit__(
        self: Self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.job_stream_send.aclose()
        await self._tg_proto.__aexit__(exc_type, exc_val, exc_tb)


job_control = JobControl()


@app.post("/train")
async def train(job: Training) -> dict[str, str]:
    j = Job(job)
    await job_control.enqueue_job(j)
    return {"id": j.id}


@app.get("/trainings")
async def list_trainings() -> list:
    jobs = await job_control.get_jobs()
    return [{"id": j.id, "config": j.training.config, "status": j.status} for j in jobs]


@app.get("/trainings/{id}")
async def get_training(id: str) -> dict:
    j = await job_control.get_job(id)
    if j is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "id": j.id,
        "config": j.training.config,
        "status": j.status,
        "progress": j.progress,
        "eta": j.eta.strftime("%Y-%m-%dT%H:%M:%S.%fZ") if j.eta is not None else None,
    }


@app.get("/trainings/{id}/result")
async def get_training_result(
    id: str, background_tasks: BackgroundTasks
) -> FileResponse:
    path = Path("./output") / id / f"{id}.safetensors"
    j = await job_control.get_job(id)
    if j is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if j.status != Status.FINISHED:
        raise HTTPException(status_code=400, detail="Job not finished")
    if not path.exists():
        raise HTTPException(status_code=410, detail="Result file not found")
    background_tasks.add_task(path.unlink)
    return FileResponse(path)


@app.websocket("/trainings/{id}/progress")
async def progress(ws: WebSocket, id: str):
    j = await job_control.get_job(id)
    if j is None:
        raise HTTPException(status_code=404, detail="Job not found")
    await ws.accept()
    await ws.send_json(
        {
            "progress": j.progress,
            "status": j.status,
            "eta": j.eta.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            if j.eta is not None
            else None,
        }
    )
    last = (j.progress, j.status)
    while True:
        j = await job_control.get_job(id)
        if j is None:
            raise HTTPException(status_code=404, detail="Job not found")
        current = (j.progress, j.status)
        if current != last:
            await ws.send_json(
                {
                    "progress": j.progress,
                    "status": j.status,
                    "eta": j.eta.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                    if j.eta is not None
                    else None,
                }
            )
            last = current
        if j.status in {Status.FINISHED, Status.FAILED}:
            break
        await asyncio.sleep(1)
    await ws.close()


@app.get("/status")
async def status() -> dict:
    jobs = await job_control.get_jobs()
    stats = Counter(j.status for j in jobs)
    for status in Status:
        stats[status] = stats.get(status, 0)
    return stats


async def main():
    async with job_control:
        config = Config()
        config.bind = ["0.0.0.0:8000"]
        config.worker_class = "trio"
        await anycorn.serve(
            app,  # pyright: ignore[reportArgumentType]
            config,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    rich.traceback.install()
    anyio.run(main)
