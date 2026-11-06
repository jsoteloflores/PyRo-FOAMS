"""Background coordinator for cancellable, revision-tagged draft batches."""

from __future__ import annotations

import queue
import threading
import uuid
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from ..core.batch import process_batch_parallel


@dataclass(frozen=True)
class BatchTargetSnapshot:
    index: int
    image_id: str
    source_revision: int
    draft_revision: int
    image: np.ndarray


@dataclass(frozen=True)
class BatchTargetResult:
    job_id: str
    project_id: Optional[str]
    session_id: Optional[str]
    index: int
    image_id: str
    source_revision: int
    draft_revision: int
    threshold: Tuple[Tuple[str, Any], ...]
    separation: Tuple[Tuple[str, Any], ...]
    binary: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class BatchJobEvent:
    job_id: str
    project_id: Optional[str]
    session_id: Optional[str]
    kind: str
    completed: int
    total: int
    result: Optional[BatchTargetResult] = None
    cancelled: bool = False


def _process_one(
    target: BatchTargetSnapshot,
    threshold: Mapping[str, Any],
    separation: Mapping[str, Any],
    job_id: str,
    project_id: Optional[str],
    session_id: Optional[str],
    frozen_threshold: Tuple[Tuple[str, Any], ...],
    frozen_separation: Tuple[Tuple[str, Any], ...],
) -> BatchTargetResult:
    binaries, labels, _ = process_batch_parallel(
        images=[target.image],
        thresh_params=dict(threshold),
        sep_params=dict(separation),
        measure=False,
        max_workers=1,
    )
    binary = binaries[0] if binaries else None
    if binary is None:
        raise RuntimeError("batch engine returned no mask")
    label_map = labels[0] if labels else None
    return BatchTargetResult(
        job_id=job_id,
        project_id=project_id,
        session_id=session_id,
        index=target.index,
        image_id=target.image_id,
        source_revision=target.source_revision,
        draft_revision=target.draft_revision,
        threshold=frozen_threshold,
        separation=frozen_separation,
        binary=np.asarray(binary).copy(),
        labels=np.asarray(label_map, dtype=np.int32).copy() if label_map is not None else None,
    )


class BatchJobCoordinator:
    """Run each selected target independently and publish results through a queue."""

    def __init__(self) -> None:
        self.events: queue.Queue[BatchJobEvent] = queue.Queue()
        self._lock = threading.Lock()
        self._cancellations: Dict[str, threading.Event] = {}
        self._futures: Dict[str, Tuple[Future, ...]] = {}
        self._threads: Dict[str, threading.Thread] = {}

    def start(
        self,
        *,
        project_id: Optional[str],
        session_id: Optional[str] = None,
        targets: Iterable[BatchTargetSnapshot],
        threshold: Mapping[str, Any],
        separation: Mapping[str, Any],
        max_workers: int,
    ) -> str:
        owned_targets = tuple(targets)
        if not owned_targets:
            raise ValueError("batch requires at least one selected target")
        job_id = uuid.uuid4().hex
        cancellation = threading.Event()
        threshold_copy = dict(threshold)
        separation_copy = dict(separation)
        frozen_threshold = tuple(sorted(threshold_copy.items()))
        frozen_separation = tuple(sorted(separation_copy.items()))
        thread = threading.Thread(
            target=self._run,
            args=(
                job_id,
                project_id,
                session_id,
                owned_targets,
                threshold_copy,
                separation_copy,
                frozen_threshold,
                frozen_separation,
                max(1, min(int(max_workers), len(owned_targets))),
                cancellation,
            ),
            daemon=True,
            name=f"batch-{job_id[:8]}",
        )
        with self._lock:
            self._cancellations[job_id] = cancellation
            self._threads[job_id] = thread
        thread.start()
        return job_id

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            cancellation = self._cancellations.get(job_id)
            futures = self._futures.get(job_id, ())
        if cancellation is None:
            return False
        cancellation.set()
        for future in futures:
            future.cancel()
        return True

    def close(self) -> None:
        with self._lock:
            job_ids = tuple(self._cancellations)
        for job_id in job_ids:
            self.cancel(job_id)

    def _run(
        self,
        job_id: str,
        project_id: Optional[str],
        session_id: Optional[str],
        targets: Tuple[BatchTargetSnapshot, ...],
        threshold: Dict[str, Any],
        separation: Dict[str, Any],
        frozen_threshold: Tuple[Tuple[str, Any], ...],
        frozen_separation: Tuple[Tuple[str, Any], ...],
        max_workers: int,
        cancellation: threading.Event,
    ) -> None:
        total = len(targets)
        completed = 0
        futures_by_target = {}
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="batch-target") as executor:
            for target in targets:
                if cancellation.is_set():
                    break
                future = executor.submit(
                    _process_one,
                    target,
                    threshold,
                    separation,
                    job_id,
                    project_id,
                    session_id,
                    frozen_threshold,
                    frozen_separation,
                )
                futures_by_target[future] = target
            with self._lock:
                self._futures[job_id] = tuple(futures_by_target)
            if cancellation.is_set():
                for future in futures_by_target:
                    future.cancel()

            for future in as_completed(futures_by_target):
                target = futures_by_target[future]
                completed += 1
                if cancellation.is_set():
                    continue
                try:
                    result = future.result()
                except CancelledError:
                    continue
                except Exception as exc:
                    result = BatchTargetResult(
                        job_id=job_id,
                        project_id=project_id,
                        session_id=session_id,
                        index=target.index,
                        image_id=target.image_id,
                        source_revision=target.source_revision,
                        draft_revision=target.draft_revision,
                        threshold=frozen_threshold,
                        separation=frozen_separation,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                self.events.put(
                    BatchJobEvent(
                        job_id=job_id,
                        project_id=project_id,
                        session_id=session_id,
                        kind="target",
                        completed=completed,
                        total=total,
                        result=result,
                    )
                )

        self.events.put(
            BatchJobEvent(
                job_id=job_id,
                project_id=project_id,
                session_id=session_id,
                kind="finished",
                completed=completed,
                total=total,
                cancelled=cancellation.is_set(),
            )
        )
        with self._lock:
            self._cancellations.pop(job_id, None)
            self._futures.pop(job_id, None)
            self._threads.pop(job_id, None)
