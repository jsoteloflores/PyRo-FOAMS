"""Headless processing-preview ownership and latest-request execution."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np

FrozenParams = Tuple[Tuple[str, Any], ...]


def _freeze(params: Mapping[str, Any]) -> FrozenParams:
    return tuple(sorted((str(key), value) for key, value in params.items()))


def _odd_kernel(value: Any, disabled_below: int = 1) -> int:
    kernel = int(value)
    if kernel < disabled_below:
        return 0
    kernel = max(1, kernel)
    return kernel if kernel % 2 else kernel + 1


@dataclass(frozen=True)
class CommittedProcessingConfig:
    """Immutable, validated processing input independent of Tk control state."""

    threshold: FrozenParams
    separation: FrozenParams

    @classmethod
    def create(
        cls,
        threshold: Mapping[str, Any],
        separation: Mapping[str, Any],
    ) -> CommittedProcessingConfig:
        return cls(_freeze(threshold), _freeze(separation))

    def threshold_dict(self) -> dict[str, Any]:
        return dict(self.threshold)

    def separation_dict(self) -> dict[str, Any]:
        return dict(self.separation)


def normalize_effective_parameters(
    threshold: Mapping[str, Any], separation: Mapping[str, Any]
) -> Tuple[FrozenParams, FrozenParams]:
    """Return only values that can affect core processing output."""
    threshold_effective = dict(threshold)
    threshold_effective.pop("overlayAlpha", None)
    for key in ("medianK", "gaussianK"):
        if key in threshold_effective:
            threshold_effective[key] = _odd_kernel(
                threshold_effective[key], disabled_below=3
            )
    if not threshold_effective.get("useCLAHE", False):
        threshold_effective.pop("claheClip", None)
        threshold_effective.pop("claheTile", None)
    if not threshold_effective.get("applyOpenClose", False):
        threshold_effective.pop("morphK", None)
    elif "morphK" in threshold_effective:
        threshold_effective["morphK"] = _odd_kernel(threshold_effective["morphK"])

    method = str(threshold_effective.get("method", "otsu"))
    active_method_keys = {
        "adaptive": {"adaptiveBlock", "adaptiveC"},
        "percentile": {"percentile"},
        "pick": {"pickValue", "pickTolerance"},
        "otsu": set(),
    }.get(method, set())
    for key in ("adaptiveBlock", "adaptiveC", "percentile", "pickValue", "pickTolerance"):
        if key not in active_method_keys:
            threshold_effective.pop(key, None)
    if method == "adaptive" and "adaptiveBlock" in threshold_effective:
        threshold_effective["adaptiveBlock"] = max(
            3, _odd_kernel(threshold_effective["adaptiveBlock"])
        )
    if method == "percentile" and "percentile" in threshold_effective:
        threshold_effective["percentile"] = float(
            np.clip(float(threshold_effective["percentile"]), 0.0, 100.0)
        )
    if method == "pick" and "pickTolerance" in threshold_effective:
        threshold_effective["pickTolerance"] = max(
            0, int(threshold_effective["pickTolerance"])
        )

    separation_effective = dict(separation)
    separation_effective.pop("overlayAlpha", None)
    if separation_effective.get("method", "none") != "watershed":
        for key in ("distanceBlurK", "peakMinDistance", "peakRelThreshold"):
            separation_effective.pop(key, None)
    else:
        if "distanceBlurK" in separation_effective:
            separation_effective["distanceBlurK"] = _odd_kernel(
                separation_effective["distanceBlurK"], disabled_below=3
            )
        if "peakMinDistance" in separation_effective:
            separation_effective["peakMinDistance"] = max(
                1, int(separation_effective["peakMinDistance"])
            )
        if "peakRelThreshold" in separation_effective:
            separation_effective["peakRelThreshold"] = float(
                np.clip(float(separation_effective["peakRelThreshold"]), 0.0, 1.0)
            )
    if "minAreaPx" in separation_effective:
        separation_effective["minAreaPx"] = max(
            1, int(separation_effective["minAreaPx"])
        )
    return _freeze(threshold_effective), _freeze(separation_effective)


@dataclass(frozen=True)
class ProcessingKey:
    image_id: str
    image_revision: int
    threshold: FrozenParams
    separation: FrozenParams


def make_processing_key(
    image_id: str,
    image_revision: int,
    threshold: Mapping[str, Any],
    separation: Mapping[str, Any],
) -> ProcessingKey:
    normalized_threshold, normalized_separation = normalize_effective_parameters(
        threshold, separation
    )
    return ProcessingKey(
        image_id=str(image_id),
        image_revision=int(image_revision),
        threshold=normalized_threshold,
        separation=normalized_separation,
    )


@dataclass(frozen=True)
class PreviewRequest:
    key: ProcessingKey
    generation: int
    image: np.ndarray
    threshold: Mapping[str, Any]
    separation: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        key: ProcessingKey,
        generation: int,
        image: np.ndarray,
        threshold: Mapping[str, Any],
        separation: Mapping[str, Any],
    ) -> PreviewRequest:
        read_only_image = image.view()
        read_only_image.flags.writeable = False
        return cls(
            key=key,
            generation=generation,
            image=read_only_image,
            threshold=MappingProxyType(dict(threshold)),
            separation=MappingProxyType(dict(separation)),
        )


@dataclass(frozen=True)
class PreviewCompletion:
    key: ProcessingKey
    generation: int
    binary: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    error: Optional[BaseException] = None


class PreviewOwnership:
    """Track which result may be displayed or applied for the desired state."""

    def __init__(self) -> None:
        self.desired_key: Optional[ProcessingKey] = None
        self.generation = 0
        self.result: Optional[PreviewCompletion] = None
        self.pending = False

    def set_desired(self, key: ProcessingKey) -> int:
        if key != self.desired_key:
            self.generation += 1
            self.desired_key = key
            self.result = None
        return self.generation

    def request_started(self) -> int:
        self.generation += 1
        self.pending = True
        self.result = None
        return self.generation

    def invalidate(self) -> None:
        self.generation += 1
        self.pending = False
        self.result = None

    def accept(self, completion: PreviewCompletion) -> bool:
        if (
            completion.generation != self.generation
            or completion.key != self.desired_key
        ):
            return False
        self.pending = False
        if completion.error is not None:
            self.result = None
            return False
        self.result = completion
        return True

    @property
    def can_apply(self) -> bool:
        return (
            not self.pending
            and self.result is not None
            and self.result.key == self.desired_key
            and self.result.generation == self.generation
            and self.result.binary is not None
        )


class LatestPreviewWorker:
    """Run one request and retain only the newest request submitted while busy."""

    def __init__(
        self,
        compute: Callable[[PreviewRequest], Tuple[np.ndarray, Optional[np.ndarray]]],
    ) -> None:
        self._compute = compute
        self._condition = threading.Condition()
        self._pending: Optional[PreviewRequest] = None
        self._running = False
        self._closed = False
        self.results: queue.Queue[PreviewCompletion] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="preview-worker")
        self._thread.start()

    def submit(self, request: PreviewRequest) -> None:
        with self._condition:
            if self._closed:
                return
            self._pending = request
            self._condition.notify()

    @property
    def has_work(self) -> bool:
        with self._condition:
            return self._running or self._pending is not None

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify_all()

    def discard_pending(self) -> None:
        """Discard queued work without interrupting the currently running call."""
        with self._condition:
            self._pending = None

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                request = self._pending
                self._pending = None
                self._running = True
            try:
                binary, labels = self._compute(request)
                completion = PreviewCompletion(
                    key=request.key,
                    generation=request.generation,
                    binary=binary,
                    labels=labels,
                )
            except BaseException as exc:
                completion = PreviewCompletion(
                    key=request.key,
                    generation=request.generation,
                    error=exc,
                )
            with self._condition:
                if not self._closed:
                    self.results.put(completion)
                self._running = False
