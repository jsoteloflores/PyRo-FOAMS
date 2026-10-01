"""Headless processing-preview ownership and latest-request execution."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Tuple

import numpy as np

FrozenParams = Tuple[Tuple[str, Any], ...]

THRESHOLD_METHODS = frozenset({"otsu", "adaptive", "percentile", "pick"})
SEPARATION_METHODS = frozenset({"none", "watershed"})
POLARITIES = frozenset({"auto", "poresDarker", "poresBrighter"})


class ProcessingConfigError(ValueError):
    """A processing candidate contains an invalid named field."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(f"{field}: {message}")


def _freeze(params: Mapping[str, Any]) -> FrozenParams:
    return tuple(sorted((str(key), value) for key, value in params.items()))


def _odd_kernel(value: Any, disabled_below: int = 1) -> int:
    kernel = int(value)
    if kernel < disabled_below:
        return 0
    kernel = max(1, kernel)
    return kernel if kernel % 2 else kernel + 1


def _finite_float(params: Mapping[str, Any], field: str) -> None:
    if field not in params:
        return
    value = params[field]
    if isinstance(value, (bool, np.bool_)):
        raise ProcessingConfigError(field, "must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ProcessingConfigError(field, "must be a finite number") from exc
    if not np.isfinite(numeric):
        raise ProcessingConfigError(field, "must be finite")
    params[field] = numeric


def _integral(params: Mapping[str, Any], field: str) -> None:
    if field not in params:
        return
    value = params[field]
    if isinstance(value, (bool, np.bool_)):
        raise ProcessingConfigError(field, "must be an integer, not Boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ProcessingConfigError(field, "must be an integer") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ProcessingConfigError(field, "must be a finite integer")
    params[field] = int(numeric)


def _boolean(params: Mapping[str, Any], field: str) -> None:
    if field in params and not isinstance(params[field], (bool, np.bool_)):
        raise ProcessingConfigError(field, "must be Boolean")
    if field in params:
        params[field] = bool(params[field])


def _validated_parameters(
    threshold: Mapping[str, Any], separation: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    threshold_values = dict(threshold)
    separation_values = dict(separation)

    method = str(threshold_values.get("method", "otsu"))
    if method not in THRESHOLD_METHODS:
        raise ProcessingConfigError("method", f"unsupported threshold method {method!r}")
    threshold_values["method"] = method
    polarity = str(threshold_values.get("polarity", "auto"))
    if polarity not in POLARITIES:
        raise ProcessingConfigError("polarity", f"unsupported value {polarity!r}")
    if "polarity" in threshold_values:
        threshold_values["polarity"] = polarity

    for field in ("useCLAHE", "applyOpenClose"):
        _boolean(threshold_values, field)
    for field in (
        "claheTile", "medianK", "gaussianK", "adaptiveBlock", "adaptiveC",
        "pickValue", "pickTolerance", "morphK",
    ):
        _integral(threshold_values, field)
    for field in ("claheClip", "percentile"):
        _finite_float(threshold_values, field)

    if threshold_values.get("useCLAHE", False):
        if "claheTile" in threshold_values and threshold_values["claheTile"] <= 0:
            raise ProcessingConfigError("claheTile", "must be a positive integer")
        if "claheClip" in threshold_values and threshold_values["claheClip"] <= 0:
            raise ProcessingConfigError("claheClip", "must be positive")
    for field in ("medianK", "gaussianK"):
        if field in threshold_values:
            threshold_values[field] = _odd_kernel(
                threshold_values[field], disabled_below=3
            )
    if "adaptiveBlock" in threshold_values:
        threshold_values["adaptiveBlock"] = max(
            3, _odd_kernel(threshold_values["adaptiveBlock"])
        )
    if "percentile" in threshold_values:
        threshold_values["percentile"] = float(
            np.clip(threshold_values["percentile"], 0.0, 100.0)
        )
    if "pickTolerance" in threshold_values:
        threshold_values["pickTolerance"] = max(
            0, threshold_values["pickTolerance"]
        )
    if "morphK" in threshold_values:
        threshold_values["morphK"] = _odd_kernel(threshold_values["morphK"])

    separation_method = str(separation_values.get("method", "none"))
    if separation_method not in SEPARATION_METHODS:
        raise ProcessingConfigError(
            "method", f"unsupported separation method {separation_method!r}"
        )
    separation_values["method"] = separation_method
    for field in ("fillHoles", "clearBorder"):
        _boolean(separation_values, field)
    for field in ("minAreaPx", "distanceBlurK", "peakMinDistance", "connectivity"):
        _integral(separation_values, field)
    _finite_float(separation_values, "peakRelThreshold")
    if "connectivity" in separation_values and separation_values["connectivity"] not in (4, 8):
        raise ProcessingConfigError("connectivity", "must be 4 or 8")
    if "minAreaPx" in separation_values:
        separation_values["minAreaPx"] = max(1, separation_values["minAreaPx"])
    if "distanceBlurK" in separation_values:
        separation_values["distanceBlurK"] = _odd_kernel(
            separation_values["distanceBlurK"], disabled_below=3
        )
    if "peakMinDistance" in separation_values:
        separation_values["peakMinDistance"] = max(
            1, separation_values["peakMinDistance"]
        )
    if "peakRelThreshold" in separation_values:
        separation_values["peakRelThreshold"] = float(
            np.clip(separation_values["peakRelThreshold"], 0.0, 1.0)
        )
    return threshold_values, separation_values


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
        threshold_values, separation_values = _validated_parameters(
            threshold, separation
        )
        return cls(_freeze(threshold_values), _freeze(separation_values))

    def threshold_dict(self) -> dict[str, Any]:
        return dict(self.threshold)

    def separation_dict(self) -> dict[str, Any]:
        return dict(self.separation)


def normalize_effective_parameters(
    threshold: Mapping[str, Any], separation: Mapping[str, Any]
) -> Tuple[FrozenParams, FrozenParams]:
    """Return only values that can affect core processing output."""
    threshold_effective, separation_effective = _validated_parameters(
        threshold, separation
    )
    threshold_effective.pop("overlayAlpha", None)
    for key in ("medianK", "gaussianK"):
        if key in threshold_effective:
            threshold_effective[key] = _odd_kernel(
                threshold_effective[key], disabled_below=3
            )
    if not threshold_effective.get("useCLAHE", False):
        threshold_effective.pop("useCLAHE", None)
        threshold_effective.pop("claheClip", None)
        threshold_effective.pop("claheTile", None)
    if not threshold_effective.get("applyOpenClose", False):
        threshold_effective.pop("applyOpenClose", None)
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

    def discard_pending(self) -> bool:
        """Discard queued work without interrupting the currently running call."""
        with self._condition:
            discarded = self._pending is not None
            self._pending = None
            return discarded

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
