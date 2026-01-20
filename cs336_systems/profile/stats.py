"""Profiler statistics extraction and data structures."""

from dataclasses import asdict, dataclass, field

from torch.profiler import profile as TorchProfile


@dataclass
class EventStats:
    """Statistics for a single profiler event."""

    name: str
    cuda_time_us: float
    cuda_time_ms: float
    cuda_time_pct: float
    cpu_time_us: float
    cpu_time_ms: float
    count: int
    cuda_time_avg_us: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ProfileStats:
    """Aggregated profiler statistics."""

    total_cuda_time_us: float
    total_cuda_time_ms: float
    total_cpu_time_us: float
    total_cpu_time_ms: float
    num_events: int
    top_events: list[EventStats] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_cuda_time_us": self.total_cuda_time_us,
            "total_cuda_time_ms": self.total_cuda_time_ms,
            "total_cpu_time_us": self.total_cpu_time_us,
            "total_cpu_time_ms": self.total_cpu_time_ms,
            "num_events": self.num_events,
            "top_events": [e.to_dict() for e in self.top_events],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProfileStats":
        return cls(
            total_cuda_time_us=data["total_cuda_time_us"],
            total_cuda_time_ms=data["total_cuda_time_ms"],
            total_cpu_time_us=data["total_cpu_time_us"],
            total_cpu_time_ms=data["total_cpu_time_ms"],
            num_events=data["num_events"],
            top_events=[EventStats(**e) for e in data["top_events"]],
        )


def extract_profiler_stats(prof: TorchProfile, top_n: int = 100) -> ProfileStats:
    """Extract key statistics from torch profiler into ProfileStats."""
    key_averages = prof.key_averages()

    sorted_events = sorted(key_averages, key=lambda e: e.device_time_total, reverse=True)
    total_cuda_time_us = sum(e.device_time_total for e in key_averages)
    total_cpu_time_us = sum(e.self_cpu_time_total for e in key_averages)

    top_events = [
        EventStats(
            name=e.key,
            cuda_time_us=e.device_time_total,
            cuda_time_ms=e.device_time_total / 1000,
            cuda_time_pct=(e.device_time_total / total_cuda_time_us * 100) if total_cuda_time_us > 0 else 0,
            cpu_time_us=e.self_cpu_time_total,
            cpu_time_ms=e.self_cpu_time_total / 1000,
            count=e.count,
            cuda_time_avg_us=e.device_time_total / e.count if e.count > 0 else 0,
        )
        for e in sorted_events[:top_n]
        if e.device_time_total > 0
    ]

    return ProfileStats(
        total_cuda_time_us=total_cuda_time_us,
        total_cuda_time_ms=total_cuda_time_us / 1000,
        total_cpu_time_us=total_cpu_time_us,
        total_cpu_time_ms=total_cpu_time_us / 1000,
        num_events=len(key_averages),
        top_events=top_events,
    )
