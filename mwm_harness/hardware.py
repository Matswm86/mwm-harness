"""Machine rail panel: CPU, GPU, memory, fans and disk of the box the harness runs on.

Linux only, no extra packages: CPU load comes from ``/proc/stat`` (difference between two reads),
memory from ``/proc/meminfo``, temperatures and fans from ``/sys/class/hwmon``, the NVIDIA GPU from
``nvidia-smi``. A source that is missing gives ``None`` and a reason in ``missing``, never a guess.
Read-only: nothing here writes to the system.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from typing import Any

HWMON = Path("/sys/class/hwmon")
PROC = Path("/proc")
CACHE_SECONDS = 1.5  # several tabs and the status strip share one reading
NVIDIA_TIMEOUT = 3.0
GPU_FIELDS = (
    "name",
    "temperature.gpu",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "power.draw",
    "power.limit",
    "clocks.gr",
    "fan.speed",
)
# (warn, hot) in Celsius. Intel laptop CPUs throttle at 100, NVIDIA laptop GPUs near 87.
CPU_TEMP_LEVELS = (85.0, 95.0)
GPU_TEMP_LEVELS = (80.0, 87.0)
CPU_SENSORS = ("coretemp", "k10temp", "zenpower", "cpu_thermal", "acpitz")
CPU_LABELS = ("Package id 0", "Tctl", "Tdie")


def level(value: float | None, levels: tuple[float, float]) -> str:
    if value is None:
        return "unknown"
    return "hot" if value >= levels[1] else "warn" if value >= levels[0] else "ok"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def cpu_times(root: Path = PROC) -> tuple[int, int] | None:
    """(busy, total) jiffies since boot from the aggregate ``cpu`` line."""
    first = _read(root / "stat").splitlines()[:1]
    if not first or not first[0].startswith("cpu "):
        return None
    values = [int(v) for v in first[0].split()[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
    total = sum(values[:8])  # guest time is already inside user/nice
    return total - idle, total


def memory(root: Path = PROC) -> dict[str, Any] | None:
    fields = {}
    for line in _read(root / "meminfo").splitlines():
        key, _, rest = line.partition(":")
        if rest.split():
            fields[key] = int(rest.split()[0]) * 1024
    if "MemTotal" not in fields or "MemAvailable" not in fields:
        return None
    swap_total = fields.get("SwapTotal", 0)
    return {
        "used": fields["MemTotal"] - fields["MemAvailable"],
        "total": fields["MemTotal"],
        "swap_used": swap_total - fields.get("SwapFree", 0),
        "swap_total": swap_total,
    }


def sensors(root: Path = HWMON) -> list[dict[str, Any]]:
    """Every hwmon chip with its temperatures (C) and fans (RPM)."""
    chips = []
    for chip in sorted(root.glob("hwmon*")):
        temps, fans = [], []
        for item in sorted(chip.glob("temp*_input")):
            value = _number(_read(item))
            if value is not None:
                label = _read(item.with_name(item.name.replace("_input", "_label")))
                temps.append(
                    {"label": label or item.name.split("_")[0], "c": round(value / 1000, 1)}
                )
        for item in sorted(chip.glob("fan*_input")):
            value = _number(_read(item))
            if value is not None:
                fans.append({"label": item.name.split("_")[0], "rpm": int(value)})
        if temps or fans:
            chips.append({"chip": _read(chip / "name") or chip.name, "temps": temps, "fans": fans})
    return chips


def cpu_temperature(chips: list[dict[str, Any]]) -> tuple[float | None, float | None, str]:
    """(package, hottest core, sensor name) from the first CPU chip found, in CPU_SENSORS order."""
    by_name = {c["chip"]: c for c in chips}
    for name in CPU_SENSORS:
        chip = by_name.get(name)
        if not chip or not chip["temps"]:
            continue
        package = next((t["c"] for t in chip["temps"] if t["label"] in CPU_LABELS), None)
        cores = [t["c"] for t in chip["temps"] if t["label"].startswith("Core")]
        if package is None:
            package = chip["temps"][0]["c"]
        return package, max(cores) if cores else None, name
    return None, None, ""


def cpu_model(root: Path = PROC) -> str:
    for line in _read(root / "cpuinfo").splitlines():
        if line.startswith("model name"):
            return line.partition(":")[2].strip()
    return ""


def cpu_mhz(root: Path = PROC) -> float | None:
    values = [
        float(line.partition(":")[2])
        for line in _read(root / "cpuinfo").splitlines()
        if line.startswith("cpu MHz")
    ]
    return round(sum(values) / len(values)) if values else None


def parse_nvidia(text: str) -> list[dict[str, Any]]:
    """Rows of ``nvidia-smi --query-gpu=GPU_FIELDS --format=csv,noheader,nounits``."""
    gpus = []
    for line in text.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) != len(GPU_FIELDS):
            continue
        row = dict(zip(GPU_FIELDS, cells, strict=True))

        def num(key: str, row: dict[str, str] = row) -> float | None:
            return _number(row[key])  # "[N/A]" (laptop fans, some power limits) gives None

        gpus.append(
            {
                "name": row["name"],
                "temp": num("temperature.gpu"),
                "util": num("utilization.gpu"),
                "mem_used": num("memory.used"),
                "mem_total": num("memory.total"),
                "power": num("power.draw"),
                "power_limit": num("power.limit"),
                "clock_mhz": num("clocks.gr"),
                "fan": num("fan.speed"),
            }
        )
    return gpus


def parse_gpu_apps(text: str) -> list[dict[str, Any]]:
    """Rows of ``nvidia-smi --query-compute-apps=pid,process_name,used_memory``: who holds VRAM."""
    apps = []
    for line in text.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) == 3 and cells[0].isdigit():
            apps.append(
                {"pid": int(cells[0]), "name": Path(cells[1]).name, "mem": _number(cells[2])}
            )
    return apps


async def _run(*args: str) -> str:
    process = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), NVIDIA_TIMEOUT)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(err.decode(errors="replace").strip() or f"exit {process.returncode}")
    return out.decode(errors="replace")


class HardwareMonitor:
    def __init__(self, disk_path: Path | None = None, proc: Path = PROC, hwmon: Path = HWMON):
        self.disk_path = disk_path or Path.home()
        self.proc, self.hwmon = proc, hwmon
        self._last_cpu = cpu_times(proc)
        self._cache: tuple[float, dict[str, Any]] | None = None
        self._lock = asyncio.Lock()

    def cpu_percent(self) -> float | None:
        now = cpu_times(self.proc)
        before, self._last_cpu = self._last_cpu, now
        if not now or not before or now[1] <= before[1]:
            return None
        return round(100 * (now[0] - before[0]) / (now[1] - before[1]), 1)

    async def gpu(
        self, missing: dict[str, str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not shutil.which("nvidia-smi"):
            missing["gpu"] = "nvidia-smi is not installed (no NVIDIA driver, or another GPU vendor)"
            return [], []
        try:
            gpus = parse_nvidia(
                await _run(
                    "nvidia-smi",
                    f"--query-gpu={','.join(GPU_FIELDS)}",
                    "--format=csv,noheader,nounits",
                )
            )
            apps = parse_gpu_apps(
                await _run(
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                )
            )
        except (TimeoutError, RuntimeError, OSError) as exc:
            missing["gpu"] = f"nvidia-smi failed: {exc or type(exc).__name__}"
            return [], []
        for gpu in gpus:
            gpu["level"] = level(gpu["temp"], GPU_TEMP_LEVELS)
        return gpus, apps

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            if self._cache and time.monotonic() - self._cache[0] < CACHE_SECONDS:
                return self._cache[1]
            missing: dict[str, str] = {}
            chips = sensors(self.hwmon)
            package, hottest, sensor = cpu_temperature(chips)
            if package is None:
                missing["cpu_temp"] = (
                    f"no CPU sensor in {self.hwmon} (looked for {', '.join(CPU_SENSORS)})"
                )
            gpus, apps = await self.gpu(missing)
            try:
                disk = shutil.disk_usage(self.disk_path)
                disk_row: dict[str, Any] | None = {
                    "path": str(self.disk_path),
                    "used": disk.used,
                    "total": disk.total,
                }
            except OSError as exc:
                disk_row, missing["disk"] = None, str(exc)
            try:
                load = list(os.getloadavg())
            except OSError:
                load = []
            data = {
                "type": "Hardware",
                "ts": time.time(),
                "cpu": {
                    "model": cpu_model(self.proc),
                    "threads": os.cpu_count(),
                    "util": self.cpu_percent(),
                    "mhz": cpu_mhz(self.proc),
                    "temp": package,
                    "hottest_core": hottest,
                    "sensor": sensor,
                    "level": level(package, CPU_TEMP_LEVELS),
                    "load": [round(v, 2) for v in load],
                },
                "memory": memory(self.proc),
                "gpus": gpus,
                "gpu_apps": apps,
                "disk": disk_row,
                "sensors": chips,
                "levels": {"cpu": CPU_TEMP_LEVELS, "gpu": GPU_TEMP_LEVELS},
                "missing": missing,
            }
            self._cache = (time.monotonic(), data)
            return data


def summary_lines(data: dict[str, Any]) -> list[str]:
    """Plain text for ``/hw`` in the terminal and the panel."""
    cpu, mem = data["cpu"], data["memory"]

    def gib(n: float) -> str:
        return f"{n / 2**30:.1f}"

    def deg(v: float | None) -> str:
        return "?" if v is None else f"{v:.0f} C"

    lines = [
        f"CPU  {cpu['util'] if cpu['util'] is not None else '?'}% of {cpu['threads']} threads · "
        f"{deg(cpu['temp'])} package ({cpu['level']}) · hottest core {deg(cpu['hottest_core'])} · "
        f"{cpu['mhz'] or '?'} MHz · load {' '.join(str(v) for v in cpu['load'])}"
    ]
    for gpu in data["gpus"]:
        lines.append(
            f"GPU  {gpu['name']}: {gpu['util'] if gpu['util'] is not None else '?'}% · {deg(gpu['temp'])} ({gpu['level']}) · "
            f"VRAM {gpu['mem_used'] or 0:.0f}/{gpu['mem_total'] or 0:.0f} MiB · {gpu['power'] or 0:.0f} W"
        )
    for app in data["gpu_apps"]:
        lines.append(f"     {app['name']} (pid {app['pid']}) holds {app['mem'] or 0:.0f} MiB")
    if mem:
        lines.append(
            f"RAM  {gib(mem['used'])}/{gib(mem['total'])} GiB · swap {gib(mem['swap_used'])}/{gib(mem['swap_total'])} GiB"
        )
    fans = [f"{c['chip']} {f['label']} {f['rpm']} rpm" for c in data["sensors"] for f in c["fans"]]
    if fans:
        lines.append("Fans " + " · ".join(fans))
    if data["disk"]:
        d = data["disk"]
        lines.append(f"Disk {d['path']}: {gib(d['used'])}/{gib(d['total'])} GiB used")
    lines += [f"missing {key}: {why}" for key, why in data["missing"].items()]
    return lines
