"""This-machine rail panel: CPU load and temperature, GPU, memory, fans, read from fake system trees."""

from __future__ import annotations

import asyncio

from mwm_harness import hardware
from mwm_harness.hardware import HardwareMonitor, cpu_temperature, parse_gpu_apps, parse_nvidia
from mwm_harness.web.server import create_app
from starlette.testclient import TestClient
from tests.test_panel import BASE, HOST, PORT, TOKEN, until

MEMINFO = """MemTotal:       40763904 kB
MemFree:        14000000 kB
MemAvailable:   20381952 kB
SwapTotal:      16777216 kB
SwapFree:        4194304 kB
"""
NVIDIA = "NVIDIA GeForce RTX 3060 Laptop GPU, 61, 25, 2385, 6144, 19.51, 80.00, 1282, [N/A]\n"
APPS = "1714973, /opt/llama/bin/llama-server, 4032\n5635, /usr/bin/python3.11, 470\n"


def fake_proc(tmp_path, busy: int, idle: int):
    proc = tmp_path / "proc"
    proc.mkdir(exist_ok=True)
    # user nice system idle iowait irq softirq steal
    (proc / "stat").write_text(f"cpu  {busy} 0 0 {idle} 0 0 0 0 0 0\ncpu0 1 0 0 1 0 0 0 0 0 0\n")
    (proc / "meminfo").write_text(MEMINFO)
    (proc / "cpuinfo").write_text(
        "model name\t: Test CPU @ 2.70GHz\ncpu MHz\t\t: 3000.0\nmodel name\t: Test CPU\ncpu MHz\t\t: 3400.0\n"
    )
    return proc


def fake_hwmon(tmp_path, package_millic: int = 95000):
    root = tmp_path / "hwmon"
    core = root / "hwmon6"
    core.mkdir(parents=True)
    (core / "name").write_text("coretemp\n")
    for n, (label, value) in enumerate(
        [("Package id 0", package_millic), ("Core 0", 80000), ("Core 1", 91000)], start=1
    ):
        (core / f"temp{n}_label").write_text(label)
        (core / f"temp{n}_input").write_text(str(value))
    fans = root / "hwmon5"
    fans.mkdir()
    (fans / "name").write_text("msi_wmi_platform")
    (fans / "fan1_input").write_text("3609")
    (root / "hwmon2").mkdir()
    (root / "hwmon2" / "name").write_text("BAT1")  # no temps, no fans: left out
    return root


def test_nvidia_rows_parse_and_na_fields_become_none():
    [gpu] = parse_nvidia(NVIDIA)
    assert gpu["name"] == "NVIDIA GeForce RTX 3060 Laptop GPU"
    assert (gpu["temp"], gpu["util"], gpu["mem_used"], gpu["mem_total"]) == (61, 25, 2385, 6144)
    assert gpu["fan"] is None  # laptop GPUs report [N/A]
    assert parse_nvidia("garbage line\n") == []
    apps = parse_gpu_apps(APPS)
    assert apps[0] == {"pid": 1714973, "name": "llama-server", "mem": 4032}


def test_cpu_temperature_prefers_the_package_sensor_and_reports_the_hottest_core(tmp_path):
    chips = hardware.sensors(fake_hwmon(tmp_path))
    assert [c["chip"] for c in chips] == ["msi_wmi_platform", "coretemp"]
    assert cpu_temperature(chips) == (95.0, 91.0, "coretemp")
    assert cpu_temperature([]) == (None, None, "")


def test_levels_mark_warm_and_hot():
    assert hardware.level(84.9, hardware.CPU_TEMP_LEVELS) == "ok"
    assert hardware.level(85, hardware.CPU_TEMP_LEVELS) == "warn"
    assert hardware.level(95, hardware.CPU_TEMP_LEVELS) == "hot"
    assert hardware.level(None, hardware.GPU_TEMP_LEVELS) == "unknown"


def test_snapshot_from_fake_trees_load_is_the_difference_between_reads(tmp_path, monkeypatch):
    proc = fake_proc(tmp_path, busy=100, idle=900)
    monitor = HardwareMonitor(tmp_path, proc=proc, hwmon=fake_hwmon(tmp_path))
    fake_proc(tmp_path, busy=400, idle=1000)  # 300 busy of 400 new jiffies = 75 %

    async def fake_run(*args):
        return APPS if "--query-compute-apps" in args[1] else NVIDIA

    monkeypatch.setattr(hardware, "_run", fake_run)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    data = asyncio.run(monitor.snapshot())
    cpu = data["cpu"]
    assert cpu["util"] == 75.0 and cpu["temp"] == 95.0 and cpu["level"] == "hot"
    assert cpu["mhz"] == 3200 and cpu["model"] == "Test CPU @ 2.70GHz"
    assert data["memory"]["used"] == (40763904 - 20381952) * 1024
    assert data["memory"]["swap_used"] == (16777216 - 4194304) * 1024
    assert data["gpus"][0]["level"] == "ok" and data["gpu_apps"][0]["name"] == "llama-server"
    assert data["missing"] == {}
    lines = hardware.summary_lines(data)
    assert lines[0].startswith("CPU  75.0% of") and "95 C package (hot)" in lines[0]
    assert any("llama-server" in line and "4032 MiB" in line for line in lines)


def test_missing_gpu_and_sensors_are_named_never_guessed(tmp_path, monkeypatch):
    empty = tmp_path / "nohwmon"
    empty.mkdir()
    monitor = HardwareMonitor(tmp_path, proc=fake_proc(tmp_path, 1, 1), hwmon=empty)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
    data = asyncio.run(monitor.snapshot())
    assert data["cpu"]["temp"] is None and data["cpu"]["level"] == "unknown"
    assert data["gpus"] == [] and "nvidia-smi is not installed" in data["missing"]["gpu"]
    assert "no CPU sensor" in data["missing"]["cpu_temp"]


def test_a_failing_nvidia_smi_is_reported(tmp_path, monkeypatch):
    monitor = HardwareMonitor(tmp_path, proc=fake_proc(tmp_path, 1, 1), hwmon=fake_hwmon(tmp_path))

    async def broken(*args):
        raise RuntimeError("NVIDIA-SMI has failed because it couldn't communicate with the driver")

    monkeypatch.setattr(hardware, "_run", broken)
    monkeypatch.setattr(hardware.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    data = asyncio.run(monitor.snapshot())
    assert data["gpus"] == [] and "couldn't communicate" in data["missing"]["gpu"]


def test_the_page_can_ask_for_hardware(make_session):
    session, _, _ = make_session([])
    app = create_app(session, {session.model.id: session.model}, TOKEN, PORT)

    async def fake_snapshot():
        return {"type": "Hardware", "cpu": {"temp": 70.0}}

    app.state.panel.hardware.snapshot = fake_snapshot
    with (
        TestClient(app, base_url=BASE) as client,
        client.websocket_connect(f"/ws?token={TOKEN}", headers=HOST) as ws,
    ):
        until(ws, "Tabs")
        ws.send_json({"type": "hardware"})
        reply = until(ws, "Hardware")[-1]
        assert reply["cpu"]["temp"] == 70.0 and "tab" not in reply
