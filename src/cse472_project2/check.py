"""Display computer specifications and check optional training prerequisites."""

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PACKAGES = ("torch", "transformers", "peft", "accelerate", "datasets",
            "sentence-transformers", "numpy", "matplotlib")


def command(args):
    """Run a read-only hardware query; unavailable tools are nonfatal."""
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              errors="replace", timeout=15, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def hardware():
    info = {"os": platform.platform(), "cpu": platform.processor() or "Unknown",
            "logical_cpus": os.cpu_count(), "ram_gib": None}
    if sys.platform == "win32":
        raw = command(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                       "$cpu = Get-CimInstance Win32_Processor; "
                       "$pc = Get-CimInstance Win32_ComputerSystem; "
                       "@{cpu=($cpu.Name -join ', '); ram=$pc.TotalPhysicalMemory} "
                       "| ConvertTo-Json -Compress"])
        if raw:
            try:
                data = json.loads(raw)
                info["cpu"] = data.get("cpu") or info["cpu"]
                info["ram_gib"] = round(float(data["ram"]) / 1024**3, 1)
            except (ValueError, KeyError, TypeError):
                pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    info["ram_gib"] = round(int(line.split()[1]) / 1024**2, 1)
        except (OSError, ValueError):
            pass
    elif sys.platform == "darwin":
        info["cpu"] = command(["sysctl", "-n", "machdep.cpu.brand_string"]) or info["cpu"]
        raw = command(["sysctl", "-n", "hw.memsize"])
        if raw and raw.isdigit():
            info["ram_gib"] = round(int(raw) / 1024**3, 1)
    return info


def inspect_environment():
    report = {"hardware": hardware(), "python": platform.python_version(),
              "python_executable": sys.executable,
              "python_supported": (3, 11) <= sys.version_info[:2] < (3, 14),
              "packages": {}, "nvidia_gpus": [], "torch_gpus": [],
              "cuda_available": False, "torch_cuda_version": None,
              "torch_error": None}
    disk = shutil.disk_usage(Path.cwd())
    report["disk"] = {"path": str(Path.cwd()), "free_gib": round(disk.free / 1024**3, 1)}
    for name in PACKAGES:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None

    raw = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                   "--format=csv,noheader,nounits"])
    report["nvidia_query_available"] = raw is not None
    if raw:
        for row in csv.reader(raw.splitlines(), skipinitialspace=True):
            if len(row) != 3:
                continue
            try:
                memory = round(float(row[1]) / 1024, 2)
            except ValueError:
                memory = None
            report["nvidia_gpus"].append({"name": row[0], "vram_gib": memory,
                                           "driver": row[2]})
    if report["packages"]["torch"]:
        try:
            import torch
            report["cuda_available"] = torch.cuda.is_available()
            report["torch_cuda_version"] = torch.version.cuda
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                with torch.cuda.device(index):
                    bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
                report["torch_gpus"].append({"name": props.name,
                    "vram_gib": round(props.total_memory / 1024**3, 2), "bf16": bf16})
        except (ImportError, OSError, RuntimeError, AssertionError, TypeError) as error:
            report["torch_error"] = str(error)

    issues = []
    if not report["python_supported"]:
        issues.append("Use Python 3.12 (supported: 3.11-3.13). Select the project's .venv interpreter in VS Code.")
    missing = [name for name, version in report["packages"].items() if version is None]
    if missing:
        issues.append("Missing packages: " + ", ".join(missing) +
                      ". On the training PC run: uv sync --locked --extra experiment")
    if report["torch_error"]:
        issues.append("PyTorch inspection failed: " + report["torch_error"])
    if not report["cuda_available"]:
        issues.append("CUDA is not available through this Python environment. Check the NVIDIA driver and CUDA-enabled PyTorch installation.")
    # Nominal 24 GB cards may expose slightly less usable memory.
    if not any(gpu["vram_gib"] >= 23 and gpu["bf16"] for gpu in report["torch_gpus"]):
        issues.append("Training requires a nominal 24 GB or larger CUDA GPU with native bf16 support, verified through PyTorch.")
    report["issues"] = issues
    report["training_prerequisites_met"] = not issues
    return report


def display(report):
    hw = report["hardware"]
    print("COMPUTER SPECIFICATIONS")
    print(f"  OS: {hw['os']}")
    print(f"  CPU: {hw['cpu']}")
    print(f"  Logical CPU cores: {hw['logical_cpus']}")
    ram = f"{hw['ram_gib']} GiB" if hw['ram_gib'] is not None else "Unavailable"
    print(f"  RAM: {ram}")
    print(f"  Free disk space: {report['disk']['free_gib']} GiB ({report['disk']['path']})")
    for gpu in report["nvidia_gpus"]:
        memory = f"{gpu['vram_gib']} GiB" if gpu['vram_gib'] is not None else "unknown VRAM"
        print(f"  NVIDIA GPU: {gpu['name']} | {memory} | Driver {gpu['driver']}")
    if not report["nvidia_gpus"]:
        print("  NVIDIA GPU: unavailable via nvidia-smi (does not prove no GPU is installed).")
    print("\nPYTHON ENVIRONMENT")
    print(f"  Python: {report['python']}")
    print(f"  Interpreter: {report['python_executable']}")
    for name, version in report["packages"].items():
        print(f"  {name}: {version or 'Not installed in this interpreter'}")
    print(f"  PyTorch CUDA available: {report['cuda_available']}")
    print(f"  PyTorch CUDA build: {report['torch_cuda_version'] or 'Unavailable'}")
    for gpu in report["torch_gpus"]:
        print(f"  PyTorch GPU: {gpu['name']} | {gpu['vram_gib']} GiB | Native bf16: {gpu['bf16']}")
    ready = report["training_prerequisites_met"]
    print("\nTRAINING CHECK: " + ("Basic prerequisites met" if ready else "Not ready"))
    for issue in report["issues"]:
        print(f"  - {issue}")
    print("  Hardware is reported separately from installed Python packages.")
    print("  This check does not download models or run training. Model access and actual memory fit must be tested separately.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON.")
    parser.add_argument("--require-training", action="store_true",
                        help="Exit with status 1 if training prerequisites are missing.")
    args = parser.parse_args()
    report = inspect_environment()
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        display(report)
    if args.require_training and not report["training_prerequisites_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
