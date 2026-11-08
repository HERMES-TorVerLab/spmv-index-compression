#!/usr/bin/env python3
"""device_profile.py — the machine constants of a campaign, read from results/<device>/device.txt.

The file is written by build/device_report on the machine that ran the campaign, so the roofs and
the peaks of the analysis follow the measurements instead of being constants in a script. Import
it from the analysis scripts; it has no dependencies beyond the standard library.
"""
import re
from pathlib import Path


class DeviceProfile:
    """The device-dependent numbers of the results section: the memory roof, the compute ceiling of
    every precision, the L2 capacity that decides cache residency, and the machine named in the
    setup paragraph.

    read_bw_gbs (pure read) and triad_bw_gbs (read-read-write) are not interchangeable: SpMV writes
    only O(rows), so the read number is its roof, and it is the one used here."""

    REQUIRED = ("name", "read_bw_gbs", "peak_fp64_gflops", "peak_fp32_gflops", "peak_fp16_gflops")

    def __init__(self, path):
        values = {}
        for line in Path(path).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
        missing = [key for key in self.REQUIRED if key not in values]
        if missing:
            raise SystemExit(f"{path}: no {', '.join(missing)}; regenerate it with build/device_report")
        self.path = Path(path)
        self.name = values["name"]
        self.arch = values.get("arch", "")
        self.cc = values.get("cc", "")
        self.cuda = values.get("cuda", "")
        self.cusparse = values.get("cusparse", "")
        self.l2_mib = float(values.get("l2_mib", 0.0))
        self.multiprocessors = float(values.get("sm", 0.0))
        self.clock_ghz = float(values.get("clock_ghz", 0.0))
        # maxThreadsPerMultiProcessor, which is 1024 on Turing and 2048 on Ampere. With the threads
        # per row of the kernel it fixes how many rows of the matrix are in flight at once, and that
        # is what decides whether the longest row of a matrix outlasts all the others.
        self.threads_per_sm = float(values.get("threads_per_sm", 0.0))
        self.read_bw_gbs = float(values["read_bw_gbs"])
        # The vendor figure and the triad, both of which the Background section quotes beside the
        # pure read to say how much of the roof a real stream reaches.
        self.theoretical_bw_gbs = float(values.get("theoretical_bw_gbs", 0.0))
        self.triad_bw_gbs = float(values.get("triad_bw_gbs", 0.0))
        self.peak_gflops = {precision: float(values["peak_" + precision + "_gflops"])
                            for precision in ("fp64", "fp32", "fp16")}
        self.tensor_fp16_gflops = float(values.get("peak_tc_fp16_gflops", 0.0))
        # "16~GB GDDR6": the capacity the vendor advertises, which the query rounds down; the
        # derived form is used when device.txt does not carry the line.
        self.memory = values.get("memory") or f"{float(values.get('global_mem_gib', 0.0)):.0f}~GB"

    def issue_rate(self):
        """Thread-instructions per second, which prices the work a layout spends decoding: four
        schedulers per multiprocessor, one instruction each per cycle, thirty-two threads per warp.
        Zero when device.txt carries neither the multiprocessor count nor the clock."""
        return self.multiprocessors * self.clock_ghz * 1e9 * 4.0 * 32.0

    def resident_threads(self):
        """Threads the whole device keeps resident. Zero when device.txt predates the field, in which
        case the callers fall back to a documented default rather than guessing an architecture."""
        return self.multiprocessors * self.threads_per_sm

    def description(self):
        """"an NVIDIA A100 (Ampere, compute capability 8.0, 64~GB HBM2e)", for the setup paragraph."""
        parts = [part for part in (self.arch, f"compute capability {self.cc}" if self.cc else "",
                                   self.memory) if part]
        # device_report copies the name the driver reports, and on some cards that name already
        # begins with the vendor, which would otherwise read "an NVIDIA NVIDIA A100"
        name = re.sub(r"^NVIDIA\s+", "", self.name, flags=re.IGNORECASE)
        return f"an NVIDIA {name} ({', '.join(parts)})"

    def toolkit(self):
        versions = [text for text in (f"CUDA {self.cuda}" if self.cuda else "",
                                      f"cuSPARSE {self.cusparse}" if self.cusparse else "") if text]
        return " with " + " and ".join(versions) if versions else ""

    def summary(self):
        return (f"{self.name} ({self.arch} {self.cc}): read roof {self.read_bw_gbs:.1f} GB/s, "
                f"L2 {self.l2_mib:.0f} MiB, FP64 peak {self.peak_gflops['fp64']:.0f} GFLOP/s")


def load_device(csv_path, explicit_path=None):
    """device.txt of the campaign: the one given on the command line, else the one next to the CSV."""
    path = Path(explicit_path) if explicit_path else Path(csv_path).parent / "device.txt"
    if not path.exists():
        raise SystemExit(f"{path} not found: run build/device_report > {path} on the machine that "
                         f"produced {csv_path}")
    return DeviceProfile(path)
