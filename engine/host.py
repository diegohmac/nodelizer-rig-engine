"""This computer's processor and memory, for the cpu and mps devices. Standard library only.

On a Mac the GPU shares the computer's memory (unified memory), so a Mac's `card` facts are the chip's
name, the memory installed, and the memory available now (what Activity Monitor counts as not used:
installed less app memory, wired memory and the compressor). The peak a step used is the process's
lifetime peak footprint (proc_pid_rusage), which counts what Metal allocated for MPS as well. On Linux
(the Docker tests) the same facts come from /proc.

A Mac's memory mode (PROTOCOL.md, Devices) is chosen here too: `fast` keeps macOS's allocator cache on,
`lean` turns it off with MallocLargeCache=0, which halves the rig step's footprint and makes it about
2.5 times slower. Both give the same rig bit for bit.
"""
import ctypes
import ctypes.util
import os
import platform
import sys


def _sysctl(name, kind):
    libc = ctypes.CDLL(None)
    if kind is str:
        size = ctypes.c_size_t(0)
        libc.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0)
        buffer = ctypes.create_string_buffer(size.value)
        libc.sysctlbyname(name.encode(), buffer, ctypes.byref(size), None, 0)
        return buffer.value.decode("utf-8", "replace")
    value = ctypes.c_uint64(0)
    size = ctypes.c_size_t(ctypes.sizeof(value))
    if libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, 0) != 0:
        return None
    return int(value.value)


class _VmStatistics64(ctypes.Structure):
    _fields_ = [("free_count", ctypes.c_uint32), ("active_count", ctypes.c_uint32),
                ("inactive_count", ctypes.c_uint32), ("wire_count", ctypes.c_uint32),
                ("zero_fill_count", ctypes.c_uint64), ("reactivations", ctypes.c_uint64),
                ("pageins", ctypes.c_uint64), ("pageouts", ctypes.c_uint64), ("faults", ctypes.c_uint64),
                ("cow_faults", ctypes.c_uint64), ("lookups", ctypes.c_uint64), ("hits", ctypes.c_uint64),
                ("purges", ctypes.c_uint64), ("purgeable_count", ctypes.c_uint32),
                ("speculative_count", ctypes.c_uint32), ("decompressions", ctypes.c_uint64),
                ("compressions", ctypes.c_uint64), ("swapins", ctypes.c_uint64), ("swapouts", ctypes.c_uint64),
                ("compressor_page_count", ctypes.c_uint32), ("throttled_count", ctypes.c_uint32),
                ("external_page_count", ctypes.c_uint32), ("internal_page_count", ctypes.c_uint32),
                ("total_uncompressed_pages_in_compressor", ctypes.c_uint64)]


def _mac_available(total):
    libc = ctypes.CDLL(None)
    stats = _VmStatistics64()
    count = ctypes.c_uint32(ctypes.sizeof(stats) // 4)
    host = libc.mach_host_self()
    if libc.host_statistics64(host, 4, ctypes.byref(stats), ctypes.byref(count)) != 0:  # HOST_VM_INFO64
        return None
    page = _sysctl("hw.pagesize", int) or 16384
    used = (stats.internal_page_count - stats.purgeable_count + stats.wire_count + stats.compressor_page_count)
    return max(0, total - used * page)


class _RusageInfoV4(ctypes.Structure):
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
        "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
        "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
        "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
        "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
        "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
        "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions",
        "ri_cycles", "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
        "ri_runnable_time")]


def _meminfo():
    found = {}
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                name, value = line.split(":", 1)
                found[name] = int(value.split()[0]) * 1024
    except OSError:
        pass
    return found


def facts():
    """{name, total_bytes, free_bytes} for this computer."""
    if sys.platform == "darwin":
        total = installed_bytes()
        return {"name": _sysctl("machdep.cpu.brand_string", str) or "Apple silicon", "total_bytes": total,
                "free_bytes": _mac_available(total) or 0}
    memory = _meminfo()
    return {"name": f"{platform.machine()} processor, {os.cpu_count()} cores",
            "total_bytes": memory.get("MemTotal", 0), "free_bytes": memory.get("MemAvailable", 0)}


def peak_footprint():
    """The process's peak memory so far: on a Mac its lifetime peak footprint, elsewhere its peak RSS."""
    if sys.platform == "darwin":
        libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        info = _RusageInfoV4()
        if libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) == 0:  # RUSAGE_INFO_V4
            return int(info.ri_lifetime_max_phys_footprint)
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)



GIB = 2**30
MEMORY_MODES = ("fast", "lean")
# Macs sold with 32 GB report a little less than 32 GiB to the system, as 16 GB ones do; 30 GiB or more
# counts as 32 GB. Fast below that would need more than a 16 or 24 GB Mac has free.
FAST_FROM_BYTES = 30 * GIB


def installed_bytes():
    if sys.platform == "darwin":
        return _sysctl("hw.memsize", int) or 0
    return _meminfo().get("MemTotal", 0)


def memory_mode(asked=None, installed=None):
    """The memory mode a Mac's rig step runs in: the asked one (`--memory`), else by installed memory."""
    if asked in MEMORY_MODES:
        return asked
    installed = installed_bytes() if installed is None else installed
    return "fast" if installed >= FAST_FROM_BYTES else "lean"


def running_mode(environ=None):
    """The mode this process was started in: the allocator reads its setting only at start."""
    environ = os.environ if environ is None else environ
    return "lean" if environ.get("MallocLargeCache") == "0" else "fast"


def environment_for(mode, environ):
    """`environ` changed so a process started with it runs in `mode`."""
    changed = dict(environ)
    if mode == "lean":
        changed["MallocLargeCache"] = "0"
    else:
        changed.pop("MallocLargeCache", None)
    return changed
