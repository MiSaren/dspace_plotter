"""
Read dSPACE ControlDesk measurement files (.mat, v7.3/HDF5).

Usage
-----
    from lab_plotting import LabDataMat

    data = LabDataMat("PMSyRM_vhz_with_lc_20260820")  # looked up in ./data/
    data.list("theta")             # print signals whose name/path contains "theta"
    t = data.t                     # time vector [s]
    w_M = data["wM_filt"]          # signal by readable name
    i_sa, i_sb, i_sc = data["i_s_abc"]   # vector signal -> stacked rows
    i_s = data.complex("i_s_abc")  # abc -> complex space vector
    psi_s = data.complex("psi_s")  # [0] + 1j*[1]
    data.plot(["w_M_ref", "wM_filt"], "tau_m")   # quick look

    from lab_plotting import measurement_files, transient_window
    for path in measurement_files("recording.mat", plot_all=True):
        with LabDataMat(path) as data:
            origin, limits = transient_window(data, pre_transient=0.2, duration=None)
            # Save measurement figures in path.parent, using origin and limits.

Signal names
------------
Each signal gets a readable key:
  - normal signal          -> its name, e.g. "wM_filt", "psi_s[0]"
  - Simulink port "Out1"   -> name of the block, e.g. "i_s_abc[0]", "psi_f_est"
  - name used many times   -> parent block(s) prepended until it is unique,
                              e.g. "Motor_control/theta_m" and
                              "Sensorless Flux Observer/theta_m"
Any unique end of the full path also works (e.g. "Filter_control/i_c[0]").
The old keys ("Out1_39", "thetam_26", ...) still work, so old scripts run as is.

From the command line:
    python lab_plotting.py PMSyRM_vhz_with_lc_20260820 [filter]
"""
import difflib
import re
from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path(__file__).resolve().parent / "data"
_PORT = re.compile(r"(Out|Value)(\d*)(\[\d+\])?")
_INDEX = re.compile(r"\[(\d+)\]$")


class LabDataMat:
    def __init__(self, fname, flocation=None) -> None:
        self.file_path = self._resolve(fname, flocation)
        self.data_mat = h5py.File(self.file_path, "r")

        try:
            # Internal name of the recording, usually the same as the file name
            self.data_name = next(k for k in self.data_mat if not k.startswith("#"))
            root = self.data_mat[self.data_name]
            self._X, self._Y = root["X"], root["Y"]

            # Kept for backwards compatibility
            self.y_data_objects = self._Y["Name"][:, 0]
            self.y_data_paths = self._Y["Path"][:, 0]

            self._read_rasters()
            self._read_signals()
            self._cache = {}
        except Exception:
            self.data_mat.close()
            raise

    # ------------------------------------------------------------------
    # File handling
    @staticmethod
    def _resolve(fname, flocation):
        name = str(fname)
        if not name.lower().endswith(".mat"):
            name += ".mat"
        candidates = [Path(flocation) / name] if flocation is not None else [
            Path(name), DATA_DIR / name]
        for p in candidates:
            if p.is_file():
                return p
        raise FileNotFoundError(
            f"{name} not found, tried: " + ", ".join(str(p) for p in candidates))

    def close(self):
        self.data_mat.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------
    # Parsing
    def _str(self, obj):
        """Decode a MATLAB char array (object reference or dataset)."""
        arr = self.data_mat[obj] if isinstance(obj, h5py.Reference) else obj
        if arr.attrs.get("MATLAB_empty", 0):
            return ""
        return "".join(map(chr, np.asarray(arr).ravel())).strip("\x00")

    def _read_rasters(self):
        """Time vectors. One raster: data inline. Several: object references."""
        X = self._X
        if X["Data"].dtype == object:
            self.raster_names = [self._str(r) for r in X["Raster"][:, 0]]
            self._raster_refs = list(X["Data"][:, 0])
        else:
            self.raster_names = [self._str(X["Raster"])]
            self._raster_refs = [X["Data"]]
        self._raster_cache = {}

    def _read_signals(self):
        Y = self._Y
        names = [self._str(r) for r in Y["Name"][:, 0]]
        paths = [self._str(r) for r in Y["Path"][:, 0]]
        x_index = [int(np.ravel(self.data_mat[r][()])[0]) - 1
                   for r in Y["XIndex"][:, 0]]
        self._refs = list(Y["Data"][:, 0])

        # Path parts of each signal, without "Model Root"
        parts = []
        for name, path in zip(names, paths):
            p = [s for s in path.split("/") if s][1:]
            m = _PORT.fullmatch(name)
            if m and p:
                # Port of a block: the block name is the meaningful name
                port = "" if m.group(2) in ("", "1") else "_out" + m.group(2)
                p[-1] = p[-1] + port + (m.group(3) or "")
            else:
                p.append(name)
            parts.append(p)

        # Shortest unique path suffix for each signal
        keys = []
        for i, p in enumerate(parts):
            for n in range(1, len(p) + 1):
                suffix = p[-n:]
                if sum(q[-n:] == suffix for q in parts) == 1:
                    break
            keys.append("/".join(suffix))
        # Identical full paths (same signal recorded twice)
        for i, k in enumerate(keys):
            if keys.count(k) > 1:
                keys[i] = f"{k}#{i}"

        self.signals = []
        for i in range(len(names)):
            self.signals.append({
                "index": i,
                "key": keys[i],
                "legacy": f"{names[i]}_{i}",
                "name": names[i],
                "path": paths[i],
                "full": "/".join(parts[i]),
                "raster": x_index[i],
                "samples": self.data_mat[self._refs[i]].shape[0],
            })
        self._by_key = {s["key"]: s for s in self.signals}
        self._by_key.update({s["legacy"]: s for s in self.signals})
        self._by_key.update({s["full"]: s for s in self.signals})

    # ------------------------------------------------------------------
    # Access
    def keys(self):
        return [s["key"] for s in self.signals]

    def _find(self, key):
        """Return signal info for a key, or a list of vector elements."""
        if isinstance(key, (int, np.integer)):
            return self.signals[key]
        if key in self._by_key:
            return self._by_key[key]
        # Unique end of the full path
        hits = [s for s in self.signals
                if s["full"] == key or s["full"].endswith("/" + key)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            # Vector signal without index, e.g. "i_s_abc" -> [0], [1], [2]
            elems = [s for s in self.signals
                     if _INDEX.sub("", s["key"]) == key]
            if elems:
                return sorted(elems,
                              key=lambda s: int(_INDEX.search(s["key"])[1]))
            close = difflib.get_close_matches(key, self.keys(), n=5)
            raise KeyError(f"No signal '{key}'. Close matches: {close}. "
                           "Use .list() to see all signals.")
        raise KeyError(f"'{key}' is ambiguous, use one of: "
                       f"{[s['key'] for s in hits]}")

    def signal_info(self, key):
        """Info dicts (key, name, path, raster, samples, ...) of a signal,
        one per element for a vector signal."""
        sig = self._find(key)
        return sig if isinstance(sig, list) else [sig]

    def _load(self, sig):
        i = sig["index"]
        if i not in self._cache:
            self._cache[i] = self.data_mat[self._refs[i]][:, 0]
        return self._cache[i]

    def __getitem__(self, key):
        sig = self._find(key)
        if isinstance(sig, list):
            return np.vstack([self._load(s) for s in sig])
        return self._load(sig)

    def __contains__(self, key):
        try:
            self._find(key)
            return True
        except KeyError:
            return False

    def complex(self, key):
        """Vector signal as complex: 2 elements -> [0] + 1j*[1],
        3 elements (abc) -> space vector 2/3*(a + b*e^(j2pi/3) + c*e^(j4pi/3))."""
        v = self[key]
        if v.ndim != 2 or v.shape[0] not in (2, 3):
            raise ValueError(f"'{key}' is not a 2- or 3-element vector signal")
        if v.shape[0] == 2:
            return v[0] + 1j*v[1]
        a = np.exp(2j*np.pi/3)
        return 2/3*(v[0] + a*v[1] + a**2*v[2])

    def time(self, key=None):
        """Time vector of a raster (by name or index) or of a signal."""
        if key is None:
            r = self._main_raster()
        elif isinstance(key, str) and key in self.raster_names:
            r = self.raster_names.index(key)
        elif isinstance(key, (int, np.integer)):
            r = key
        else:
            sig = self._find(key)
            r = (sig[0] if isinstance(sig, list) else sig)["raster"]
        if r not in self._raster_cache:
            ref = self._raster_refs[r]
            ds = self.data_mat[ref] if isinstance(ref, h5py.Reference) else ref
            self._raster_cache[r] = ds[:, 0]
        return self._raster_cache[r]

    def _main_raster(self):
        rasters = [s["raster"] for s in self.signals]
        return max(set(rasters), key=rasters.count)

    @property
    def t(self):
        """Time vector of the raster that most signals use."""
        return self.time()

    # ------------------------------------------------------------------
    # Printing and plotting
    def list(self, filter="", legacy=False) -> None:
        """Print signals whose key or path contains filter (case-insensitive)."""
        f = filter.lower()
        rows = [s for s in self.signals
                if f in s["full"].lower() or f in s["path"].lower()
                or f in s["legacy"].lower()]
        if not rows:
            print(f"No signals matching '{filter}'")
            return
        w = max(len(s["key"]) for s in rows)
        wl = max(len(s["legacy"]) for s in rows)
        main = self._main_raster()
        for s in rows:
            line = f"{s['index']:3d}  {s['key']:<{w}}  "
            if legacy:
                line += f"{s['legacy']:<{wl}}  "
            line += s["path"]
            if s["raster"] != main:
                line += (f"  [{self.raster_names[s['raster']]},"
                         f" {s['samples']} samples]")
            print(line)

    def info(self) -> None:
        print(f"File: {self.file_path}")
        print(f"Signals: {len(self.signals)}")
        for i, name in enumerate(self.raster_names):
            t = self.time(i)
            n = sum(s["raster"] == i for s in self.signals)
            dt = f", dt = {np.median(np.diff(t))*1e6:.1f} us" if t.size > 100 else ""
            print(f"Raster {name}: {n} signals, {t.size} samples, "
                  f"t = {t[0]:.3f} ... {t[-1]:.3f} s{dt}")

    def __repr__(self):
        return (f"LabDataMat('{self.file_path.name}', "
                f"{len(self.signals)} signals)")

    def plot(self, *groups, t_offset=0.0, xlim=None):
        """Quick-look plot. Each argument is one subplot: a key or a list
        of keys. Vector keys (e.g. "i_s_abc") plot all elements."""
        import matplotlib.pyplot as plt
        if not groups:
            raise ValueError("Give signal keys to plot")
        fig, axes = plt.subplots(len(groups), 1, sharex=True, squeeze=False)
        for ax, group in zip(axes[:, 0], groups):
            for key in [group] if isinstance(group, (str, int)) else group:
                y = self[key]
                t = self.time(key) - t_offset
                for k, row in enumerate(np.atleast_2d(y)):
                    label = f"{key}[{k}]" if y.ndim == 2 else str(key)
                    ax.plot(t, row, label=label,
                            drawstyle="steps-post" if t.size < 100 else None)
            ax.legend(loc="upper right")
            ax.grid(True)
        axes[-1, 0].set_xlabel("Time (s)")
        if xlim is not None:
            axes[-1, 0].set_xlim(xlim)
        fig.tight_layout()
        return fig, axes[:, 0]

    # ------------------------------------------------------------------
    # Old interface, kept so that existing scripts keep working
    def objKeys(self, objects) -> dict:
        return {f"{self._str(obj)}_{i}": i for i, obj in enumerate(objects)}

    def getXValues(self) -> dict:
        if len(self.raster_names) == 1:
            return {self.raster_names[0]: self.time(0)}
        return {f"{n}_{i}": self.time(i) for i, n in enumerate(self.raster_names)}

    def getYValues(self) -> dict:
        return {s["legacy"]: self._load(s) for s in self.signals}

    def dataRouteAndName(self, filter="") -> None:
        self.list(filter, legacy=True)


def measurement_files(meas, plot_all=False):
    """Resolve a measurement, optionally selecting all .mat files in its folder.

    Folder selection is non-recursive and case-insensitive. Returned paths are
    absolute, so saving next to the source does not depend on the working folder.
    """
    source = LabDataMat._resolve(meas, None).resolve()
    if not plot_all:
        return [source]
    return sorted(
        (p.resolve() for p in source.parent.iterdir()
         if p.is_file() and p.suffix.lower() == ".mat"),
        key=lambda p: (p.name.casefold(), p.name),
    )


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _time_vector(t):
    t = np.asarray(t, dtype=float)
    if (t.ndim != 1 or not t.size or not np.all(np.isfinite(t))
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Time must be a non-empty, finite, increasing vector")
    return t


def detect_transients(t, y, threshold=None):
    """Return (start, end) times of reference changes, including whole ramps.

    Adjacent finite samples differing by more than an absolute threshold count
    as changes. Consecutive changes form a ramp, starting at its preceding sample;
    an isolated step starts at the changed sample. NaN gaps are not bridged.
    The default threshold is 1e-6 of the finite signal range, with a floating-point
    noise floor. Supply a threshold in the signal's own units to tune detection.
    On a sparse OnChange raster, consecutive changes cannot be distinguished
    from a ramp without additional samples and are treated as a changing interval.
    """
    t = _time_vector(t)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size != t.size:
        raise ValueError("Reference must be a scalar signal matching its time vector")
    finite = np.isfinite(y)
    if threshold is not None:
        threshold = _nonnegative(threshold, "threshold")
    elif np.any(finite):
        values = y[finite]
        threshold = max(
            float(np.ptp(values))*1e-6,
            np.finfo(float).eps*max(1.0, float(np.max(np.abs(values))))*16,
        )
    else:
        return []
    changes = np.flatnonzero(
        finite[:-1] & finite[1:] & (np.abs(np.diff(y)) > threshold))
    if not changes.size:
        return []
    runs = np.split(changes, np.flatnonzero(np.diff(changes) > 1) + 1)
    return [(float(t[run[0] if run.size > 1 else run[0] + 1]),
             float(t[run[-1] + 1])) for run in runs]


def detect_spikes(t, y, threshold):
    """Return intervals where |y - median(finite y)| exceeds a raw-unit threshold.

    The median is calculated over the whole selected signal. NaN gaps split
    intervals; crossings use sample times, without interpolating or bridging.
    Long deviations count as events too: this does not measure peak prominence
    or settling time, and normalization/plot edits do not affect detection.
    """
    t = _time_vector(t)
    values = np.asarray(y)
    if (values.ndim != 1 or values.size != t.size or np.iscomplexobj(values)
            or not np.issubdtype(values.dtype, np.number)):
        raise ValueError('Spike detection needs a real scalar signal on its own time raster')
    threshold = _nonnegative(threshold, 'spike_threshold')
    if threshold == 0:
        raise ValueError('spike_threshold must be positive')
    finite = np.isfinite(values)
    if not np.any(finite):
        return []
    baseline = np.median(values[finite])
    active = np.flatnonzero(finite & (np.abs(values - baseline) > threshold))
    if not active.size:
        return []
    runs = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
    return [(float(t[run[0]]), float(t[run[-1]])) for run in runs]


def transient_window(data, pre_transient=0.2, duration=None, post_transient=0.2,
                     thresholds=None, t_offset=None, t_span=None, *,
                     detector='references', spike_signal=None, spike_threshold=None):
    """Return recording-time origin and relative plot limits for a measurement.

    By default, wm_ref and tauLref (or TL_ref when tauLref is absent) determine
    the first and last reference changes, each on its own time raster. Missing
    references are skipped. ``thresholds`` maps reference keys to absolute
    change thresholds; None values use automatic thresholds. A supplied mapping
    also selects the detection signals.

    detector='spikes' instead uses a selected real spike_signal, on its own
    raster, where deviation from its whole-recording median exceeds the positive
    spike_threshold in recorded units. Missing selected signals raise an error.
    No events fall back to the recording bounds, not to the other detector.

    Zero precedes the first change by ``pre_transient`` seconds. ``duration``
    is measured from zero; otherwise the window ends ``post_transient`` seconds
    after the last change. Without changes, use the whole recording (or the
    requested duration from its beginning) and print a notice. Explicit
    ``t_offset`` and/or ``t_span`` override the automatic origin/limits.
    Limits outside the recording are retained, with a notice, so the requested
    origin and tick alignment do not silently change. No data is extrapolated.
    """
    t = _time_vector(data.t)
    if t.size < 2:
        raise ValueError("A plot window requires at least two time samples")
    pre_transient = _nonnegative(pre_transient, "pre_transient")
    post_transient = _nonnegative(post_transient, "post_transient")
    if duration is not None:
        duration = _nonnegative(duration, "duration")
        if duration == 0:
            raise ValueError("duration must be positive")
    if t_offset is not None:
        t_offset = float(t_offset)
        if not np.isfinite(t_offset):
            raise ValueError("t_offset must be finite")
    if t_span is not None:
        t_span = np.asarray(t_span, dtype=float)
        if (t_span.shape != (2,) or not np.all(np.isfinite(t_span))
                or t_span[1] <= t_span[0]):
            raise ValueError("t_span must contain two finite, increasing limits")
        if duration is not None:
            raise ValueError("Specify duration or t_span, not both")
    if detector not in ('references', 'spikes'):
        raise ValueError('detector must be references or spikes')
    if detector == 'spikes':
        if not isinstance(spike_signal, str) or not spike_signal:
            raise ValueError('Select a spike_signal')
        if spike_signal not in data:
            raise KeyError(f"Spike signal '{spike_signal}' is not in the recording")
        if spike_threshold is None or _nonnegative(spike_threshold, 'spike_threshold') == 0:
            raise ValueError('Specify a positive spike_threshold in recorded units')
    references = {"wm_ref": None, "tauLref": None} if thresholds is None else dict(thresholds)
    for key, threshold in references.items():
        if threshold is not None:
            _nonnegative(threshold, f"threshold for {key}")
    events = []
    if t_offset is None or (t_span is None and duration is None):
        if detector == 'spikes':
            candidates = detect_spikes(data.time(spike_signal), data[spike_signal], spike_threshold)
        else:
            candidates = []
            for key, threshold in references.items():
                if key == "tauLref" and key not in data and "TL_ref" in data:
                    key = "TL_ref"
                if key not in data:
                    continue
                candidates.extend(detect_transients(data.time(key), data[key], threshold))
        for start, end in candidates:
            if start <= t[-1] and end >= t[0]:
                events.append((max(start, float(t[0])), min(end, float(t[-1]))))
        if not events:
            message = 'No measurement spikes found' if detector == 'spikes' else 'No reference transients found'
            print(f'{message}; using the recording bounds.')
    if t_offset is None:
        t_offset = min(start for start, _ in events) - pre_transient if events else float(t[0])
    if t_span is None:
        end = max(end for _, end in events) + post_transient if events else float(t[-1])
        t_span = (0.0, duration if duration is not None else end - t_offset)
        if t_span[1] <= 0:
            raise ValueError("The plot window ends before its origin; specify duration or t_span")
    start, end = (t_offset + float(limit) for limit in t_span)
    if end <= t[0] or start >= t[-1]:
        raise ValueError("The plot window does not overlap the recording")
    if start < t[0] or end > t[-1]:
        print("Plot window extends beyond the recording; unavailable margins remain empty.")
    return float(t_offset), tuple(float(limit) for limit in t_span)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python lab_plotting.py <file[.mat]> [filter]")
        print(f"Files in {DATA_DIR}:")
        for p in sorted(DATA_DIR.glob("*.mat")):
            print("  " + p.stem)
        sys.exit()
    with LabDataMat(sys.argv[1]) as data:
        data.info()
        print()
        data.list(sys.argv[2] if len(sys.argv) > 2 else "")
