"""Reusable measurement plots extracted from Lab_plotter.ipynb.

load_measurement returns an independent, in-memory SimpleNamespace after closing
its recording. Built-in plots retain the notebook units, labels and limits, and
use the caller's Matplotlib style. No backend, display or export is configured.
FIGURE_TYPES maps GUI names to (function, keyword options, export suffix).
For custom panels, open LabDataMat in a with block, select from data.keys(), and
pass lists of readable keys to plot_signals; each component keeps its own raster.

Built-in plots read quantities (roles, see lab_plotter_config.SIGNAL_ROLES)
through a signal map: role -> recording key or ordered alternative keys. The
first alternative present with the expected shape is used; missing quantities
are NaN. Reference transients follow the mapped w_m_ref and tau_Lref keys.
Custom panels, added traces and spike detection accept ``@role`` keys, which
select the signal map's key for that role and default to its recorded unit.
Every trace has a derived processing (PROCESSING, e.g. magnitude, real part,
angle wrap) applied before its gain/offset; transforms may replace it.
"""
import ast
import re
import string
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from lab_plotting import LabDataMat, transient_window
from lab_plotter_config import (
    PROCESSING_IDS,
    ROLE_UNITS,
    SCALAR_PROCESSING,
    SIGNAL_ROLE_IDS,
    SIGNAL_ROLES,
    VECTOR_PROCESSING,
    default_signal_map,
    signal_keys,
    signal_role,
)
from plot_style import downsample

@dataclass
class BaseValues:
    # pylint: disable=too-many-instance-attributes
    U_nom: float
    I_nom: float
    f_nom: float
    P_nom: float
    tau_nom: float
    n_p: int

    def __post_init__(self):
        self.u = np.sqrt(2/3)*self.U_nom
        self.i = np.sqrt(2)*self.I_nom
        self.w = 2*np.pi*self.f_nom
        self.w_M = self.w/self.n_p
        self.psi = self.u/self.w
        self.p = 1.5*self.u*self.i
        self.Z = self.u/self.i
        self.L = self.Z/self.w
        self.C = 1/(self.Z*self.w)
        self.tau = self.n_p*self.p/self.w


DEFAULT_BASE = BaseValues(U_nom=460, I_nom=8.8, f_nom=60,
                          tau_nom=29.2, P_nom=5.6e3, n_p=2)
DEFAULT_SIGNAL_MAP = default_signal_map()['signals']
VECTOR_ROLES = frozenset(role for role, _, _, vector, _ in SIGNAL_ROLES if vector)
REFERENCE_ROLES = ('w_m_ref', 'tau_Lref')


def _wrap(y):
    return np.angle(np.exp(1j*y))


# Derived processing: id -> function(values, time, base). Vector operations take
# a complex space vector; the others act on real values.
PROCESSING = {
    'none': lambda y, t, base: y,
    'abs': lambda y, t, base: np.abs(y),
    'wrap': lambda y, t, base: _wrap(y),
    'unwrap': lambda y, t, base: np.unwrap(y),
    'degrees': lambda y, t, base: np.degrees(y),
    'wrap_degrees': lambda y, t, base: np.degrees(_wrap(y)),
    'wrap_mech_degrees': lambda y, t, base: np.degrees(_wrap(y))/base.n_p,
    'derivative': lambda y, t, base: np.gradient(y, t),
    'magnitude': lambda z, t, base: np.abs(z),
    'real': lambda z, t, base: z.real,
    'imag': lambda z, t, base: z.imag,
    'angle': lambda z, t, base: np.angle(z),
}
assert tuple(PROCESSING) == PROCESSING_IDS
PROCESSING_LABELS = {
    'none': 'none', 'abs': 'absolute value |y|', 'wrap': 'wrap to (−π, π]',
    'unwrap': 'unwrap (continuous angle)', 'degrees': 'radians → degrees',
    'wrap_degrees': 'wrap to (−180°, 180°]', 'wrap_mech_degrees': 'wrap, mechanical degrees (÷ n_p)',
    'derivative': 'time derivative d/dt', 'magnitude': 'magnitude |x + jy|', 'real': 'real part',
    'imag': 'imaginary part', 'angle': 'angle ∠(x + jy) (rad)'}
# Operations offered for a preset trace keep the quantity of its axis: an angle
# of a normalized flux would be meaningless after p.u./SI scaling.
PRESET_PROCESSING = {'real': ('none', 'abs'), 'complex': ('magnitude', 'real', 'imag'),
                     'angle': ('none', 'wrap', 'unwrap', 'degrees', 'wrap_degrees', 'wrap_mech_degrees')}
_RADIAN_OUTPUT = frozenset(('wrap', 'unwrap', 'angle'))
_DEGREE_OUTPUT = frozenset(('degrees', 'wrap_degrees', 'wrap_mech_degrees'))
_VECTOR_LABELS = {'magnitude': '|{key}|', 'real': 'Re({key})', 'imag': 'Im({key})', 'angle': '∠{key}'}
_INPUT_NOTES = {'i_c': 'dq frame: i_cs·exp(−jθm)', 'theta_err': 'measured − estimated angle'}


def processing_unit(processing, unit):
    """Displayed unit after processing; None keeps the signal's unit conversion."""
    if processing in _RADIAN_OUTPUT:
        return 'rad'
    if processing in _DEGREE_OUTPUT:
        return 'deg'
    return None


def process_components(components, processing, base, key):
    """Apply processing to the (true key, time, values) components of one signal.

    Vector operations combine 2 (x + jy) or 3 (abc) components on one raster
    into a single trace; scalar operations keep one trace per component.
    """
    if processing in VECTOR_PROCESSING:
        if len(components) not in (2, 3) or any(not np.array_equal(time, components[0][1])
                                                for _, time, _ in components):
            raise ValueError(f"{key}: '{processing}' needs a 2- or 3-element vector on one time raster")
        values = [np.asarray(values, dtype=float) for _, _, values in components]
        if len(values) == 2:
            vector = values[0] + 1j*values[1]
        else:
            a = np.exp(2j*np.pi/3)
            vector = 2/3*(values[0] + a*values[1] + a**2*values[2])
        time = components[0][1]
        name = re.sub(r'\[\d+\]$', '', components[0][0])
        return [(_VECTOR_LABELS[processing].format(key=name), time, PROCESSING[processing](vector, time, base))]
    if processing not in SCALAR_PROCESSING:
        raise ValueError(f'{key}: unknown processing {processing!r}')
    result = []
    for true_key, time, values in components:
        if processing == 'derivative' and time.size < 2:
            raise ValueError(f'{key}: a derivative needs at least two samples')
        label = f'd{true_key}/dt' if processing == 'derivative' else true_key
        result.append((label, time, PROCESSING[processing](np.asarray(values, dtype=float), time, base)))
    return result


def complete_signal_map(signal_map=None):
    """Return role -> list of alternative keys; omitted roles use the defaults."""
    if signal_map is None:
        signal_map = {}
    if not isinstance(signal_map, dict):
        raise ValueError('signal_map must map roles to recording keys')
    unknown = set(signal_map) - set(SIGNAL_ROLE_IDS)
    if unknown:
        raise ValueError(f'Unknown signal roles: {sorted(unknown)}')
    return {role: signal_keys(signal_map.get(role, DEFAULT_SIGNAL_MAP[role]), role)
            for role in SIGNAL_ROLE_IDS}


def signal_matches(data, key, vector):
    """Whether a key exists with the shape a scalar or a 2/3-element vector role needs."""
    if not key or key not in data:
        return False
    try:
        count = len(data.signal_info(key))
    except KeyError:
        return False
    return count in (2, 3) if vector else count == 1


def resolve_signal_map(data, signal_map=None):
    """Return role -> first matching key in data (LabDataMat or SignalCatalog), or None."""
    return {role: next((key for key in keys if signal_matches(data, key, role in VECTOR_ROLES)), None)
            for role, keys in complete_signal_map(signal_map).items()}


def reference_thresholds(data, signal_map=None):
    """transient_window thresholds for the mapped speed and load-torque references."""
    resolved = resolve_signal_map(data, signal_map)
    return dict.fromkeys(key for key in (resolved[role] for role in REFERENCE_ROLES) if key)


def role_key(data, key, signal_map=None):
    """Return the recording key an ``@role`` key selects; other keys pass through.

    A key the data already knows (such as a detached snapshot entry) is kept.
    """
    role = signal_role(key)
    if role is None or key in data:
        return key
    mapping = complete_signal_map(signal_map)
    if role not in mapping:
        raise ValueError(f'Unknown quantity role: {key}')
    resolved = next((candidate for candidate in mapping[role]
                     if signal_matches(data, candidate, role in VECTOR_ROLES)), None)
    if resolved is None:
        raise ValueError(f"{key}: no signal {' / '.join(mapping[role]) or '(not linked)'} "
                         "in the recording; check the signal map")
    return resolved


def role_unit(key, default='raw'):
    """Recorded unit of an ``@role`` key, or default for a recording key."""
    return ROLE_UNITS.get(signal_role(key), default)


def detection_options(data, signal_map=None, **window_options):
    """transient_window options with mapped references and a resolved @role spike signal."""
    options = dict(window_options)
    if options.get('thresholds') is None:
        options['thresholds'] = reference_thresholds(data, signal_map)
    if options.get('detector') == 'spikes' and signal_role(options.get('spike_signal')):
        options['spike_signal'] = role_key(data, options['spike_signal'], signal_map)
    return options


def load_measurement(path, motor_base=None, *, x_step=None, extra_signals=None,
                     signal_map=None, **window_options):
    """Load preset quantities via signal_map and detach requested extra components.

    context.signal_map holds the key used for each role (None when missing).
    Extra components keep their own rasters.
    """
    motor_base = DEFAULT_BASE if motor_base is None else motor_base
    mapping = complete_signal_map(signal_map)
    _validate_window(0, None, x_step)
    with LabDataMat(path) as data:
        resolved = resolve_signal_map(data, mapping)
        origin, limits = transient_window(data, **detection_options(data, mapping, **window_options))
        context = SimpleNamespace(
            file_path=data.file_path.resolve(), name=data.file_path.stem,
            base=replace(motor_base), t=data.t.copy(), t_offset=origin,
            t_span=limits, x_step=x_step, loaded=[], signal_rows=[], extra_signals={},
            signal_map=resolved)

        if extra_signals is not None:
            if isinstance(extra_signals, (str, bytes)):
                raise ValueError('extra_signals must be an iterable of signal keys')
            for key in extra_signals:
                if not isinstance(key, str) or not key:
                    raise ValueError('Extra signal keys must be nonempty strings')
                try:
                    signals = data.signal_info(role_key(data, key, mapping))
                    if not signals:
                        raise ValueError('no components found')
                    components = []
                    for signal in signals:
                        true_key = signal['key']
                        time = np.asarray(data.time(true_key), dtype=float)
                        values = np.asarray(data[true_key])
                        if (time.ndim != 1 or not time.size or not np.all(np.isfinite(time))
                                or np.any(np.diff(time) <= 0)):
                            raise ValueError(f"component '{true_key}' needs a finite increasing time vector")
                        if (values.ndim != 1 or values.size != time.size
                                or not np.issubdtype(values.dtype, np.number)
                                or np.iscomplexobj(values)):
                            raise ValueError(f"component '{true_key}' needs real values matching its time")
                        components.append((true_key, time.copy(), values.copy()))
                    context.extra_signals[key] = components
                except (KeyError, TypeError, ValueError, IndexError) as error:
                    raise ValueError(f"{context.name}: cannot load extra signal '{key}': {error}") from error

        def get(role):
            as_complex = role in VECTOR_ROLES
            key = resolved[role]
            missing = np.full(context.t.size, np.nan, dtype=complex if as_complex else float)
            if key is None:
                candidates = mapping[role]
                context.loaded.append(candidates[0] if candidates else role)
                if candidates:
                    print(f"{context.name}: Warning: no unique {'vector ' if as_complex else ''}"
                          f"signal {' / '.join(candidates)} for {role}; check the signal map")
                return missing
            context.loaded.append(key)
            try:
                y = data.complex(key) if as_complex else data[key]
            except (KeyError, ValueError) as error:
                print(f"{context.name}: Warning: {error.args[0]}")
                return missing
            t_y = data.time(key)
            if not np.array_equal(t_y, context.t):
                idx = np.clip(np.searchsorted(t_y, context.t, side="right") - 1,
                              0, t_y.size - 1)
                y = y[..., idx]
            return y.copy()

        for role in SIGNAL_ROLE_IDS:
            setattr(context, role, get(role))
        context.theta_mest = context.theta_m.copy()
        context.theta_err = context.theta_meas - context.theta_m
        context.w_m = context.w_mest
        context.i_c = context.i_cs*np.exp(-1j*context.theta_m)
        context.u_s = np.abs(context.u_c)
        context.theta_err_M = np.degrees(np.angle(np.exp(
            1j*(context.theta_meas - context.theta_m))))/context.base.n_p
        context.eps_inj_M = np.degrees(context.eps_inj)

        for key in context.loaded:
            if key not in data:
                context.signal_rows.append((key, "-", "NOT IN RECORDING", "", "", ""))
                continue
            for signal in data.signal_info(key):
                y = data[signal["key"]]
                route = signal["path"].removeprefix("Model Root/")
                if not np.array_equal(data.time(signal["key"]), context.t):
                    route += (f"  [{data.raster_names[signal['raster']]}, "
                              f"{signal['samples']} samples]")
                context.signal_rows.append((
                    signal["key"], signal["name"], route,
                    f"{np.min(y):.4g}", f"{np.max(y):.4g}", f"{np.mean(y):.4g}"))
    return context


def plot_window(context, t_offset=None, t_span=None, x_step=None):
    origin = context.t_offset if t_offset is None else t_offset
    limits = context.t_span if t_span is None else t_span
    step = context.x_step if x_step is None else x_step
    origin, limits, step = _validate_window(origin, limits, step)
    return context.t - origin, limits, step


def plot_overview(context, t_offset=None, t_span=None, bottom="voltage", x_step=None):
    # Speed, flux, torque, currents and voltages (bottom="voltage")
    # or injection voltages (bottom="injection").
    t, t_span, x_step = plot_window(context, t_offset, t_span, x_step)
    base = context.base
    w_m_ref = context.w_m_ref
    w_mest = context.w_mest
    psi_sref = context.psi_sref
    psi_sest = context.psi_sest
    tau_Mref_lim = context.tau_Mref_lim
    tau_Mest = context.tau_Mest
    tau_Lref = context.tau_Lref
    i_c = context.i_c
    u_dc = context.u_dc
    u_s = context.u_s
    u_inj = context.u_inj
    if bottom not in ("voltage", "injection"):
        raise ValueError("bottom must be voltage or injection")

    width = plt.rcParams["figure.figsize"][0]
    height = plt.rcParams["figure.figsize"][1]*3*4/5
    fig, axes = plt.subplots(5, 1, figsize=(width, height), sharex=True)
    for ax in axes:
        ax.xaxis.set_major_locator(MultipleLocator(1 if x_step is None else x_step))
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, color="#B0B0B0")

    # Speed
    axes[0].step(t, w_m_ref/base.w, "--", where="post",
                 label=r"$\omega_\mathrm{m}^\mathrm{{ref}}$", linewidth=1.3)
    axes[0].plot(t, w_mest/base.w, label=r"$\omega_\mathrm{m}$", alpha=0.8)
    axes[0].set_ylabel(r"$\omega_\mathrm{m}$ (p.u.)")

    # Flux linkage
    axes[1].plot(t, np.abs(psi_sref)/base.psi, "--",
                 label=r"$\psi_\mathrm{s}^\mathrm{ref}$", linewidth=1.3)
    axes[1].step(t, np.abs(psi_sest)/base.psi, where="post",
                 label=r"$\hat\psi_\mathrm{s}$", alpha=0.8)
    axes[1].set_ylabel(r"$\psi_\mathrm{s}$ (p.u.)")
    axes[1].set_ylim([-0.05, 1.2])

    # Torque
    axes[2].plot(t, tau_Mref_lim/base.tau, "--",
                 label=r"$\overline{\tau}_\mathrm{m}^\mathrm{ref}$", linewidth=1.3)
    axes[2].plot(t, tau_Mest/base.tau, label=r"$\hat \tau_\mathrm{m}$", alpha=0.8)
    axes[2].plot(t, tau_Lref, "--", label=r"$\tau_\mathrm{L}$", linewidth=1.3)
    axes[2].set_ylabel(r"$\tau_\mathrm{m}$ (p.u.)")
    axes[2].set_ylim([-1.1, 2])

    # Currents
    axes[3].step(t, i_c.real/base.i, where="post", label=r"$i_\mathrm{d}$")
    axes[3].step(t, i_c.imag/base.i, where="post", label=r"$i_\mathrm{q}$")
    # axes[3].step(t, i_ref.real/base.i, where="post", label=r"$i_\mathrm{d}^\mathrm{ref}$")
    # axes[3].step(t, i_ref.imag/base.i, where="post", label=r"$i_\mathrm{q}^\mathrm{ref}$")
    axes[3].set_ylabel(r"$i_\mathrm{d}, i_\mathrm{q}$ (p.u.)")

    # Voltages
    if bottom == "voltage":
        axes[4].plot(t, u_dc/np.sqrt(3)/base.u, "--",
                     label=r"$u_\mathrm{dc}/\sqrt{3}$", ds="steps-post")
        axes[4].plot(t, u_s/base.u, label=r"$u_\mathrm{s}$", ds="steps-post")
        axes[4].set_ylabel("Voltage (p.u.)")
    else:
        axes[4].plot(t, u_inj.real, label=r"$u_\mathrm{d}^{\mathrm{inj}}$",
                     ds="steps-post")
        axes[4].plot(t, u_inj.imag, label=r"$u_\mathrm{q}^{\mathrm{inj}}$",
                     ds="steps-post")
        axes[4].set_ylabel("Injection Voltages (V)")

    for ax in axes:
        ax.legend(loc="best")
    axes[-1].set_xlim(t_span)
    axes[-1].set_xlabel("Time (s)")
    fig.align_ylabels()
    return fig, axes


def plot_hfsi(context, t_offset=None, t_span=None, x_step=None):
    # Left: speed, flux, torque. Right: currents, angle errors, injection and weight.
    # x_step: spacing of the time ticks (s), None = automatic
    t, t_span, x_step = plot_window(context, t_offset, t_span, x_step)
    base = context.base
    w_m_ref = context.w_m_ref
    w_mest = context.w_mest
    w_m_meas = context.w_m_meas
    psi_sref = context.psi_sref
    psi_sest = context.psi_sest
    tau_Mref_lim = context.tau_Mref_lim
    tau_Mest = context.tau_Mest
    tau_Lref = context.tau_Lref
    i_c = context.i_c
    eps_inj_M = context.eps_inj_M
    theta_err_M = context.theta_err_M
    theta_mest = context.theta_mest
    theta_m = context.theta_m
    theta_meas = context.theta_meas
    width = 2*plt.rcParams["figure.figsize"][0]
    height = plt.rcParams["figure.figsize"][1]*3*4/5*3/5
    fig, axes = plt.subplots(3, 2, figsize=(width, height), sharex=True,
                             layout="constrained")
    for ax in axes.flat:
        if x_step is not None:
            ax.xaxis.set_major_locator(MultipleLocator(x_step))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # minor lines between majors
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.grid(True, which="major", linestyle="-", linewidth=0.8, color="#B0B0B0")
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, color="#C8C8C8")
    (ax_w, ax_i), (ax_psi, ax_err), (ax_tau, ax_u) = axes

    # Speed
    ax_w.step(t, w_m_ref/base.w, "--", where="post",
              label=r"$\omega_\mathrm{m}^\mathrm{{ref}}$", linewidth=1.3)
    ax_w.plot(t, w_mest/base.w, label=r"$\omega_\mathrm{m}$", alpha=0.8)
    ax_w.plot(t, (w_m_meas/2)/base.w, label=r"$\omega_\mathrm{m}^\mathrm{meas}$", alpha=0.8)
    ax_w.set_ylabel(r"$\omega_\mathrm{m}$ (p.u.)")
    ax_w.set_ylim([-0.2, 0.2])
    # ax_w.set_ylim([-0.15, 1.15])


    # Flux linkage
    ax_psi.plot(t, np.abs(psi_sref)/base.psi, "--",
                label=r"$\psi_\mathrm{s}^\mathrm{ref}$", linewidth=1.3)
    ax_psi.step(t, np.abs(psi_sest)/base.psi, where="post",
                label=r"$\hat\psi_\mathrm{s}$", alpha=0.8)
    ax_psi.set_ylabel(r"$\psi_\mathrm{s}$ (p.u.)")
    ax_psi.set_ylim([-0.05, 1.2])

    # Torque
    ax_tau.plot(t, tau_Mref_lim/base.tau_nom, "--",
                label=r"$\overline{\tau}_\mathrm{m}^\mathrm{ref}$", linewidth=1.3)
    ax_tau.plot(t, tau_Mest/base.tau_nom, label=r"$\hat \tau_\mathrm{m}$", alpha=0.8)
    ax_tau.plot(t, tau_Lref/.88, "--", label=r"$\tau_\mathrm{L}$", linewidth=1.3)
    ax_tau.set_ylabel(r"$\tau_\mathrm{m}$ (per rated)")
    ax_tau.set_ylim([-3, 3])

    # Currents
    ax_i.step(t, i_c.real/base.i, where="post", label=r"$i_\mathrm{d}$")
    ax_i.step(t, i_c.imag/base.i, where="post", label=r"$i_\mathrm{q}$")
    ax_i.set_ylabel(r"$i_\mathrm{d}, i_\mathrm{q}$ (p.u.)")
    # ax_i.set_ylim([-1.5, 1])
    ax_i.set_ylim([-2, 2])

    # Mechanical angle error, injection error signal behind it
    # (not grey: it blends into the grid lines)
    ax_err.plot(t, eps_inj_M, color="C1", linewidth=1, alpha=0.7, zorder=1.5,
                label=r"$\varepsilon_\mathrm{i}$")
    ax_err.plot(t, theta_err_M, color="C0", zorder=2.5,
                label=r"$\tilde\vartheta$")
    ax_err.set_ylabel("Error (mech. deg)")
    ax_err.set_ylim([-10, 10])  # eps_inj spikes when the injection starts or stops

    # # Injection voltage, d-axis
    # ax_u.plot(t, u_inj.real, color="C0", label=r"$\mathbf{u}_\mathrm{i}$",
    #           ds="steps-post")
    # ax_u.set_ylabel(r"$\mathbf{u}_\mathrm{i}$ (V)")
    # ax_u.tick_params(axis="y")
    # angles
    wrap = lambda x: np.angle(np.exp(1j*x))  # wrap to (-pi, pi]
    ax_u.plot(t, wrap(theta_mest), color="C0", label=r"$\hat\vartheta_\mathrm{m}$")
    ax_u.plot(t, wrap(theta_meas), color="C1", label=r"$\vartheta_\mathrm{m}$")
    ax_u.set_ylabel(r"$\vartheta_\mathrm{m}$ (rad)")
    ax_u.set_ylim([-np.pi, np.pi])
    ax_u.set_yticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax_u.tick_params(axis="y")

    for ax in (ax_w, ax_psi, ax_tau, ax_i, ax_err):
        ax.legend(loc="upper center", bbox_to_anchor=(0.7, 1.0))
    for ax in (ax_tau, ax_w, ax_psi):
        ax.legend(loc="lower center", bbox_to_anchor=(0.34, 0))  # a bit left of center
    ax_err.legend(loc="upper right")
    ax_u.legend(loc="lower right")
    lines = ax_u.get_lines() #+ ax_weight.get_lines()
    # ax_weight.legend(lines, [line.get_label() for line in lines], loc="lower right")
    for ax in axes[-1]:
        ax.set_xlim(t_span)
        ax.set_xlabel("Time (s)")
    fig.align_ylabels(axes)
    return fig, axes


def plot_presentation(context, t_offset=None, t_span=None, x_step=None,
                      include_plots=("speeds", "fluxes", "torques", "currents"),
                      font_size=16):
    include_plots = tuple(include_plots)
    allowed = {"speeds", "fluxes", "torques", "currents", "voltages"}
    if not include_plots or set(include_plots) - allowed:
        raise ValueError("include_plots must be a nonempty selection of presentation panels")
    t, t_span, x_step = plot_window(context, t_offset, t_span, x_step)
    base = context.base
    w_m_ref = context.w_m_ref
    psi_sref = context.psi_sref
    psi_sest = context.psi_sest
    tau_Mref_lim = context.tau_Mref_lim
    tau_Mest = context.tau_Mest
    tau_Lref = context.tau_Lref
    i_c = context.i_c
    u_dc = context.u_dc
    u_s = context.u_s
    w_m = context.w_m
    M = r"\mathrm{m}"
    with plt.rc_context({key: font_size for key in (
            "font.size", "axes.labelsize", "legend.fontsize",
            "xtick.labelsize", "ytick.labelsize")}):
        n_plots = len(include_plots)
        fig, axes = plt.subplots(n_plots, 1, figsize=(8, 2*n_plots), sharex=True,
                                 squeeze=False, layout="tight")
        axes = axes[:, 0]

        for ax in axes:
            if x_step is not None:
                ax.xaxis.set_major_locator(MultipleLocator(x_step))
            ax.minorticks_on()
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(True, which='major', linestyle='-', linewidth=0.8)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.7)

        for ax, plot_type in zip(axes, include_plots):
            ax.set_xlim(t_span)

            if plot_type == 'speeds':
                ax.step(t, w_m_ref/base.w, "--", where="post", label=r"$\omega_{m}^\mathrm{{ref}}$")
                ax.plot(t, w_m/base.w, label=r"$\omega_{m}$")
                ax.legend(loc='lower right')
                ax.set_ylabel(r"$\omega_{m}$ (p.u.)")

            elif plot_type == 'fluxes':
                ax.plot(t, np.abs(psi_sref)/base.psi, '--', label=r"$\psi_\mathrm{s}^\mathrm{ref}$")
                ax.step(t, np.abs(psi_sest)/base.psi, where="post", label=r"$\hat\psi_\mathrm{s}$")
                ax.legend(loc='right')
                ax.set_ylim([-0.1, 1.25])
                ax.set_ylabel(r"$\psi_\mathrm{s}$ (p.u.)")

            elif plot_type == 'torques':
                ax.plot(t, tau_Mest/base.tau, label=rf"$\hat \tau_{M}$")
                # ax.plot(t, tau_Mref/base.tau, "--", label=rf"$\tau_{M}^\mathrm{{ref}}$")
                ax.step(t, tau_Mref_lim/base.tau, "--", label=rf"$\tau_{M}^\mathrm{{ref,lim}}$")
                ax.step(t, tau_Lref, "--", label=r"$\tau_\mathrm{L}$")
                ax.legend(loc='right')
                ax.set_ylim([-0.1, 1.5])
                ax.set_ylabel(rf"$\tau_{M}$ (p.u.)")

            elif plot_type == 'currents':
                ax.step(t, i_c.real/base.i, where="post", label=r"$i_\mathrm{d}$")
                ax.step(t, i_c.imag/base.i, where="post", label=r"$i_\mathrm{q}$")
                # ax.step(t, i_ref.real/base.i, where="post", label=r"$i_\mathrm{sd,ref}$")
                # ax.step(t, i_ref.imag/base.i, where="post", label=r"$i_\mathrm{sq,ref}$")
                ax.legend(loc='right')
                ax.set_ylabel(r"$\mathbf{i}_\mathrm{s}$ (p.u.)")

            elif plot_type == 'voltages':
                ax.plot(t, u_s/base.u, label=r"$u_\mathrm{c}$", ds="steps-post")
                ax.plot(t, u_dc/np.sqrt(3)/base.u, "--", label=r"$u_\mathrm{dc}/\sqrt{3}$",
                        ds="steps-post")
                ax.legend(loc='right')
                ax.set_ylim([0.0, 1.2])
                ax.set_ylabel("Voltage (p.u.)")

        fig.align_ylabels()
        axes[-1].set_xlabel("Time (s)")
        return fig, axes


def _validate_window(t_offset, t_span, x_step):
    origin = float(t_offset)
    if not np.isfinite(origin):
        raise ValueError("t_offset must be finite")
    limits = None
    if t_span is not None:
        limits = np.asarray(t_span, dtype=float)
        if (limits.shape != (2,) or not np.all(np.isfinite(limits))
                or limits[1] <= limits[0]):
            raise ValueError("t_span must contain two finite, increasing limits")
        limits = tuple(float(value) for value in limits)
    step = None if x_step is None else float(x_step)
    if step is not None and (not np.isfinite(step) or step <= 0):
        raise ValueError("x_step must be finite and positive")
    return origin, limits, step


FIGURE_TYPES = {
    "overview": (plot_overview, {}, "_1"),
    "injection-overview": (plot_overview, {"bottom": "injection"}, "_2"),
    "hfsi": (plot_hfsi, {}, "_hfsi_2"),
    "presentation": (plot_presentation, {}, "_online_meas"),
}


def scalar_expression(expression, base):
    """Evaluate finite scalar arithmetic without executing Python code.

    Allowed names: n_p, sqrt3, u_base, i_base, w_base, psi_base, tau_base,
    tau_nom. Only +, -, *, / and parentheses are supported.
    """
    text = str(expression).strip()
    if len(text) > 120:
        raise ValueError('Scalar expression is too long')
    try:
        tree = ast.parse(text, mode='eval')
    except SyntaxError as error:
        raise ValueError('Invalid scalar expression') from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError('Scalar expression is too complex')
    names = {'n_p': base.n_p, 'sqrt3': np.sqrt(3), 'u_base': base.u,
             'i_base': base.i, 'w_base': base.w, 'psi_base': base.psi,
             'tau_base': base.tau, 'tau_nom': base.tau_nom}

    def calculate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return float(names[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = calculate(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = calculate(node.left), calculate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError('Use numbers, base names, +, -, *, / and parentheses only')

    try:
        result = calculate(tree.body)
    except (ZeroDivisionError, OverflowError) as error:
        raise ValueError('Scalar expression must produce a finite number') from error
    if not np.isfinite(result):
        raise ValueError('Scalar expression must produce a finite number')
    return result


def subplot_traces(kind, groups=(), signal_map=None, unlinked='({role} not linked)'):
    """Describe every plotted trace, its derived processing and pre-normalization gain.

    'processing' is the default derived operation, 'operations' those a
    transform may choose instead, and 'input' the context attribute it acts on
    (presets: (processing(input) * scale + offset) / divisor). 'stage' is an
    English description. Offsets use the processed signal's input units, before
    p.u. conversion. Preset and @role sources show the signal_map keys
    (unlinked formats a role without keys); 'roles' lists the quantities used.
    """
    mapping = complete_signal_map(signal_map)
    names = {role: ' / '.join(keys) if keys else unlinked.format(role=role)
             for role, keys in mapping.items()}
    # (source template, input attribute, input kind, default processing, gain, divisor)
    speed = [('{w_m_ref}', 'w_m_ref', 'real', 'none', '1', 'w'),
             ('{w_mest}', 'w_mest', 'real', 'none', '1', 'w')]
    flux = [('{psi_sref}', 'psi_sref', 'real', 'abs', '1', 'psi'),
            ('{psi_sest}', 'psi_sest', 'complex', 'magnitude', '1', 'psi')]
    torque = [('{tau_Mref_lim}', 'tau_Mref_lim', 'real', 'none', '1', 'tau'),
              ('{tau_Mest}', 'tau_Mest', 'real', 'none', '1', 'tau'),
              ('{tau_Lref}', 'tau_Lref', 'real', 'none', '1', '')]
    currents = [('{i_cs} + {theta_m}', 'i_c', 'complex', 'real', '1', 'i'),
                ('{i_cs} + {theta_m}', 'i_c', 'complex', 'imag', '1', 'i')]
    voltage = [('{u_dc}', 'u_dc', 'real', 'none', '1/sqrt3', 'u'),
               ('{u_c}', 'u_c', 'complex', 'magnitude', '1', 'u')]
    if kind == 'custom':
        def source(key):
            role = signal_role(key)
            return f'{key} → {names[role]}' if role in names else key
        return [{'panel': panel, 'title': f'Subplot {panel + 1}',
                 'id': f'custom:{panel}:{key}', 'source': source(key), 'stage': 'none',
                 'scale': '1', 'offset': '0', 'divisor': '', 'input': key, 'processing': 'none',
                 'operations': PROCESSING_IDS, 'note': '',
                 'roles': (signal_role(key),) if signal_role(key) in names else ()}
                for panel, group in enumerate(groups) for key in group]
    if kind == 'hfsi':
        torque = [('{tau_Mref_lim}', 'tau_Mref_lim', 'real', 'none', '1', 'tau_nom'),
                  ('{tau_Mest}', 'tau_Mest', 'real', 'none', '1', 'tau_nom'),
                  ('{tau_Lref}', 'tau_Lref', 'real', 'none', '1/0.88', '')]
        panels = [('Speed', speed + [('{w_m_meas}', 'w_m_meas', 'real', 'none', '1/2', 'w')]),
                  ('Currents', currents), ('Flux', flux),
                  ('Angle error', [('{eps_inj}', 'eps_inj', 'angle', 'degrees', '1', ''),
                                 ('{theta_meas} - {theta_m}', 'theta_err', 'angle', 'wrap_mech_degrees',
                                  '1', '')]),
                  ('Torque', torque),
                  ('Angle', [('{theta_m} (observer)', 'theta_mest', 'angle', 'wrap', '1', ''),
                             ('{theta_meas} (measured)', 'theta_meas', 'angle', 'wrap', '1', '')])]
    elif kind in ('overview', 'injection-overview', 'presentation'):
        if kind == 'presentation':
            torque = [torque[1], torque[0], torque[2]]
        panels = [('Speed', speed), ('Flux', flux), ('Torque', torque), ('Currents', currents)]
        if kind == 'overview':
            panels.append(('Voltages', voltage))
        elif kind == 'injection-overview':
            panels.append(('Injection voltages', [('{u_inj}', 'u_inj', 'complex', 'real', '1', ''),
                                                 ('{u_inj}', 'u_inj', 'complex', 'imag', '1', '')]))
    else:
        raise ValueError(f'Unknown figure type: {kind}')
    return [{'panel': panel, 'title': title, 'id': f'{kind}:{panel}:{line}',
             'source': source.format(**names), 'scale': scale, 'offset': '0', 'divisor': divisor,
             'input': attribute, 'processing': processing, 'operations': PRESET_PROCESSING[input_kind],
             'note': _INPUT_NOTES.get(attribute, ''),
             'stage': PROCESSING_LABELS[processing] + (f' · {_INPUT_NOTES[attribute]}'
                                                       if attribute in _INPUT_NOTES else ''),
             'roles': tuple(dict.fromkeys(field for _, field, _, _ in string.Formatter().parse(source)
                                          if field))}
            for panel, (title, traces) in enumerate(panels)
            for line, (source, attribute, input_kind, processing, scale, divisor) in enumerate(traces)]


def _apply_trace_transforms(context, kind, axes, transforms):
    if not transforms:
        return
    panels = np.asarray(axes).flat
    lines = {panel: iter(axis.get_lines()) for panel, axis in enumerate(panels)}
    modified = set()
    for trace in subplot_traces(kind):
        line = next(lines[trace['panel']])
        if trace['id'] not in transforms:
            continue
        change = transforms[trace['id']]
        processing = change.get('processing', trace['processing'])
        if processing not in trace['operations']:
            raise ValueError(f"{trace['id']}: processing {processing!r} is not available; "
                             f"use one of {list(trace['operations'])}")
        scale = scalar_expression(change['scale'], context.base)
        offset = scalar_expression(change['offset'], context.base)
        divisor = getattr(context.base, trace['divisor']) if trace['divisor'] else 1.
        # Recompute from the trace's input, replacing the preset gain and processing.
        values = PROCESSING[processing](np.asarray(getattr(context, trace['input'])), context.t, context.base)
        line.set_ydata((values * scale + offset) / divisor)
        modified.add(trace['panel'])
    for panel, axis in enumerate(np.asarray(axes).flat):
        if panel in modified:
            axis.relim()
            axis.autoscale(enable=True, axis='y')


SIGNAL_UNITS = ('raw', 'A', 'V', 'rad/s', 'Nm', 'Wb', 'Vs',
                'rad/s (mechanical)', 'rpm (mechanical)',
                'p.u. (A)', 'p.u. (V)', 'p.u. (rad/s)', 'p.u. (Nm)', 'p.u. (Wb)',
                'p.u. (rad/s mechanical)', 'p.u. (Vs)')


def signal_scale(unit, base, per_unit, *, torque_mode='per_unit'):
    """Return scale and displayed unit; already-p.u. signals need a quantity.

    raw explicitly leaves a signal unchanged. rad/s denotes electrical speed;
    mechanical speed uses w_M and rpm is converted to rad/s. Vs and Wb share
    the flux base. Preset angular units and model-specific calibrations remain.
    """
    if torque_mode not in ('per_unit', 'rated'):
        raise ValueError("torque_mode must be 'per_unit' or 'rated'")
    if unit not in SIGNAL_UNITS:
        raise ValueError(f"Unknown signal unit: {unit}")
    if unit == 'raw':
        return 1., 'raw'
    already_pu = unit.startswith('p.u. (')
    physical = unit[6:-1] if already_pu else unit
    mechanical = physical in ('rad/s (mechanical)', 'rpm (mechanical)', 'rad/s mechanical')
    reference = base.w_M if mechanical else {'A': base.i, 'V': base.u, 'rad/s': base.w,
                                             'Nm': base.tau, 'Wb': base.psi, 'Vs': base.psi}[physical]
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError(f"Invalid base value for {physical}")
    if physical == 'Nm' and torque_mode == 'rated':
        if not np.isfinite(base.tau_nom) or base.tau_nom <= 0:
            raise ValueError('Invalid rated torque')
        return (reference if already_pu else 1.) / base.tau_nom, 'per rated'
    factor = 2*np.pi/60 if physical == 'rpm (mechanical)' else 1.
    if per_unit:
        return (1. if already_pu else factor/reference), 'p.u.'
    return (reference if already_pu else factor), ('rad/s' if mechanical else physical)


def _apply_preset_units(context, figure_type, axes, per_unit, torque_mode):
    base = context.base
    if torque_mode == 'rated':
        signal_scale('Nm', base, per_unit, torque_mode=torque_mode)
    if figure_type == 'hfsi':
        panels = [(axes[0, 0], base.w, 'rad/s'), (axes[1, 0], base.psi, 'Wb'),
                  (axes[0, 1], base.i, 'A')]
        torque_axis = axes[2, 0]
        factor = 1. if torque_mode == 'rated' else (base.tau_nom/base.tau if per_unit else base.tau_nom)
        label = torque_axis.get_ylabel().replace(
            'per rated', 'per rated' if torque_mode == 'rated' else ('p.u.' if per_unit else 'Nm'))
        _scale_axis(torque_axis, factor, label)
    else:
        panels = [(axes[0], base.w, 'rad/s'), (axes[1], base.psi, 'Wb'),
                  (axes[2], base.tau, 'Nm'), (axes[3], base.i, 'A')]
        if torque_mode == 'rated':
            torque_axis = axes[2]
            _scale_axis(torque_axis, base.tau/base.tau_nom,
                        torque_axis.get_ylabel().replace('p.u.', 'per rated'))
        if figure_type == 'overview':
            panels.append((axes[4], base.u, 'V'))
        elif figure_type == 'injection-overview' and per_unit is not None:
            axis = axes[4]
            _scale_axis(axis, 1./base.u if per_unit else 1.,
                        'Injection Voltages (p.u.)' if per_unit else 'Injection Voltages (V)')
    if per_unit is False:
        for axis, reference, unit in panels:
            if torque_mode == 'rated' and unit == 'Nm':
                continue
            _scale_axis(axis, reference, axis.get_ylabel().replace('p.u.', unit))


def _scale_axis(axis, factor, label):
    limits = np.asarray(axis.get_ylim()) * factor
    for line in axis.get_lines():
        line.set_ydata(np.asarray(line.get_ydata()) * factor)
    axis.set_ylim(limits)
    axis.set_ylabel(label)


def plot_measurement(context, figure_type, *, per_unit=None, transforms=None,
                     disabled_traces=None, added_traces=None, units=None,
                     torque_mode='per_unit', downsample_output=True):
    """Render a preset, optionally removing native traces or adding detached signals.

    Added trace transforms precede conversion from units[key]; missing units are
    raw. None preserves preset legacy units and selects physical units for additions.
    Rated torque uses base.tau_nom independently of other quantities' units.
    """
    try:
        function, options, _ = FIGURE_TYPES[figure_type]
    except (KeyError, TypeError):
        raise ValueError(f"figure_type must select from {tuple(FIGURE_TYPES)}") from None
    if torque_mode not in ('per_unit', 'rated'):
        raise ValueError("torque_mode must be 'per_unit' or 'rated'")
    traces = subplot_traces(figure_type)
    native_ids = {trace['id'] for trace in traces}
    panel_count = max(trace['panel'] for trace in traces) + 1
    if isinstance(disabled_traces, (str, bytes)):
        raise ValueError('disabled_traces must be a collection of trace IDs')
    disabled = set(disabled_traces or ())
    if disabled - native_ids:
        raise ValueError(f'Unknown preset trace IDs: {disabled - native_ids}')
    if added_traces is not None and not isinstance(added_traces, list):
        raise ValueError('added_traces must be a list')
    prepared = []
    ids = set(native_ids)
    for trace in added_traces or []:
        if not isinstance(trace, dict):
            raise ValueError('Each added trace must be a dictionary')
        trace_id, panel, key = trace.get('id'), trace.get('panel'), trace.get('key')
        if not isinstance(trace_id, str) or not trace_id or trace_id in ids:
            raise ValueError('Added trace IDs must be unique nonempty strings')
        ids.add(trace_id)
        if type(panel) is not int or not 0 <= panel < panel_count:
            raise ValueError(f'Invalid added trace panel: {panel}')
        if not isinstance(key, str) or not key:
            raise ValueError('Added signal keys must be nonempty strings')
        components = getattr(context, 'extra_signals', {}).get(key)
        if not components:
            raise ValueError(f"Extra signal '{key}' was not loaded")
        change = (transforms or {}).get(trace_id, {})
        scale = scalar_expression(change.get('scale', '1'), context.base)
        offset = scalar_expression(change.get('offset', '0'), context.base)
        processing = change.get('processing', 'none')
        for (true_key, _, _), (label, time, values) in zip(
                components, process_components(components, processing, context.base, key)):
            unit = (units or {}).get(true_key, (units or {}).get(key, role_unit(key)))
            factor, display_unit = signal_scale(unit, context.base, bool(per_unit),
                                                torque_mode=torque_mode)
            if processing_unit(processing, display_unit):
                factor, display_unit = 1., processing_unit(processing, display_unit)
            elif processing == 'derivative':
                display_unit += '/s'
            prepared.append((panel, f'{label} ({display_unit})',
                             time - context.t_offset, (values * scale + offset) * factor))
    existing = set(plt.get_fignums())
    try:
        fig, axes = function(context, **options)
        _apply_trace_transforms(context, figure_type, axes, transforms)
        if per_unit is not None or torque_mode == 'rated':
            _apply_preset_units(context, figure_type, axes, per_unit, torque_mode)
        if figure_type == 'hfsi' and torque_mode == 'rated':
            torque_axis = axes[2, 0]
            load_line = torque_axis.lines[2]
            load_id = 'hfsi:4:2'
            # The preset input is already base p.u.; its legacy 1/0.88 gain is
            # retained only when the caller explicitly supplies a transform.
            undo_legacy = 1. if load_id in (transforms or {}) else .88
            load_line.set_ydata(np.asarray(load_line.get_ydata()) * undo_legacy
                                * context.base.tau / context.base.tau_nom)
            if load_id in (transforms or {}):
                torque_axis.relim()
                torque_axis.autoscale(enable=True, axis='y')
        if figure_type == 'injection-overview':
            axes[4].legend(loc='center')
        flat = list(np.asarray(axes).flat)
        lines = {panel: iter(list(axis.get_lines())) for panel, axis in enumerate(flat)}
        affected = set()
        for trace in traces:
            line = next(lines[trace['panel']])
            if trace['id'] in disabled:
                line.remove()
                affected.add(trace['panel'])
        added_panels = set()
        for panel, label, time, values in prepared:
            flat[panel].plot(time, values, label=label,
                             drawstyle='steps-post' if time.size < 100 else 'default')
            affected.add(panel)
            added_panels.add(panel)
        for panel in affected:
            axis = flat[panel]
            legend = axis.get_legend()
            handles, labels = axis.get_legend_handles_labels()
            if legend is not None:
                if handles:
                    # Rebuild the entries in place to retain the preset's placement and styling.
                    title = legend.get_title().get_text()
                    location = legend._loc
                    legend._init_legend_box(handles, labels)
                    legend.set_title(title)
                    legend.set_loc(location)
                else:
                    legend.remove()
            elif handles:
                axis.legend(handles, labels)
            if panel in added_panels:
                axis.relim(visible_only=True)
                axis.autoscale(enable=True, axis='y')
        if downsample_output and figure_type in ('injection-overview', 'presentation'):
            downsample(fig)
        return fig, axes
    except Exception:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)
        raise


def plot_signals(data, groups, t_offset=0, t_span=None, x_step=None, *,
                 per_unit=False, base=None, units=None, transforms=None,
                 torque_mode='per_unit', columns=1, signal_map=None):
    """Plot signals in one panel per list, on each component's own raster.

    Arrange panels in row-major order using 1 to 8 columns; return only the
    occupied axes as a one-dimensional array in group order.

    units maps custom:{panel_index}:{key}, component keys, or selected keys to
    SIGNAL_UNITS, in that precedence order. Missing entries remain raw (an
    @role key uses its role's recorded unit); p.u. inputs include their
    physical quantity so they can be denormalized. @role keys select the
    signal_map key (default map when None) unless data already knows them.
    Rated torque uses base.tau_nom independently of other quantities' units.
    Per-line legend units distinguish mixed quantities on the same subplot.

    Select readable keys using data.keys(). Vector keys expand through
    data.signal_info(), so legends use the actual component keys rather than
    invented indices. All signal and window validation precedes figure creation.
    The recording remains owned by the caller; returned lines hold copied data.
    """
    if isinstance(columns, bool) or not isinstance(columns, int) or not 1 <= columns <= 8:
        raise ValueError("columns must be an integer between 1 and 8")
    origin, limits, step = _validate_window(t_offset, t_span, x_step)
    if (not isinstance(groups, list) or not groups
            or any(not isinstance(group, list) or not group for group in groups)):
        raise ValueError("groups must be a nonempty list of nonempty key lists")
    motor_base = DEFAULT_BASE if base is None else base
    panels = []
    for panel_index, group in enumerate(groups):
        panel = []
        for key in group:
            if not isinstance(key, str) or not key:
                raise ValueError("Signal keys must be nonempty readable strings")
            signals = data.signal_info(role_key(data, key, signal_map))
            if not signals:
                raise ValueError(f"Signal '{key}' has no components")
            components = []
            for signal in signals:
                true_key = signal["key"]
                time = np.asarray(data.time(true_key), dtype=float)
                values = np.asarray(data[true_key])
                if (time.ndim != 1 or not time.size
                        or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0)):
                    raise ValueError(f"Signal '{true_key}' needs a finite increasing time vector")
                if (values.ndim != 1 or values.size != time.size
                        or not np.issubdtype(values.dtype, np.number)
                        or np.iscomplexobj(values)):
                    raise ValueError(f"Signal '{true_key}' must be a real vector matching its time")
                components.append((true_key, time.copy(), values.copy()))
            change = (transforms or {}).get(f'custom:{panel_index}:{key}', {'scale': '1', 'offset': '0'})
            scale = scalar_expression(change['scale'], motor_base)
            offset = scalar_expression(change['offset'], motor_base)
            processing = change.get('processing', 'none')
            for (true_key, _, _), (name, time, values) in zip(
                    components, process_components(components, processing, motor_base, key)):
                unit_map = units or {}
                trace_unit = unit_map.get(f'custom:{panel_index}:{key}',
                                          unit_map.get(true_key, unit_map.get(key, role_unit(key))))
                factor, unit = signal_scale(trace_unit, motor_base, per_unit, torque_mode=torque_mode)
                if processing_unit(processing, unit):
                    factor, unit = 1., processing_unit(processing, unit)
                elif processing == 'derivative':
                    unit += '/s'
                label = f"{name} ({unit})" if units is not None else name
                panel.append((label, time - origin, (values*scale + offset)*factor))
        panels.append(panel)
    existing = set(plt.get_fignums())
    try:
        columns = min(columns, len(panels))
        rows = (len(panels) + columns - 1) // columns
        default_width, default_height = plt.rcParams['figure.figsize']
        size = (max(default_width, columns*3.2), max(default_height, rows*1.7))
        fig, grid = plt.subplots(rows, columns, sharex=True, squeeze=False, layout="tight", figsize=size)
        flat = grid.ravel()
        for ax in flat[len(panels):]:
            fig.delaxes(ax)
        axes = flat[:len(panels)]
        for index, (ax, panel) in enumerate(zip(axes, panels)):
            for key, time, values in panel:
                ax.plot(time, values, label=key,
                        drawstyle="steps-post" if time.size < 100 else "default")
            if step is not None:
                ax.xaxis.set_major_locator(MultipleLocator(step))
            ax.legend(loc="upper right")
            ax.grid(True)
            bottom = index + columns >= len(panels)
            ax.tick_params(axis="x", labelbottom=bottom)
            if bottom:
                ax.set_xlabel("Time (s)")
        if limits is not None:
            axes[-1].set_xlim(limits)
        return fig, axes
    except Exception:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)
        raise
