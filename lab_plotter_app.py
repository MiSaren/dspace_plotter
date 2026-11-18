r"""Local measurement browser. Run: .venv\Scripts\python.exe lab_plotter_app.py

Select files/folders, preview a preset or build custom subplots from recorded
signal keys. The p.u. checkbox applies to presets, previews and exports. Assign
input units of added/custom signals explicitly (including the quantity for already-p.u.
inputs); raw leaves unclassified signals unchanged. Motor settings supply base
values. Rated-torque mode uses the selected/editable motor profile's nominal
torque in Nm, even when other quantities are shown in physical units.
Preset angles retain degrees/radians. Signal time rasters are preserved.
PDF export handles selected files independently, using the selected detector's
windows and each source folder unless an output folder is specified. All files
use the motor parameters shown in the settings. No notebook is required.

The preview header shows the full source path of the displayed figure, even
when selection changes or a batch export updates the status line; it is not
included in exported PDFs.

The subplot tree lists every preset/custom trace, source and derived stage.
For a preset, select a target subplot or its trace, choose recording signals,
then add them. Toggle a preset trace off/on or remove an added trace. Added
signals use the selected input unit (raw by default), keep their own rasters,
and report a per-file error if missing. Settings last for the session and batch.
Select a trace to replace its affine gain/offset before unit normalization;
for mechanical wm_meas use gain n_p instead of the legacy 1/2. Scalar arithmetic
supports numbers, +, -, *, /, parentheses and the named motor base values.
Preview refreshes after applying/resetting an edit; batch export snapshots all
settings. Drag legends in the preview with the navigation toolbar's zoom/pan
turned off. Their axes-relative positions are remembered per figure type and
subplot for this session and applied to subsequent previews and PDF batch exports.
The reset button restores default positions for the current figure type.
Y limits default to autoscaling over the full recording. If all finite plotted
values are nonnegative, the lower limit is -0.05 in displayed units. Manual
subplot bounds override autoscaling; toggling p.u. does not rescale them. Palettes and screen/legacy LaTeX styles are independent choices.
LaTeX rendering requires a working TeX installation with newtxtext/newtxmath.
The minor-grid checkbox is remembered separately for each figure type. X major
and minor intervals are shared by all subplots; Y intervals are set per subplot.
Blank intervals preserve preset defaults, with automatic minor divisions when
needed. Intervals are in displayed units and are not rescaled when p.u. changes.
All settings apply to previews and PDF exports for this session.

references detects wm_ref/tauLref changes; spikes detects deviations beyond a
positive raw-unit threshold from a selected signal's whole-recording median.
Each detector uses original data on its own raster, before plotting transforms;
without events the full recording is used. This is not peak-prominence or
settling-time detection, and a missing selected spike signal fails that file.
"""
import copy
import sys
import tempfile
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import to_rgba
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator, ScalarFormatter

from lab_plotter_config import MOTOR_PATH, SCENARIO_PATH, load_motor_profiles, load_scenarios
from lab_plotting import LabDataMat, transient_window
from lab_plotter_core import (
    SIGNAL_UNITS,
    BaseValues,
    load_measurement,
    plot_measurement,
    plot_signals,
    scalar_expression,
    subplot_traces,
)
from plot_style import PALETTES, downsample, set_latex_style, set_screen_style

@dataclass
class SignalSnapshot:
    t: np.ndarray
    signals: dict
    times: dict
    infos: dict

    def signal_info(self, key):
        return self.infos[key]

    def __getitem__(self, key):
        return self.signals[key]

    def time(self, key):
        return self.times[key]


def load_plot_request(path, options):
    """Read files off the UI thread; return detached data, never open handles."""
    if options['kind'] not in ('custom', 'configured'):
        extra = [trace['key'] for trace in options.get('added_traces', ())]
        return load_measurement(path, options['base'], x_step=options['x_step'],
                                extra_signals=extra, **options['window'])
    with LabDataMat(path) as data:
        origin, limits = transient_window(data, **options['window'])
        keys = dict.fromkeys(key for group in options['groups'] for key in group)
        infos = {key: data.signal_info(key) for key in keys}
        components = dict.fromkeys(info['key'] for signals in infos.values() for info in signals)
        snapshot = SignalSnapshot(data.t.copy(),
                                  {key: data[key].copy() for key in components},
                                  {key: data.time(key).copy() for key in components}, infos)
    return snapshot, origin, limits


def autoscale_y(figure):
    """Scale each axis using all finite values of its visible full-length lines."""
    for axis in figure.axes:
        minimum = None
        for line in axis.get_lines():
            if not line.get_visible():
                continue
            values = np.ma.asarray(line.get_ydata(orig=False)).compressed()
            values = values[np.isfinite(values)]
            if values.size:
                value = float(np.min(values))
                minimum = value if minimum is None else min(minimum, value)
        if minimum is None:
            continue
        axis.relim(visible_only=True)
        axis.autoscale(enable=True, axis='y')
        if minimum >= 0:
            axis.set_ylim(bottom=-0.05)
        else:
            axis.set_ylim(axis.get_ylim())


def apply_y_limits(figure, limits):
    """Apply manual bounds last, in displayed units, without altering other axes."""
    for panel, bounds in limits.items():
        if panel < 0 or panel >= len(figure.axes):
            raise ValueError('The selected subplot does not exist')
        axis = figure.axes[panel]
        current = axis.get_ylim()
        lower = current[0] if bounds[0] is None else bounds[0]
        upper = current[1] if bounds[1] is None else bounds[1]
        if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
            raise ValueError('Y-axis limits must be finite and increasing in displayed units')
        axis.set_ylim(lower, upper)


def apply_grid_and_ticks(figure, *, minor_grid, x_minor_step, y_steps):
    """Apply display-unit tick intervals and minor grid without changing major grid."""
    for index, axis in enumerate(figure.axes):
        major, minor = y_steps.get(index, (None, None))
        if major is not None:
            axis.yaxis.set_major_locator(MultipleLocator(major))
            # Presets may have a fixed pi-label formatter for a different locator.
            axis.yaxis.set_major_formatter(ScalarFormatter())
        if minor is not None:
            axis.yaxis.set_minor_locator(MultipleLocator(minor))
        elif minor_grid and isinstance(axis.yaxis.get_minor_locator(), NullLocator):
            axis.yaxis.set_minor_locator(AutoMinorLocator(2))
        if x_minor_step is not None:
            axis.xaxis.set_minor_locator(MultipleLocator(x_minor_step))
        elif minor_grid and isinstance(axis.xaxis.get_minor_locator(), NullLocator):
            axis.xaxis.set_minor_locator(AutoMinorLocator(2))
        if minor_grid:
            axis.grid(True, which='minor', axis='both', linestyle=':',
                      linewidth=0.5, color='#C8C8C8')
        else:
            axis.grid(False, which='minor', axis='both')


def apply_legend_positions(figure, positions):
    """Place legends using lower-left coordinates relative to each axes."""
    for index, position in positions.items():
        if 0 <= index < len(figure.axes):
            legend = figure.axes[index].get_legend()
            if legend is not None:
                legend.set_bbox_to_anchor(None)
                legend.set_loc(tuple(position))


def render_request(payload, options):
    existing = set(plt.get_fignums())
    try:
        with plt.rc_context():
            style = options.get('style', 'screen')
            palette = options.get('palette', 'screen')
            if style == 'latex_style':
                set_latex_style(palette=palette)
            elif style == 'screen':
                set_screen_style(palette=palette)
            else:
                raise ValueError('Unknown plot style')
            if options['kind'] in ('custom', 'configured'):
                data, origin, limits = payload
                figure, axes = plot_signals(data, options['groups'], t_offset=origin,
                                           t_span=limits, x_step=options['x_step'],
                                           per_unit=options['per_unit'], base=options['base'],
                                           units=options['units'], transforms=options['transforms'],
                                           torque_mode=options.get('torque_mode', 'per_unit'),
                                           columns=options.get('columns', 1))
                titles = options.get('panel_titles', ())
                for index, axis in enumerate(axes):
                    title = titles[index] if index < len(titles) else ''
                    if title:
                        axis.set_title(title)
                    labels = [line.get_label() for line in axis.lines]
                    displayed = {label.rsplit(' (', 1)[-1][:-1] for label in labels
                                 if ' (' in label and label.endswith(')')}
                    unit = next(iter(displayed)) if len(displayed) == 1 else ('mixed' if displayed else '')
                    if title or unit:
                        axis.set_ylabel(f'{title} ({unit})' if title and unit else title or f'({unit})')
            else:
                figure, axes = plot_measurement(payload, options['kind'], per_unit=options['per_unit'],
                                                transforms=options['transforms'],
                                                disabled_traces=options.get('disabled_traces'),
                                                added_traces=options.get('added_traces'),
                                                units=options['units'],
                                                torque_mode=options.get('torque_mode', 'per_unit'),
                                                downsample_output=False)
            autoscale_y(figure)
            apply_y_limits(figure, options.get('y_limits', {}))
            if 'minor_grid' in options:
                apply_grid_and_ticks(figure, minor_grid=options['minor_grid'],
                                     x_minor_step=options['x_minor_step'],
                                     y_steps=options['y_steps'])
            apply_legend_positions(figure, options.get('legend_positions', {}))
            # Resolve C0/C1 while this palette is active, before the embedded canvas draws.
            for axis in figure.axes:
                for line in axis.lines:
                    line.set_color(to_rgba(line.get_color()))
            return figure, axes
    except Exception:  # noqa: BLE001 - release partially rendered figures before propagating
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)
        raise


class PlotterApp:
    def __init__(self, root):
        self.root = root
        root.title('Lab plotter — mittausselain')
        root.geometry('1400x900')
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.busy = False
        self.batch_active = False
        self.cancelled = False
        self.closing = False
        self.figure = self.canvas = self.toolbar = None
        self.scenarios = load_scenarios()
        self.motor_profiles = load_motor_profiles()
        self.signal_keys = []
        self.groups = []
        self.scene_groups = {'custom': self.groups}
        self.scene_panel_metadata = {}
        self.scene_signal_units = {'custom': {}}
        self.signal_units = self.scene_signal_units['custom']
        self.transform_overrides = {}
        self.default_transforms = {}
        self.preset_disabled = set()
        self.preset_added = {}
        self.trace_records = {}
        self.panel_records = {}
        self.axis_limits = {}
        self.axis_tick_steps = {}
        self.legend_positions = {}
        self.y_major_step = tk.StringVar()
        self.y_minor_step = tk.StringVar()
        self.minor_grids = {scene_id: scene['kind'] in ('hfsi', 'presentation')
                            for scene_id, scene in self.scenarios.items()}
        self.minor_grids['custom'] = False
        first_scene = 'hfsi' if 'hfsi' in self.scenarios else next(iter(self.scenarios))
        self.minor_grid = tk.BooleanVar(value=self.minor_grids[first_scene])
        self.scene_columns = {'custom': 1}
        self.columns = tk.StringVar(value='1')
        self.profile_id = tk.StringVar(value=next(iter(self.motor_profiles)))
        self.profile_baseline = None
        self.y_min = tk.StringVar()
        self.y_max = tk.StringVar()
        self.style = tk.StringVar(value='screen')
        self.palette = tk.StringVar(value='screen')
        self.detector = tk.StringVar(value='references')
        self.spike_signal = tk.StringVar()
        self.spike_threshold = tk.StringVar()
        self.edit_scale = tk.StringVar(value='1')
        self.edit_offset = tk.StringVar(value='0')
        self.trace_detail = tk.StringVar()
        self.per_unit = tk.BooleanVar(value=True)
        self.torque_mode = tk.StringVar(value='per_unit')
        self.input_unit = tk.StringVar(value='raw')
        self.path_ids = {}
        self.status = tk.StringVar(value='Lisää mittaustiedostoja tai kansio.')
        self.preview_source = tk.StringVar(value='Esikatselu: ei kuvaa')
        self.kind = tk.StringVar(value=first_scene)
        self.active_scene = first_scene
        self.scenario_choice = tk.StringVar()
        self._initialize_scenarios(self.scenarios)
        self.signal_units = self.scene_signal_units[first_scene]
        self.groups = self.scene_groups.get(first_scene, self.scene_groups['custom'])
        self.columns.set(str(self.scene_columns.get(first_scene, 1)))
        self.output = tk.StringVar()
        self.search = tk.StringVar()
        self.parameters = {}
        self._build()
        self.kind.trace_add('write', self.figure_kind_changed)
        self.figure_kind_changed()
        self._config_stamps = self.config_stamps()
        self._failed_config_stamps = None
        self._config_timer = root.after(1200, self.poll_configuration)
        root.protocol('WM_DELETE_WINDOW', self.close)

    def scene_kind(self, scene_id=None):
        scene_id = self.kind.get() if scene_id is None else scene_id
        return 'custom' if scene_id == 'custom' else self.scenarios[scene_id]['kind']

    @staticmethod
    def _native_ui_id(scene_id, native_id):
        return f'{scene_id}:{native_id.split(":", 1)[1]}'

    def _initialize_scenarios(self, scenarios, changed=None):
        changed = set(scenarios) if changed is None else set(changed)
        for scene_id in changed:
            scene = scenarios.get(scene_id)
            self.preset_disabled = {key for key in self.preset_disabled
                                    if not key.startswith(f'{scene_id}:')}
            self.transform_overrides = {
                key: value for key, value in self.transform_overrides.items()
                if not key.startswith((f'{scene_id}:', f'configadded:{scene_id}:', f'added:{scene_id}:'))}
            self.default_transforms = {
                key: value for key, value in self.default_transforms.items()
                if not key.startswith((f'{scene_id}:', f'configadded:{scene_id}:', f'added:{scene_id}:'))}
            self.preset_added.pop(scene_id, None)
            self.scene_signal_units.pop(scene_id, None)
            self.scene_groups.pop(scene_id, None)
            self.scene_panel_metadata.pop(scene_id, None)
            if scene is None:
                self.scene_columns.pop(scene_id, None)
                continue
            kind = scene['kind']
            self.scene_signal_units[scene_id] = {}
            self.minor_grids.setdefault(scene_id, kind in ('hfsi', 'presentation'))
            if kind == 'configured':
                self.scene_columns[scene_id] = scene['columns']
                groups = [[signal['key'] for signal in panel['signals']] for panel in scene['panels']]
                self.scene_groups[scene_id] = groups
                self.scene_panel_metadata[scene_id] = copy.deepcopy(scene['panels'])
                for panel, definition in enumerate(scene['panels']):
                    for signal in definition['signals']:
                        trace_id = f'{scene_id}:{panel}:{signal["key"]}'
                        change = {'scale': signal['scale'], 'offset': signal['offset']}
                        self.default_transforms[trace_id] = change.copy()
                        self.transform_overrides[trace_id] = change.copy()
            else:
                self.preset_added[scene_id] = []
                added_map = {}
                for trace in scene['added_traces']:
                    ui_id = f'configadded:{scene_id}:{trace["id"]}'
                    added_map[trace['id']] = ui_id
                    self.preset_added[scene_id].append({'id': ui_id, 'panel': trace['panel'], 'key': trace['key']})
                self.preset_disabled.update(self._native_ui_id(scene_id, trace_id)
                                            for trace_id in scene['disabled_traces'])
                for trace_id, transform in scene['transforms'].items():
                    ui_id = added_map.get(trace_id)
                    if ui_id is None:
                        ui_id = self._native_ui_id(scene_id, trace_id)
                    self.default_transforms[ui_id] = transform.copy()
                    self.transform_overrides[ui_id] = transform.copy()

    def update_scenario_choices(self):
        self.scenario_choices = {f'{scene["label"]} [{scene_id}]': scene_id
                                 for scene_id, scene in self.scenarios.items()}
        self.scenario_choices['Vapaa kuva [custom]'] = 'custom'
        self.kind_selector.configure(values=tuple(self.scenario_choices))
        for label, scene_id in self.scenario_choices.items():
            if scene_id == self.kind.get():
                self.scenario_choice.set(label)
                break

    def choose_scenario(self, _event=None):
        scene_id = self.scenario_choices.get(self.scenario_choice.get())
        if scene_id is not None:
            self.kind.set(scene_id)
            self.refresh_groups(switch_to_custom=False)

    def update_profile_choices(self):
        self.profile_choices = {f'{profile["name"]} [{profile_id}]': profile_id
                                for profile_id, profile in self.motor_profiles.items()}
        self.profile_selector.configure(values=tuple(self.profile_choices))
        for label, profile_id in self.profile_choices.items():
            if profile_id == self.profile_id.get():
                self.profile_choice.set(label)
                break

    def apply_profile(self, profile_id):
        profile = self.motor_profiles[profile_id]
        for key in ('U_nom', 'I_nom', 'f_nom', 'P_nom', 'tau_nom', 'n_p'):
            self.parameters[key].set(str(profile[key]))
        self.profile_id.set(profile_id)
        self.profile_baseline = {key: self.parameters[key].get()
                                 for key in ('U_nom', 'I_nom', 'f_nom', 'P_nom', 'tau_nom', 'n_p')}
        self.update_profile_choices()

    def choose_profile(self, _event=None):
        profile_id = self.profile_choices.get(self.profile_choice.get())
        if profile_id is not None:
            self.apply_profile(profile_id)
            self.units_changed()

    @staticmethod
    def config_stamps():
        def stamp(path):
            try:
                stat = path.stat()
                return stat.st_mtime_ns, stat.st_size
            except OSError:
                return None
        return stamp(SCENARIO_PATH), stamp(MOTOR_PATH)

    def poll_configuration(self):
        if self.closing:
            return
        stamps = self.config_stamps()
        if not self.busy and not self.batch_active and stamps != self._config_stamps and stamps != self._failed_config_stamps:
            self.reload_configuration(automatic=True)
        self._config_timer = self.root.after(1200, self.poll_configuration)

    def reload_configuration(self, automatic=False):
        if self.busy or self.batch_active:
            self.status.set('Asetukset ladataan vasta käynnissä olevan työn jälkeen.')
            return
        stamps = self.config_stamps()
        try:
            scenarios = load_scenarios()
            profiles = load_motor_profiles()
        except ValueError as error:
            self._failed_config_stamps = stamps
            self.status.set(f'Asetustiedoston virhe; aiemmat asetukset säilytetty: {error}')
            if not automatic:
                messagebox.showerror('Asetustiedoston virhe', str(error), parent=self.root)
            return
        old_scenarios = self.scenarios
        changed = {key for key in set(old_scenarios) | set(scenarios)
                   if old_scenarios.get(key) != scenarios.get(key)}
        dirty = any(self.parameters[key].get() != value for key, value in self.profile_baseline.items())
        old_profile = self.motor_profiles.get(self.profile_id.get())
        self.scenarios = scenarios
        self.motor_profiles = profiles
        self._initialize_scenarios(scenarios, changed)
        for scene_id in changed:
            for mapping in (self.axis_limits, self.axis_tick_steps, self.legend_positions):
                for key in list(mapping):
                    if key.startswith(f'{scene_id}:'):
                        del mapping[key]
        if self.kind.get() not in scenarios and self.kind.get() != 'custom':
            self.kind.set(next(iter(scenarios)))
        else:
            self.figure_kind_changed()
        self.update_scenario_choices()
        selected = self.profile_id.get()
        if selected not in profiles:
            selected = next(iter(profiles))
            self.profile_id.set(selected)
        self.update_profile_choices()
        profile_changed = old_profile != profiles[selected]
        if profile_changed and not dirty:
            self.apply_profile(selected)
        self._config_stamps = stamps
        self._failed_config_stamps = None
        self.refresh_groups(switch_to_custom=False)
        self.status.set('Asetukset päivitetty.' +
                        (' Muokatut moottoriarvot säilytettiin; valitse profiili uudelleen päivittääksesi ne.'
                         if profile_changed and dirty else '') +
                        (' Muuttuneiden skenaarioiden istuntokohtaiset muokkaukset nollattiin.' if changed else ''))
        if (changed or (profile_changed and not dirty)) and self.figure is not None:
            self.preview()

    def _build(self):
        pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True)
        controls = ttk.Frame(pane)
        pane.add(controls, weight=0)
        control_canvas = tk.Canvas(controls, width=480, highlightthickness=0)
        control_scroll = ttk.Scrollbar(controls, orient=tk.VERTICAL, command=control_canvas.yview)
        control_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        control_canvas.configure(yscrollcommand=control_scroll.set)
        sidebar = ttk.Frame(control_canvas, padding=8)
        window = control_canvas.create_window((0, 0), window=sidebar, anchor='nw')
        sidebar.bind('<Configure>', lambda _event: control_canvas.configure(
            scrollregion=control_canvas.bbox('all')))
        control_canvas.bind('<Configure>', lambda event: control_canvas.itemconfigure(
            window, width=event.width))
        self.plot_area = ttk.Frame(pane)
        pane.add(self.plot_area, weight=1)
        preview_label = ttk.Label(self.plot_area, textvariable=self.preview_source,
                                  anchor=tk.W, padding=6, wraplength=600)
        preview_label.pack(side=tk.TOP, fill=tk.X)
        self.plot_area.bind('<Configure>', lambda event: preview_label.configure(
            wraplength=max(200, event.width - 12)))
        buttons = ttk.Frame(sidebar)
        buttons.pack(fill=tk.X)
        for label, command in [('Tiedostot…', self.add_files), ('Kansio…', self.add_folder),
                               ('Poista', self.remove_files)]:
            ttk.Button(buttons, text=label, command=command).pack(side=tk.LEFT)
        ttk.Label(sidebar, text='Ctrl/Shift: monivalinta; Ctrl+A: kaikki; aktiivinen rivi esikatseluun').pack(anchor=tk.W)
        frame = ttk.Frame(sidebar)
        frame.pack(fill=tk.X)
        self.files = ttk.Treeview(frame, columns=('state',), height=7, selectmode='extended')
        self.files.heading('#0', text='Mittaus')
        self.files.heading('state', text='Tila')
        self.files.column('#0', width=290)
        self.files.column('state', width=100)
        self.files.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.files.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.files.configure(yscrollcommand=scroll.set)
        self.files.bind('<<TreeviewSelect>>', self.selection_changed)
        self.files.bind('<Control-a>', self.select_all)
        self.file_label = ttk.Label(sidebar, text='', wraplength=400)
        self.file_label.pack(anchor=tk.W)

        settings = ttk.LabelFrame(sidebar, text='Kuva ja aikaikkuna', padding=5)
        settings.pack(fill=tk.X, pady=5)
        self.kind_selector = ttk.Combobox(settings, textvariable=self.scenario_choice, state='readonly')
        self.kind_selector.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.kind_selector.bind('<<ComboboxSelected>>', self.choose_scenario)
        self.update_scenario_choices()
        for row, (key, label, value) in enumerate([
                ('pre', 'Ennen transienttia (s)', '.2'),
                ('post', 'Viimeisen transientin jälkeen (s)', '.2'),
                ('duration', 'Pituus (s), tyhjä = automaattinen', ''),
                ('x_step', 'X pääjako (s), tyhjä = oletus', ''),
                ('x_minor_step', 'X apujako (s), tyhjä = oletus', ''),
                ('offset', 'Nollakohta (s), tyhjä = autom.', '')], 1):
            self.parameters[key] = tk.StringVar(value=value)
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky='w')
            ttk.Entry(settings, textvariable=self.parameters[key], width=12).grid(row=row, column=1)
        ttk.Checkbutton(settings, text='Käytä p.u.-arvoja (kaikki kuvat)',
                        variable=self.per_unit, command=self.units_changed).grid(
                            row=7, column=0, columnspan=2, sticky='w')
        ttk.Checkbutton(settings, text='Näytä apuruudukko (minor grid)',
                        variable=self.minor_grid, command=self.minor_grid_changed).grid(
                            row=8, column=0, columnspan=2, sticky='w')
        ttk.Label(settings, text='Kulmat säilyvät deg/rad-yksiköissä.').grid(
            row=9, column=0, columnspan=2, sticky='w')
        ttk.Label(settings, text='Palstat (custom / asetusskenaario)').grid(row=10, column=0, sticky='w')
        self.columns_entry = ttk.Entry(settings, textvariable=self.columns, width=12)
        self.columns_entry.grid(row=10, column=1)
        ttk.Button(settings, text='Lataa asetukset uudelleen', command=self.reload_configuration).grid(
            row=11, column=0, columnspan=2, sticky='w')
        appearance = ttk.LabelFrame(sidebar, text='Ulkoasu ja väripaletti', padding=5)
        appearance.pack(fill=tk.X, pady=5)
        ttk.Label(appearance, text='Tyyli').grid(row=0, column=0, sticky='w')
        style_box = ttk.Combobox(appearance, textvariable=self.style,
                                 values=('screen', 'latex_style'), state='readonly', width=18)
        style_box.grid(row=0, column=1)
        style_box.bind('<<ComboboxSelected>>', self.style_changed)
        ttk.Label(appearance, text='Paletti').grid(row=1, column=0, sticky='w')
        palette_box = ttk.Combobox(appearance, textvariable=self.palette,
                                   values=tuple(PALETTES), state='readonly', width=18)
        palette_box.grid(row=1, column=1)
        palette_box.bind('<<ComboboxSelected>>', lambda _event: self.units_changed())
        ttk.Label(appearance, text='latex_style käyttää vanhaa julkaisutyyliä; vaatii TeX-asennuksen '
                  'ja newtx-paketit. Paletti valitaan erikseen.', wraplength=430).grid(
                      row=2, column=0, columnspan=2, sticky='w')
        detection = ttk.LabelFrame(sidebar, text='Transienttitunnistus', padding=5)
        detection.pack(fill=tk.X, pady=5)
        detector_box = ttk.Combobox(detection, textvariable=self.detector,
                                    values=('references', 'spikes'), state='readonly', width=20)
        detector_box.grid(row=0, column=0, columnspan=2, sticky='ew')
        detector_box.bind('<<ComboboxSelected>>', lambda _event: self.units_changed())
        ttk.Label(detection, text='Piikkisignaali').grid(row=1, column=0, sticky='w')
        self.spike_selector = ttk.Combobox(detection, textvariable=self.spike_signal,
                                           values=(), state='readonly', width=30)
        self.spike_selector.grid(row=1, column=1)
        ttk.Label(detection, text='Amplitudikynnys (raakayksikkö)').grid(row=2, column=0, sticky='w')
        ttk.Entry(detection, textvariable=self.spike_threshold, width=12).grid(row=2, column=1)
        ttk.Label(detection, text='references: wm_ref- ja tauLref/TL_ref-ohjeiden muutokset. '
                  'spikes: |y − koko signaalin mediaani| > kynnys valitussa mittaussignaalissa. '
                  'Tunnistus käyttää alkuperäistä dataa, ei p.u.-muunnoksia. Ei tapahtumia → koko tallenne.',
                  wraplength=430).grid(row=3, column=0, columnspan=2, sticky='w')
        motor = ttk.LabelFrame(sidebar, text='Moottori — samat arvot kaikille valituille', padding=5)
        motor.pack(fill=tk.X)
        ttk.Label(motor, text='Moottoriprofiili').grid(row=0, column=0, sticky='w')
        self.profile_choice = tk.StringVar()
        self.profile_selector = ttk.Combobox(motor, textvariable=self.profile_choice,
                                             state='readonly', width=23)
        self.profile_selector.grid(row=0, column=1, columnspan=2, sticky='ew')
        self.profile_selector.bind('<<ComboboxSelected>>', self.choose_profile)
        profile = self.motor_profiles[self.profile_id.get()]
        labels = [('U_nom', 'U RMS, pääjännite (V)'), ('I_nom', 'I RMS (A)'),
                  ('f_nom', 'f (Hz)'), ('P_nom', 'P (W)'),
                  ('tau_nom', 'T nimellinen (Nm)'), ('n_p', 'Napaparit')]
        for index, (key, label) in enumerate(labels):
            self.parameters[key] = tk.StringVar(value=str(profile[key]))
            row, col = divmod(index, 3)
            ttk.Label(motor, text=label).grid(row=1 + row*2, column=col)
            ttk.Entry(motor, textvariable=self.parameters[key], width=12).grid(row=2 + row*2, column=col)
        self.profile_baseline = {key: self.parameters[key].get() for key, _ in labels}
        self.update_profile_choices()
        ttk.Label(motor, text='Momentin esitystapa').grid(row=5, column=0, sticky='w')
        torque_selector = ttk.Combobox(motor, textvariable=self.torque_mode,
                                      values=('per_unit', 'rated'), state='readonly', width=12)
        torque_selector.grid(row=5, column=1, sticky='w')
        torque_selector.bind('<<ComboboxSelected>>', self.torque_mode_changed)
        ttk.Label(motor, text='rated: momentit / moottoriprofiilin nimellismomentti; '
                  'muut suureet noudattavat p.u.-valintaa.', wraplength=430).grid(
                      row=6, column=0, columnspan=3, sticky='w')

        custom = ttk.LabelFrame(sidebar, text='Mittaussignaalit / custom-alikuvaajat', padding=5)
        custom.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Entry(custom, textvariable=self.search).pack(fill=tk.X)
        self.search.trace_add('write', lambda *_: self.filter_signals())
        listframe = ttk.Frame(custom)
        listframe.pack(fill=tk.BOTH, expand=True)
        self.signals = tk.Listbox(listframe, selectmode=tk.EXTENDED, exportselection=False, height=7)
        self.signals.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(listframe, command=self.signals.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.signals.configure(yscrollcommand=scroll.set)
        self.signals.bind('<<ListboxSelect>>', self.unit_selection_changed)
        unit_row = ttk.Frame(custom)
        unit_row.pack(fill=tk.X)
        ttk.Label(unit_row, text='Tallennetun signaalin yksikkö:').pack(side=tk.LEFT)
        ttk.Combobox(unit_row, textvariable=self.input_unit, values=SIGNAL_UNITS,
                     state='readonly', width=14).pack(side=tk.LEFT)
        ttk.Button(unit_row, text='Aseta valituille', command=self.assign_units).pack(side=tk.LEFT)
        action = ttk.Frame(custom)
        action.pack(fill=tk.X)
        for label, command in [('Lisää alikuvaaja', self.add_group),
                               ('Korvaa valittu', self.replace_group),
                               ('Poista alikuvaaja', self.remove_group)]:
            ttk.Button(action, text=label, command=command).pack(side=tk.LEFT)
        self.group_list = tk.Listbox(custom, height=4, exportselection=False)
        self.group_list.pack(fill=tk.X)
        ttk.Label(custom, text='Ctrl/Shift: useita signaaleja samalle akselille. raw = ei muunnosta; '
                  'p.u. (Nm) ym. = jo normalisoitu syöte. Valitse ensin syötteen yksikkö ja paina Aseta valituille.',
                  wraplength=420).pack(anchor=tk.W)
        editor = ttk.LabelFrame(sidebar, text='Alikuvaajien signaalit ja laskutoimitukset', padding=5)
        editor.pack(fill=tk.X, pady=5)
        tree_frame = ttk.Frame(editor)
        tree_frame.pack(fill=tk.X)
        self.trace_tree = ttk.Treeview(tree_frame, columns=('stage', 'scale', 'offset'),
                                      selectmode='browse', height=9)
        self.trace_tree.heading('#0', text='Alikuvaaja / signaalilähde')
        self.trace_tree.heading('stage', text='Nykyinen johdettu käsittely')
        self.trace_tree.heading('scale', text='Kerroin')
        self.trace_tree.heading('offset', text='Lisäys')
        self.trace_tree.column('#0', width=210)
        self.trace_tree.column('stage', width=250)
        self.trace_tree.column('scale', width=90)
        self.trace_tree.column('offset', width=70)
        self.trace_tree.grid(row=0, column=0, sticky='nsew')
        vertical = ttk.Scrollbar(tree_frame, command=self.trace_tree.yview)
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.trace_tree.xview)
        horizontal.grid(row=1, column=0, sticky='ew')
        tree_frame.columnconfigure(0, weight=1)
        self.trace_tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.trace_tree.bind('<<TreeviewSelect>>', self.trace_selected)
        signal_actions = ttk.Frame(editor)
        signal_actions.pack(fill=tk.X)
        ttk.Button(signal_actions, text='Lisää valitut signaalit',
                   command=self.add_preset_signals).pack(side=tk.LEFT)
        ttk.Button(signal_actions, text='Poista / palauta signaali',
                   command=self.toggle_preset_signal).pack(side=tk.LEFT)
        ttk.Button(signal_actions, text='Palauta signaalit',
                   command=self.reset_preset_signals).pack(side=tk.LEFT)
        ttk.Label(editor, text='Esiasetus: valitse tästä kohdealikuvaaja sekä yllä mittaussignaalit ja '
                  'niiden syöttöyksiköt. Piilotetun alkuperäisen voi palauttaa samalla painikkeella. '
                  'Lisätyt signaalit käyttävät tallenteen omaa aikarasteria.', wraplength=430).pack(anchor=tk.W)
        ttk.Label(editor, textvariable=self.trace_detail, wraplength=430).pack(anchor=tk.W)
        y_row = ttk.Frame(editor)
        y_row.pack(fill=tk.X)
        ttk.Label(y_row, text='Y min').pack(side=tk.LEFT)
        ttk.Entry(y_row, textvariable=self.y_min, width=12).pack(side=tk.LEFT)
        ttk.Label(y_row, text='Y max').pack(side=tk.LEFT)
        ttk.Entry(y_row, textvariable=self.y_max, width=12).pack(side=tk.LEFT)
        y_buttons = ttk.Frame(editor)
        y_buttons.pack(fill=tk.X)
        ttk.Button(y_buttons, text='Aseta Y-rajat', command=self.assign_y_limits).pack(side=tk.LEFT)
        ttk.Button(y_buttons, text='Oletusrajat', command=self.reset_y_limits).pack(side=tk.LEFT)
        ttk.Label(editor, text='Valitse alikuvaaja tai sen signaali. Tyhjä raja = oletus/autom. '
                  'Rajojen luvut ovat näkyvissä yksiköissä; p.u.-vaihto ei muunna syötettyjä lukuja.',
                  wraplength=430).pack(anchor=tk.W)
        ticks_row = ttk.Frame(editor)
        ticks_row.pack(fill=tk.X)
        ttk.Label(ticks_row, text='Y pääjako').pack(side=tk.LEFT)
        ttk.Entry(ticks_row, textvariable=self.y_major_step, width=10).pack(side=tk.LEFT)
        ttk.Label(ticks_row, text='Y apujako').pack(side=tk.LEFT)
        ttk.Entry(ticks_row, textvariable=self.y_minor_step, width=10).pack(side=tk.LEFT)
        ticks_buttons = ttk.Frame(editor)
        ticks_buttons.pack(fill=tk.X)
        ttk.Button(ticks_buttons, text='Aseta Y-jaot', command=self.assign_y_ticks).pack(side=tk.LEFT)
        ttk.Button(ticks_buttons, text='Palauta Y-jaot', command=self.reset_y_ticks).pack(side=tk.LEFT)
        ttk.Label(editor, text='Jaot ovat näkyvissä yksiköissä. Tyhjä = oletus; '
                  'p.u.-vaihto ei muunna asetettuja jakoja.', wraplength=430).pack(anchor=tk.W)
        edit_row = ttk.Frame(editor)
        edit_row.pack(fill=tk.X)
        ttk.Label(edit_row, text='Kerroin').pack(side=tk.LEFT)
        ttk.Entry(edit_row, textvariable=self.edit_scale, width=14).pack(side=tk.LEFT)
        ttk.Label(edit_row, text='Lisäys').pack(side=tk.LEFT)
        ttk.Entry(edit_row, textvariable=self.edit_offset, width=14).pack(side=tk.LEFT)
        edit_buttons = ttk.Frame(editor)
        edit_buttons.pack(fill=tk.X)
        ttk.Button(edit_buttons, text='Aseta muunnos', command=self.assign_transform).pack(side=tk.LEFT)
        ttk.Button(edit_buttons, text='Palauta valittu oletukseen', command=self.reset_transform).pack(side=tk.LEFT)
        ttk.Label(editor, text='y → kerroin × y + lisäys ennen p.u.-muunnosta. '
                  'Mekaaninen → sähköinen nopeus: kerroin n_p (korvaa nykyisen kertoimen). '
                  'Sallittu esim. 1/n_p, 2*n_p, -1, i_base. Muokatun akselin y-rajaus on automaattinen.',
                  wraplength=430).pack(anchor=tk.W)
        self.refresh_trace_editor()
        export = ttk.LabelFrame(sidebar, text='PDF-tallennus', padding=5)
        export.pack(fill=tk.X)
        ttk.Label(export, text='Tyhjä = kunkin mittauksen lähdekansio').pack(anchor=tk.W)
        ttk.Entry(export, textvariable=self.output).pack(fill=tk.X)
        ttk.Button(export, text='Tallennuskansio…', command=self.choose_output).pack(side=tk.LEFT)
        ttk.Button(export, text='Lähdekansio', command=lambda: self.output.set('')).pack(side=tk.LEFT)
        actions = ttk.Frame(sidebar)
        actions.pack(fill=tk.X, pady=5)
        self.preview_button = ttk.Button(actions, text='Esikatsele', command=self.preview)
        self.preview_button.pack(side=tk.LEFT)
        self.export_button = ttk.Button(actions, text='Tallenna valitut PDF', command=self.export)
        self.export_button.pack(side=tk.LEFT)
        ttk.Button(actions, text='Keskeytä eräajo', command=self.cancel).pack(side=tk.LEFT)
        ttk.Button(sidebar, text='Palauta tämän kuvatyypin legendit',
                   command=self.reset_legends).pack(anchor=tk.W)
        ttk.Label(sidebar, text='Raahaa legendoja esikatselussa (zoom/siirto pois). '
                  'Sijainnit säilyvät tämän istunnon PDF-eräajossa.', wraplength=430).pack(anchor=tk.W)
        ttk.Label(self.root, textvariable=self.status, anchor=tk.W, padding=5).pack(fill=tk.X)

    def select_all(self, _event=None):
        self.files.selection_set(self.files.get_children())
        return 'break'

    def selected_paths(self):
        return [self.path_ids[item] for item in self.files.selection()]

    def active_path(self):
        selected = self.files.selection()
        focus = self.files.focus()
        return self.path_ids[focus if focus in selected else selected[0]] if selected else None

    def insert_paths(self, paths):
        first = None
        for path in sorted({Path(p).resolve() for p in paths}, key=lambda p: str(p).casefold()):
            if path not in self.path_ids.values():
                item = self.files.insert('', tk.END, text=path.name, values=('Lisätty',))
                self.path_ids[item] = path
                first = first or item
        if first:
            self.files.selection_set(first)
            self.files.focus(first)

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[('MAT-mittaukset', '*.mat *.MAT'), ('Kaikki', '*.*')])
        self.insert_paths(paths)

    def add_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.insert_paths(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() == '.mat')

    def remove_files(self):
        if self.busy:
            return
        for item in self.files.selection():
            del self.path_ids[item]
            self.files.delete(item)
        self.signal_keys = []
        self.filter_signals()

    def choose_output(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output.set(folder)

    def selection_changed(self, _event=None):
        path = self.active_path()
        if not path or self.busy or self.batch_active:
            return
        self.file_label.configure(text=str(path))

        def read_keys():
            with LabDataMat(path) as data:
                return data.keys()

        def loaded(keys):
            if self.active_path() == path:
                self.signal_keys = keys
                self.spike_selector.configure(values=keys)
                self.filter_signals()
                self.status.set(f'{path.name}: {len(keys)} signaalia. Valitse kuva tai kokoa alikuvaajat.')
            else:
                self.selection_changed()
        self.start_job(read_keys, loaded)

    def filter_signals(self):
        selected = {self.signals.get(index) for index in self.signals.curselection()}
        self.signals.delete(0, tk.END)
        query = self.search.get().casefold()
        for key in self.signal_keys:
            if query in key.casefold():
                self.signals.insert(tk.END, key)
                if key in selected:
                    self.signals.selection_set(tk.END)

    def current_traces(self):
        scene_id = self.kind.get()
        kind = self.scene_kind(scene_id)
        traces = subplot_traces('custom' if kind == 'configured' else kind, self.groups)
        for trace in traces:
            native_id = trace['id']
            if scene_id != kind or kind == 'configured':
                trace['id'] = self._native_ui_id(scene_id, native_id)
            if kind == 'configured' and trace['panel'] < len(self.scene_panel_metadata[scene_id]):
                trace['title'] = self.scene_panel_metadata[scene_id][trace['panel']]['title']
            if kind == 'hfsi' and native_id == 'hfsi:4:2' and self.torque_mode.get() == 'rated':
                trace['stage'] = 'recorded p.u.; muunnetaan nimellismomenttiin'
                trace['scale'] = '1'
        if kind not in ('custom', 'configured'):
            titles = {trace['panel']: trace['title'] for trace in traces}
            for added in self.preset_added[scene_id]:
                traces.append({'id': added['id'], 'panel': added['panel'],
                               'title': titles[added['panel']], 'source': added['key'],
                               'stage': 'lisätty, tallenteen oma aikarasteri',
                               'scale': '1', 'offset': '0', 'divisor': ''})
        return traces

    def refresh_trace_editor(self, select_id=None):
        if select_id is None:
            selection = self.trace_tree.selection()
            select_id = selection[0] if selection else None
        self.trace_tree.delete(*self.trace_tree.get_children())
        self.trace_records = {trace['id']: trace for trace in self.current_traces()}
        self.panel_records = {}
        parents = {}
        for trace in self.trace_records.values():
            panel = trace['panel']
            if panel not in parents:
                parent = f'panel:{self.kind.get()}:{panel}'
                parents[panel] = self.trace_tree.insert('', tk.END, iid=parent, text=trace['title'], open=True)
                self.panel_records[parent] = {'key': f'{self.kind.get()}:{panel}', 'title': trace['title']}
            change = self.transform_overrides.get(trace['id'], trace)
            name = trace['source'] + (' [pois]' if trace['id'] in self.preset_disabled else '')
            self.trace_tree.insert(parents[panel], tk.END, iid=trace['id'], text=name,
                                   values=(trace['stage'], change['scale'], change['offset']))
        if select_id in self.trace_records or select_id in self.panel_records:
            self.trace_tree.selection_set(select_id)
            self.trace_selected()
        else:
            self.y_min.set('')
            self.y_max.set('')
            self.y_major_step.set('')
            self.y_minor_step.set('')
            self.trace_detail.set('Valitse alikuvaaja sen Y-rajoja varten tai signaali sen laskutoimituksia varten.')

    def add_preset_signals(self):
        kind = self.kind.get()
        panel_key = self.selected_panel_key()
        if self.scene_kind() in ('custom', 'configured') or panel_key is None:
            self.status.set('Valitse ensin valmiin esiasetuksen kohdealikuvaaja tai sen signaali.')
            return
        signals = [self.signals.get(index) for index in self.signals.curselection()]
        if not signals:
            self.status.set('Valitse lisättävät mittaussignaalit signaaliluettelosta.')
            return
        panel = int(panel_key.rsplit(':', 1)[1])
        added = self.preset_added[kind]
        existing = {(item['panel'], item['key']) for item in added}
        new_ids = []
        for key in signals:
            if (panel, key) not in existing:
                item = {'id': f'added:{kind}:{panel}:{uuid4().hex}', 'panel': panel, 'key': key}
                added.append(item)
                existing.add((panel, key))
                new_ids.append(item['id'])
        if new_ids:
            self.refresh_trace_editor(new_ids[-1])
            self.units_changed()
        else:
            self.status.set('Valitut signaalit ovat jo kyseisessä alikuvaajassa.')

    def toggle_preset_signal(self):
        kind = self.kind.get()
        selected = self.trace_tree.selection()
        if self.scene_kind() in ('custom', 'configured') or not selected or selected[0] not in self.trace_records:
            self.status.set('Valitse valmiin esiasetuksen yksittäinen signaali.')
            return
        trace_id = selected[0]
        added = self.preset_added[kind]
        item = next((item for item in added if item['id'] == trace_id), None)
        if item is not None:
            added.remove(item)
            self.transform_overrides.pop(trace_id, None)
            panel = f'panel:{kind}:{item["panel"]}'
            self.refresh_trace_editor(panel)
        else:
            if trace_id in self.preset_disabled:
                self.preset_disabled.remove(trace_id)
            else:
                self.preset_disabled.add(trace_id)
            self.refresh_trace_editor(trace_id)
        self.units_changed()

    def reset_preset_signals(self):
        kind = self.kind.get()
        if self.scene_kind() in ('custom', 'configured'):
            self.status.set('Valitse valmis esiasetus palautettavaksi.')
            return
        self._initialize_scenarios(self.scenarios, {kind})
        self.signal_units = self.scene_signal_units[kind]
        self.refresh_trace_editor()
        self.units_changed()

    def selected_panel_key(self):
        selected = self.trace_tree.selection()
        if not selected:
            return None
        if selected[0] in self.panel_records:
            return self.panel_records[selected[0]]['key']
        if selected[0] in self.trace_records:
            return f'{self.kind.get()}:{self.trace_records[selected[0]]["panel"]}'
        return None

    def trace_selected(self, _event=None):
        selected = self.trace_tree.selection()
        key = self.selected_panel_key()
        if key is None:
            return
        bounds = self.axis_limits.get(key, (None, None))
        self.y_min.set('' if bounds[0] is None else str(bounds[0]))
        self.y_max.set('' if bounds[1] is None else str(bounds[1]))
        steps = self.axis_tick_steps.get(key, (None, None))
        self.y_major_step.set('' if steps[0] is None else str(steps[0]))
        self.y_minor_step.set('' if steps[1] is None else str(steps[1]))
        if selected[0] not in self.trace_records:
            self.trace_detail.set(f'{self.panel_records[selected[0]]["title"]}: muokkaa Y-akselin rajoja.')
            return
        trace = self.trace_records[selected[0]]
        change = self.transform_overrides.get(trace['id'], trace)
        self.edit_scale.set(change['scale'])
        self.edit_offset.set(change['offset'])
        self.trace_detail.set(f'{trace["title"]}: {trace["source"]}. '
                              f'Johdettu käsittely: {trace["stage"] or "ei"}. '
                              f'Oletuskerroin: {trace["scale"]}.')

    def assign_y_limits(self):
        key = self.selected_panel_key()
        if key is None:
            return
        try:
            values = [entry.get().strip().replace(',', '.') for entry in (self.y_min, self.y_max)]
            bounds = tuple(float(value) if value else None for value in values)
            if any(value is not None and not np.isfinite(value) for value in bounds):
                raise ValueError('Y-rajojen on oltava äärellisiä')
            if bounds[0] is not None and bounds[1] is not None and bounds[1] <= bounds[0]:
                raise ValueError('Y max on oltava suurempi kuin Y min')
        except ValueError as error:
            self.report_error(error)
            return
        if bounds == (None, None):
            self.axis_limits.pop(key, None)
        else:
            self.axis_limits[key] = bounds
        self.units_changed()

    def reset_y_limits(self):
        self.y_min.set('')
        self.y_max.set('')
        self.assign_y_limits()

    def assign_y_ticks(self):
        key = self.selected_panel_key()
        if key is None:
            return
        try:
            values = [entry.get().strip().replace(',', '.')
                      for entry in (self.y_major_step, self.y_minor_step)]
            steps = tuple(float(value) if value else None for value in values)
            if any(value is not None and (not np.isfinite(value) or value <= 0) for value in steps):
                raise ValueError('Y-jaon on oltava positiivinen ja äärellinen')
        except ValueError as error:
            self.report_error(error)
            return
        if steps == (None, None):
            self.axis_tick_steps.pop(key, None)
        else:
            self.axis_tick_steps[key] = steps
        self.units_changed()

    def reset_y_ticks(self):
        self.y_major_step.set('')
        self.y_minor_step.set('')
        self.assign_y_ticks()

    def figure_kind_changed(self, *_args):
        scene_id = self.kind.get()
        if scene_id not in self.minor_grids:
            return
        if scene_id != self.active_scene:
            self.scene_columns[self.active_scene] = self.columns.get()
            self.active_scene = scene_id
        self.groups = self.scene_groups.get(scene_id, self.scene_groups['custom'])
        self.signal_units = self.scene_signal_units[scene_id]
        self.columns.set(str(self.scene_columns.get(scene_id, 1)))
        self.minor_grid.set(self.minor_grids[scene_id])
        self.columns_entry.configure(state='normal' if self.scene_kind() in ('custom', 'configured') else 'disabled')
        self.update_scenario_choices()
        if hasattr(self, 'group_list'):
            self.refresh_groups(switch_to_custom=False)

    def minor_grid_changed(self):
        self.minor_grids[self.kind.get()] = self.minor_grid.get()
        self.units_changed()

    def torque_mode_changed(self, _event=None):
        self.units_changed()

    def style_changed(self, _event=None):
        if self.style.get() == 'latex_style' and self.palette.get() == 'screen':
            self.palette.set('okabe_ito')
        self.units_changed()

    def assign_transform(self):
        selected = self.trace_tree.selection()
        if not selected or selected[0] not in self.trace_records:
            return
        change = {'scale': self.edit_scale.get().strip().replace(',', '.'),
                  'offset': self.edit_offset.get().strip().replace(',', '.')}
        try:
            base = self.options()['base']
            scalar_expression(change['scale'], base)
            scalar_expression(change['offset'], base)
        except ValueError as error:
            self.report_error(error)
            return
        self.transform_overrides[selected[0]] = change
        self.refresh_trace_editor(selected[0])
        self.units_changed()

    def reset_transform(self):
        selected = self.trace_tree.selection()
        if selected and selected[0] in self.trace_records:
            if selected[0] in self.default_transforms:
                self.transform_overrides[selected[0]] = self.default_transforms[selected[0]].copy()
            else:
                self.transform_overrides.pop(selected[0], None)
            self.refresh_trace_editor(selected[0])
            self.units_changed()

    def units_changed(self):
        self.refresh_trace_editor()
        if self.figure is not None and not self.busy and not self.batch_active:
            self.preview()

    def unit_selection_changed(self, _event=None):
        selected = self.signals.curselection()
        if selected:
            key = self.signals.get(selected[0])
            unit = self.scenarios.get(self.kind.get(), {}).get('units', {}).get(key, 'raw')
            if self.scene_kind() == 'configured':
                panel_key = self.selected_panel_key()
                panels = self.scene_panel_metadata[self.kind.get()]
                selected_panel = int(panel_key.rsplit(':', 1)[1]) if panel_key else None
                candidates = [panels[selected_panel]] if selected_panel is not None else panels
                for panel in candidates:
                    match = next((signal for signal in panel['signals'] if signal['key'] == key), None)
                    if match is not None:
                        unit = match['unit']
                        break
            self.input_unit.set(self.signal_units.get(key, unit))

    def assign_units(self):
        unit = self.input_unit.get()
        for index in self.signals.curselection():
            self.signal_units[self.signals.get(index)] = unit
        self.refresh_groups(switch_to_custom=False)
        self.units_changed()

    def refresh_groups(self, switch_to_custom=True):
        self.group_list.delete(0, tk.END)
        for index, group in enumerate(self.groups, 1):
            names = [f'{key} [{self.signal_units.get(key, "raw")}]' for key in group]
            self.group_list.insert(tk.END, f'{index}: ' + ', '.join(names))
        if switch_to_custom and self.scene_kind() not in ('custom', 'configured'):
            self.kind.set('custom')
        self.refresh_trace_editor()

    def add_group(self):
        keys = [self.signals.get(i) for i in self.signals.curselection()]
        if keys:
            if self.scene_kind() not in ('custom', 'configured'):
                defaults = self.scenarios[self.kind.get()].get('units', {})
                target_units = self.scene_signal_units['custom']
                for key in keys:
                    unit = self.signal_units.get(key, defaults.get(key))
                    if unit is not None:
                        target_units[key] = unit
            self.groups.append(keys)
            if self.scene_kind() == 'configured':
                self.scene_panel_metadata[self.kind.get()].append({
                    'title': f'Alikuvaaja {len(self.groups)}',
                    'signals': [{'key': key, 'unit': self.signal_units.get(key, 'raw'),
                                 'scale': '1', 'offset': '0'} for key in keys]})
            self.refresh_groups()

    def replace_group(self):
        indices = self.group_list.curselection()
        keys = [self.signals.get(i) for i in self.signals.curselection()]
        if indices and keys:
            scene_id = self.kind.get() if self.scene_kind() in ('custom', 'configured') else 'custom'
            if scene_id == 'custom' and self.scene_kind() not in ('custom', 'configured'):
                defaults = self.scenarios[self.kind.get()].get('units', {})
                for key in keys:
                    unit = self.signal_units.get(key, defaults.get(key))
                    if unit is not None:
                        self.scene_signal_units['custom'][key] = unit
            prefix = f'{scene_id}:{indices[0]}:'
            self.transform_overrides = {key: value for key, value in self.transform_overrides.items()
                                        if not key.startswith(prefix)}
            self.default_transforms = {key: value for key, value in self.default_transforms.items()
                                       if not key.startswith(prefix)}
            self.axis_limits.pop(f'{scene_id}:{indices[0]}', None)
            self.axis_tick_steps.pop(f'{scene_id}:{indices[0]}', None)
            self.legend_positions.pop(f'{scene_id}:{indices[0]}', None)
            if self.scene_kind() == 'configured':
                self.scene_panel_metadata[scene_id][indices[0]]['signals'] = [
                    {'key': key, 'unit': self.signal_units.get(key, 'raw'), 'scale': '1', 'offset': '0'}
                    for key in keys]
            self.groups[indices[0]] = keys
            self.refresh_groups()

    def remove_group(self):
        scene_id = self.kind.get() if self.scene_kind() in ('custom', 'configured') else 'custom'
        removed = set(self.group_list.curselection())
        remaining = [index for index in range(len(self.groups)) if index not in removed]
        remap = {old: new for new, old in enumerate(remaining)}
        overrides = {}
        for key, value in self.transform_overrides.items():
            if not key.startswith(f'{scene_id}:'):
                overrides[key] = value
                continue
            _, panel, signal = key.split(':', 2)
            if int(panel) in remap:
                overrides[f'{scene_id}:{remap[int(panel)]}:{signal}'] = value
        self.transform_overrides = overrides
        defaults = {}
        for key, value in self.default_transforms.items():
            if not key.startswith(f'{scene_id}:'):
                defaults[key] = value
                continue
            _, panel, signal = key.split(':', 2)
            if int(panel) in remap:
                defaults[f'{scene_id}:{remap[int(panel)]}:{signal}'] = value
        self.default_transforms = defaults
        limits = {}
        for key, bounds in self.axis_limits.items():
            if not key.startswith(f'{scene_id}:'):
                limits[key] = bounds
            elif int(key.split(':', 1)[1]) in remap:
                limits[f'{scene_id}:{remap[int(key.split(":", 1)[1])]}'] = bounds
        self.axis_limits = limits
        ticks = {}
        for key, steps in self.axis_tick_steps.items():
            if not key.startswith(f'{scene_id}:'):
                ticks[key] = steps
            elif int(key.split(':', 1)[1]) in remap:
                ticks[f'{scene_id}:{remap[int(key.split(":", 1)[1])]}'] = steps
        self.axis_tick_steps = ticks
        positions = {}
        for key, position in self.legend_positions.items():
            if not key.startswith(f'{scene_id}:'):
                positions[key] = position
            elif int(key.split(':', 1)[1]) in remap:
                positions[f'{scene_id}:{remap[int(key.split(":", 1)[1])]}'] = position
        self.legend_positions = positions
        self.groups = [self.groups[index] for index in remaining]
        self.scene_groups[scene_id] = self.groups
        if scene_id in self.scene_panel_metadata:
            self.scene_panel_metadata[scene_id] = [self.scene_panel_metadata[scene_id][index]
                                                   for index in remaining]
        self.refresh_groups()

    def options(self):
        def number(key, optional=False):
            value = self.parameters[key].get().strip().replace(',', '.')
            if optional and not value:
                return None
            result = float(value)
            if not np.isfinite(result):
                raise ValueError(f'{key}: arvon on oltava äärellinen')
            return result
        motor = {key: number(key) for key in ('U_nom', 'I_nom', 'f_nom', 'P_nom', 'tau_nom', 'n_p')}
        if any(value <= 0 for value in motor.values()) or not motor['n_p'].is_integer():
            raise ValueError('Moottoriarvojen on oltava positiivisia; n_p on kokonaisluku')
        motor['n_p'] = int(motor['n_p'])
        torque_mode = self.torque_mode.get()
        if torque_mode not in ('per_unit', 'rated'):
            raise ValueError('Valitse momentin esitystapa')
        pre, post = number('pre'), number('post')
        duration, step = number('duration', True), number('x_step', True)
        minor_step = number('x_minor_step', True)
        if (pre < 0 or post < 0 or (duration is not None and duration <= 0)
                or (step is not None and step <= 0) or (minor_step is not None and minor_step <= 0)):
            raise ValueError('Marginaalit ≥ 0; pituus ja aikajaot > 0')
        scene_id = self.kind.get()
        kind = self.scene_kind()
        generic = kind in ('custom', 'configured')
        if generic and not self.groups:
            raise ValueError('Lisää vähintään yksi alikuvaaja')
        columns = 1
        if generic:
            try:
                columns = int(self.columns.get().strip())
            except ValueError as error:
                raise ValueError('Palstojen määrän on oltava kokonaisluku 1–8') from error
            if not 1 <= columns <= 8:
                raise ValueError('Palstojen määrän on oltava kokonaisluku 1–8')
        base = BaseValues(**motor)
        traces = self.current_traces()
        transforms = {}
        for trace in traces:
            if trace['id'] in self.transform_overrides:
                change = self.transform_overrides[trace['id']].copy()
                scalar_expression(change['scale'], base)
                scalar_expression(change['offset'], base)
                target_id = trace['id']
                if target_id.startswith(f'{scene_id}:'):
                    target_id = f'{"custom" if generic else kind}:{target_id.split(":", 1)[1]}'
                transforms[target_id] = change
        if self.style.get() not in ('screen', 'latex_style') or self.palette.get() not in PALETTES:
            raise ValueError('Valitse tunnettu tyyli ja paletti')
        detector = self.detector.get()
        if detector not in ('references', 'spikes'):
            raise ValueError('Valitse references tai spikes')
        window = {'pre_transient': pre, 'post_transient': post,
                  'duration': duration, 't_offset': number('offset', True), 'detector': detector}
        if detector == 'spikes':
            signal = self.spike_signal.get().strip()
            if not signal:
                raise ValueError('Valitse mittauspiikkien signaali')
            threshold = float(self.spike_threshold.get().strip().replace(',', '.'))
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError('Piikkikynnyksen on oltava positiivinen ja äärellinen')
            window.update(spike_signal=signal, spike_threshold=threshold)
        panels = {trace['panel'] for trace in traces}
        y_limits = {panel: self.axis_limits[f'{scene_id}:{panel}'] for panel in panels
                    if f'{scene_id}:{panel}' in self.axis_limits}
        y_steps = {panel: self.axis_tick_steps[f'{scene_id}:{panel}'] for panel in panels
                   if f'{scene_id}:{panel}' in self.axis_tick_steps}
        added_traces = [item.copy() for item in self.preset_added.get(scene_id, ())]
        native = subplot_traces(kind, self.groups) if not generic else []
        disabled_traces = {trace['id'] for trace in native
                           if self._native_ui_id(scene_id, trace['id']) in self.preset_disabled}
        scene = self.scenarios.get(scene_id, {})
        units = {**scene.get('units', {}), **self.signal_units}
        panel_titles = []
        if kind == 'configured':
            metadata = self.scene_panel_metadata[scene_id]
            panel_titles = [panel['title'] for panel in metadata]
            for panel_index, panel in enumerate(metadata):
                for signal in panel['signals']:
                    units[f'custom:{panel_index}:{signal["key"]}'] = self.signal_units.get(
                        signal['key'], signal['unit'])
        return {'kind': kind, 'scene_id': scene_id, 'base': base, 'window': window,
                'suffix': scene.get('suffix', '_custom'), 'columns': columns,
                'panel_titles': panel_titles,
                'added_traces': added_traces, 'disabled_traces': disabled_traces,
                'x_step': step, 'groups': [group.copy() for group in self.groups],
                'per_unit': self.per_unit.get(), 'torque_mode': torque_mode,
                'units': units,
                'transforms': transforms, 'y_limits': y_limits,
                'minor_grid': self.minor_grid.get(), 'x_minor_step': minor_step, 'y_steps': y_steps,
                'legend_positions': {panel: self.legend_positions[f'{self.kind.get()}:{panel}']
                                     for panel in panels if f'{self.kind.get()}:{panel}' in self.legend_positions},
                'style': self.style.get(), 'palette': self.palette.get()}

    def set_busy(self, value):
        self.busy = value
        for button in (self.preview_button, self.export_button):
            button.configure(state=tk.DISABLED if value or self.batch_active else tk.NORMAL)

    def start_job(self, work, success, failure=None):
        if self.busy or self.closing:
            return
        self.set_busy(True)
        future = self.executor.submit(work)

        def poll():
            if self.closing:
                return
            if not future.done():
                self.root.after(50, poll)
                return
            self.set_busy(False)
            try:
                payload = future.result()
                success(payload)
            except Exception as error:  # noqa: BLE001 - keep the GUI usable after file/render failures
                if failure:
                    failure(error)
                else:
                    self.report_error(error)
        self.root.after(50, poll)

    def report_error(self, error):
        self.status.set(f'Virhe: {error}')
        messagebox.showerror('Plottaus epäonnistui', str(error), parent=self.root)

    def preview(self):
        path = self.active_path()
        if not path:
            return
        try:
            options = self.options()
        except ValueError as error:
            self.report_error(error)
            return
        self.status.set(f'Luetaan {path.name}…')

        def loaded(payload):
            fig, _ = render_request(payload, options)
            self.show_figure(fig, kind=options['scene_id'])
            self.preview_source.set(f'Esikatselu: {path}')
            self.status.set(f'{path.name}: {options["kind"]}. Raahaa legenda; zoom/siirto pois raahauksen ajaksi.')
        self.start_job(lambda: load_plot_request(path, options), loaded)

    def reset_legends(self):
        prefix = f'{self.kind.get()}:'
        self.legend_positions = {key: value for key, value in self.legend_positions.items()
                                 if not key.startswith(prefix)}
        self.units_changed()

    def show_figure(self, fig, kind=None):
        kind = self.kind.get() if kind is None else kind
        canvas = toolbar = None
        try:
            downsample(fig)
            canvas = FigureCanvasTkAgg(fig, master=self.plot_area)
            toolbar = NavigationToolbar2Tk(canvas, self.plot_area, pack_toolbar=False)
            canvas.draw()
        except Exception:
            if toolbar is not None:
                toolbar.destroy()
            if canvas is not None:
                canvas.get_tk_widget().destroy()
            plt.close(fig)
            raise
        if self.toolbar:
            self.toolbar.destroy()
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.figure:
            plt.close(self.figure)
        self.figure = fig
        self.canvas = canvas
        self.toolbar = toolbar
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        legends = {}
        for panel, axis in enumerate(fig.axes):
            legend = axis.get_legend()
            if legend is not None:
                legend.set_draggable(True, use_blit=False, update='loc')
                legend.set_picker(lambda artist, mouse: (
                    not bool(self.toolbar.mode) and artist.contains(mouse)[0], {}))
                legends[legend] = (panel, axis)
        picked = {'legend': None}

        def pick(event):
            if not self.toolbar.mode and event.artist in legends:
                picked['legend'] = event.artist

        def release(_event):
            legend = picked.pop('legend', None)
            if legend is None or self.toolbar.mode:
                return
            panel, axis = legends[legend]
            bounds = legend.get_window_extent(self.canvas.get_renderer())
            x, y = axis.transAxes.inverted().transform((bounds.x0, bounds.y0))
            if np.isfinite(x) and np.isfinite(y):
                self.legend_positions[f'{kind}:{panel}'] = (float(x), float(y))

        canvas.mpl_connect('pick_event', pick)
        canvas.mpl_connect('button_release_event', release)
        canvas.draw_idle()

    def export(self):
        paths = self.selected_paths()
        if not paths:
            return
        try:
            options = self.options()
        except ValueError as error:
            self.report_error(error)
            return
        folder = self.output.get().strip()
        targets = [((Path(folder).expanduser().resolve() if folder else path.parent) /
                    f'{path.stem}{options["suffix"]}.pdf') for path in paths]
        if len(set(targets)) != len(targets):
            self.report_error(ValueError('Saman nimiset mittaukset: käytä lähdekansioita tai nimeä tiedostot eri tavoin.'))
            return
        existing = sum(target.exists() for target in targets)
        if existing and not messagebox.askyesno('Korvataanko PDF:t?',
                                               f'{existing} tiedostoa on jo olemassa. Korvataanko?', parent=self.root):
            return
        self.cancelled = False
        self.batch_active = True
        self.export_next(list(zip(paths, targets)), options, [], [])

    def export_next(self, pending, options, saved, errors):
        if not pending or self.cancelled or self.closing:
            self.batch_active = False
            self.set_busy(False)
            self.status.set(f'Eräajo: {len(saved)} tallennettu, {len(errors)} virhettä'
                            + ('; keskeytetty.' if self.cancelled else '.'))
            if errors and not self.closing:
                messagebox.showwarning('Eräajon virheet', '\n'.join(errors), parent=self.root)
            return
        path, target = pending.pop(0)
        self.status.set(f'Tallennetaan {path.name}… (jäljellä {len(pending)})')

        def update_row(text):
            for item, source in self.path_ids.items():
                if source == path:
                    self.files.set(item, 'state', text)

        def next_file():
            self.root.after(1, lambda: self.export_next(pending, options, saved, errors))

        def loaded(payload):
            fig = None
            temporary = None
            try:
                fig, _ = render_request(payload, options)
                if options['kind'] in ('injection-overview', 'presentation'):
                    downsample(fig)
                target.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=target.parent, suffix='.pdf', delete=False) as handle:
                    temporary = Path(handle.name)
                fig.savefig(temporary, dpi=300)
                temporary.replace(target)
                saved.append(target)
                update_row('Tallennettu')
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                if fig is not None:
                    plt.close(fig)
            next_file()

        def failed(error):
            errors.append(f'{path.name}: {error}')
            update_row('Virhe')
            next_file()
        self.start_job(lambda: load_plot_request(path, options), loaded, failed)

    def cancel(self):
        self.cancelled = True
        if self.busy:
            self.status.set('Keskeytetään nykyisen tiedoston jälkeen…')

    def close(self):
        self.closing = True
        if getattr(self, '_config_timer', None) is not None:
            self.root.after_cancel(self._config_timer)
            self._config_timer = None
        self.executor.shutdown(wait=False, cancel_futures=True)
        if self.figure:
            plt.close(self.figure)
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        app = PlotterApp(root)
    except ValueError as error:
        root.withdraw()
        messagebox.showerror('Asetustiedoston virhe', str(error), parent=root)
        root.destroy()
        return
    app.insert_paths(sys.argv[1:])
    root.mainloop()


if __name__ == '__main__':
    main()
