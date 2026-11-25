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

The signal-map profile links the quantities of the built-in figures to the
recording keys of a dSPACE model. Its status line reports which quantities of
the current figure the active file lacks. The link editor lists the active
file's signals per quantity and saves the profile to signal_maps.json, which is
reloaded like the other configuration files. Unchanged rows keep their saved
alternative keys. Custom and configured figures use recording keys directly,
or @role quantities (listed first in the signal list) that follow the signal
map and default to the role's recorded unit.

The interface language (Finnish or English) is switched at the top of the
sidebar; the choice is stored per user in lab_plotter_settings.json.

references detects changes of the mapped speed and load-torque references
(default wm_ref, tauLref/TL_ref); spikes detects deviations beyond a
positive raw-unit threshold from a selected signal's whole-recording median.
Each detector uses original data on its own raster, before plotting transforms;
without events the full recording is used. This is not peak-prominence or
settling-time detection, and a missing selected spike signal fails that file.
"""
import copy
import re
import sys
import tempfile
import tkinter as tk
import tkinter.font as tkfont
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from uuid import uuid4

import matplotlib
import numpy as np

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import to_rgba
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator, ScalarFormatter

from lab_plotter_config import (
    MOTOR_PATH,
    SCENARIO_PATH,
    SIGNAL_MAP_PATH,
    SIGNAL_ROLES,
    load_motor_profiles,
    load_scenarios,
    load_signal_maps,
    save_signal_map,
    signal_role,
)
from lab_plotter_i18n import LANGUAGES, SETTINGS_PATH, load_language, save_language, translate
from lab_plotting import LabDataMat, signal_source, transient_window
from lab_plotter_core import (
    FIGURE_TYPES,
    REFERENCE_ROLES,
    SIGNAL_UNITS,
    VECTOR_ROLES,
    BaseValues,
    detection_options,
    load_measurement,
    plot_measurement,
    plot_signals,
    resolve_signal_map,
    role_key,
    role_unit,
    scalar_expression,
    signal_matches,
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

    def __contains__(self, key):
        return key in self.infos or key in self.signals

    def __getitem__(self, key):
        return self.signals[key]

    def time(self, key):
        return self.times[key]


def load_plot_request(path, options):
    """Read files off the UI thread; return detached data, never open handles."""
    if options['kind'] not in ('custom', 'configured'):
        extra = [trace['key'] for trace in options.get('added_traces', ())]
        return load_measurement(path, options['base'], x_step=options['x_step'],
                                extra_signals=extra, signal_map=options.get('signal_map'),
                                **options['window'])
    signal_map = options.get('signal_map')
    with LabDataMat(path) as data:
        origin, limits = transient_window(data, **detection_options(data, signal_map, **options['window']))
        keys = dict.fromkeys(key for group in options['groups'] for key in group)
        # @role keys stay as group keys; their infos name the linked recording signals.
        infos = {key: data.signal_info(role_key(data, key, signal_map)) for key in keys}
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


_INDEXED = re.compile(r'\[\d+\]$')


def key_source(catalog, key):
    """Block path a key (or the first element of a vector key) is recorded from."""
    try:
        return signal_source(catalog.signal_info(key)[0])
    except (KeyError, IndexError):
        return ''


def choice_text(catalog, key):
    source = key_source(catalog, key)
    return f'{key}  ·  {source}' if source and source != key else key


def format_range(bounds):
    if bounds is None:
        return '–'
    low, high = bounds
    return f'{low:.4g}' if low == high else f'{low:.4g} … {high:.4g}'


def signal_listing(catalog, name, tr):
    """Text table of every signal: index, key, finite range, samples and block path."""
    several = len({s['raster'] for s in catalog.signals}) > 1 and catalog.raster_names
    rows = []
    for s in catalog.signals:
        samples = str(s['samples'])
        if several:
            samples += f" [{catalog.raster_names[s['raster']]}]"
        rows.append((str(s['index']), s['key'], format_range(catalog.ranges.get(s['key'])),
                     samples, signal_source(s) or '—'))
    header = ('#', tr('Key'), tr('Range'), tr('Samples'), tr('Source (block path)'))
    widths = [max(len(row[column]) for row in (header, *rows)) for column in range(4)]
    lines = [tr('Recording {name}: {count} signals', name=name, count=len(rows)), '']
    for row in (header, *rows):
        cells = [row[0].rjust(widths[0])] + [cell.ljust(width) for cell, width in zip(row[1:4], widths[1:])]
        lines.append('  '.join(cells + [row[4]]))
    return '\n'.join(lines) + '\n'


def role_usage(tr, scenarios=None):
    """Short description of the figures, scenarios and detection reading each role."""
    usage = {role: [] for role, *_ in SIGNAL_ROLES}
    for kind in FIGURE_TYPES:
        for trace in subplot_traces(kind):
            for role in trace['roles']:
                if kind not in usage[role]:
                    usage[role].append(kind)
    extra = {role: [] for role in usage}
    for scene_id, scene in (scenarios or {}).items():
        keys = [trace['key'] for trace in scene.get('added_traces', ())]
        keys += [signal['key'] for panel in scene.get('panels', ()) for signal in panel['signals']]
        for role in dict.fromkeys(signal_role(key) for key in keys):
            if role in extra:
                extra[role].append(scene_id)
    result = {}
    for role, kinds in usage.items():
        names = [tr('all figures')] if len(kinds) == len(FIGURE_TYPES) else kinds
        text = ', '.join(names + extra[role]) or tr('not plotted')
        result[role] = tr('{figures} + transients', figures=text) if role in REFERENCE_ROLES else text
    return result


class SignalMapDialog:
    """Link each quantity of the built-in figures to a key of the active recording.

    A row left at its initial value keeps the profile's saved alternative keys;
    an edited row stores exactly the chosen key, and an empty row unlinks it.
    """

    def __init__(self, app):
        self.app = app
        self.map_id = app.signal_map_id.get()
        self.profile = copy.deepcopy(app.signal_maps[self.map_id])
        self.catalog = None
        self.window = tk.Toplevel(app.root)
        tr = app.tr
        self.window.title(tr('Signal links — {name} [{id}]', name=self.profile['name'], id=self.map_id))
        self.window.transient(app.root)
        self.header = tk.StringVar()
        self.message = tk.StringVar(value=tr('Choose a recording signal for each quantity. '
                                             'Empty = not linked (the quantity is plotted empty).'))
        ttk.Label(self.window, textvariable=self.header, padding=6, wraplength=1000).pack(anchor=tk.W)
        grid = ttk.Frame(self.window, padding=(6, 0))
        grid.pack(fill=tk.BOTH, expand=True)
        for column, text in enumerate((tr('Quantity'), tr('Role'), tr('Used by'),
                                       tr('Recording signal'), tr('Status'))):
            ttk.Label(grid, text=text, font='TkHeadingFont').grid(row=0, column=column, sticky='w', padx=3)
        usage = role_usage(tr, app.scenarios)
        self.rows = {}
        for row, (role, label, _, vector, _) in enumerate(SIGNAL_ROLES, 1):
            variable, status = tk.StringVar(), tk.StringVar()
            ttk.Label(grid, text=tr(label)).grid(row=row, column=0, sticky='w', padx=3)
            ttk.Label(grid, text=role).grid(row=row, column=1, sticky='w', padx=3)
            ttk.Label(grid, text=usage[role]).grid(row=row, column=2, sticky='w', padx=3)
            box = ttk.Combobox(grid, textvariable=variable, width=44, style='SignalMap.TCombobox')
            box.grid(row=row, column=3, sticky='ew', padx=3, pady=1)
            box.bind('<<ComboboxSelected>>', lambda _event, role=role: self.choice_selected(role))
            ttk.Label(grid, textvariable=status, wraplength=330).grid(
                row=row, column=4, sticky='w', padx=3)
            self.rows[role] = {'variable': variable, 'status': status, 'box': box,
                               'initial': '', 'vector': vector, 'choices': [], 'display': {}}
            variable.trace_add('write', lambda *_args, role=role: self.update_status(role))
        grid.columnconfigure(3, weight=1)
        ttk.Label(self.window, textvariable=self.message, padding=6, wraplength=1000).pack(anchor=tk.W)
        buttons = ttk.Frame(self.window, padding=6)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text=tr('Save to profile'), command=self.save).pack(side=tk.LEFT)
        ttk.Button(buttons, text=tr('Save as new profile…'), command=self.save_new).pack(side=tk.LEFT)
        ttk.Button(buttons, text=tr('List all signals…'), command=app.show_signal_listing).pack(side=tk.LEFT)
        ttk.Button(buttons, text=tr('Close'), command=self.close).pack(side=tk.RIGHT)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.set_catalog(app.signal_catalog, app.signal_catalog_name)

    def set_catalog(self, catalog, name):
        """Offer a recording's keys; unedited rows show the key the profile resolves to."""
        self.catalog = catalog
        keys = catalog.keys() if catalog is not None else []
        vectors = [key for key in dict.fromkeys(_INDEXED.sub('', key) for key in keys if _INDEXED.search(key))
                   if signal_matches(catalog, key, True)]
        resolved = resolve_signal_map(catalog, self.profile['signals']) if catalog is not None else {}
        for role, row in self.rows.items():
            row['choices'] = vectors if row['vector'] else list(keys)
            # The list shows where each signal comes from; choosing an entry enters its key.
            row['display'] = {choice_text(catalog, key): key for key in row['choices']}
            row['box'].configure(values=['', *row['display']])
            if row['variable'].get() == row['initial']:
                saved = self.profile['signals'][role]
                row['initial'] = resolved.get(role) or (saved[0] if saved else '')
                row['variable'].set(row['initial'])
            self.update_status(role)
        texts = [text for row in self.rows.values() for text in row['display']]
        if texts:
            # ttk sizes the drop-down to the combobox; widen it to fit the block paths.
            longest = max(tkfont.nametofont('TkDefaultFont').measure(text) for text in texts)
            extra = max(0, longest + 40 - next(iter(self.rows.values()))['box'].winfo_reqwidth())
            ttk.Style(self.window).configure('SignalMap.TCombobox', postoffset=(0, 0, extra, 0))
        tr = self.app.tr
        self.header.set(tr('Recording: {name} ({count} signals). The list shows the block path of each '
                           'signal; vector quantities list 2–3-element vectors without [index].',
                           name=name, count=len(keys)) if catalog is not None else
                        tr('No measurement read: select a file in the main window to see its signals '
                           'and the link status here.'))

    def choice_selected(self, role):
        row = self.rows[role]
        key = row['display'].get(row['variable'].get())
        if key is not None:
            row['variable'].set(key)

    def update_status(self, role):
        tr = self.app.tr
        row = self.rows[role]
        value = row['variable'].get().strip()
        if not value:
            text = tr('— not linked')
        elif self.catalog is None:
            text = tr('no recording to check')
        elif signal_matches(self.catalog, value, row['vector']):
            source = key_source(self.catalog, value)
            text = tr('✓ found') + (f' · {source}' if source and source != value else '')
        elif value in self.catalog:
            text = tr('✗ wrong shape: {expected}',
                      expected=tr('2–3-element vector') if row['vector'] else tr('single signal'))
        else:
            # Several block paths end with this name: list them instead of guessing.
            matches = [key for key in row['choices'] if key.endswith('/' + value)]
            text = (tr('✗ ambiguous, choose: {keys}', keys=' | '.join(matches)) if matches
                    else tr('✗ not in the recording'))
        saved = self.profile['signals'][role]
        if value == row['initial'] and len(saved) > 1:
            text += ' · ' + tr('alternatives {keys}', keys=' / '.join(saved))
        row['status'].set(text)

    def collect(self):
        signals = {}
        for role, row in self.rows.items():
            value = row['variable'].get().strip()
            signals[role] = (list(self.profile['signals'][role]) if value == row['initial']
                             else [value] if value else [])
        return signals

    def save(self):
        name = self.profile['name']
        if messagebox.askyesno(self.app.tr('Save signal map'),
                               self.app.tr('Replace the links of profile "{name}" in {file}?',
                                           name=name, file=SIGNAL_MAP_PATH.name), parent=self.window):
            self.write(self.map_id, name)

    def save_new(self):
        name = simpledialog.askstring(self.app.tr('New signal map'),
                                      self.app.tr('Profile name (e.g. the name of the dSPACE model):'),
                                      parent=self.window)
        if name is None or not name.strip():
            return
        base = re.sub(r'[^A-Za-z0-9_-]+', '-', name.strip()).strip('-') or 'map'
        map_id, number = base, 2
        while map_id in self.app.signal_maps:
            map_id, number = f'{base}-{number}', number + 1
        self.write(map_id, name.strip())

    def write(self, map_id, name):
        try:
            save_signal_map({'id': map_id, 'name': name, 'signals': self.collect()}, SIGNAL_MAP_PATH)
        except (OSError, ValueError) as error:
            messagebox.showerror(self.app.tr('Saving failed'), str(error), parent=self.window)
            return
        self.close()
        self.app.signal_map_saved(map_id)

    def close(self):
        if self.window.winfo_exists():
            self.window.destroy()
        if self.app.map_dialog is self:
            self.app.map_dialog = None


class PlotterApp:
    def __init__(self, root, language=None):
        self.root = root
        self.language = language if language in LANGUAGES else load_language(SETTINGS_PATH)
        self._translations = []
        self.file_states = {}
        self.preview_path = None
        root.title(self.tr('Lab plotter — measurement browser'))
        root.geometry('1400x900')
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.busy = False
        self.batch_active = False
        self.cancelled = False
        self.closing = False
        self.figure = self.canvas = self.toolbar = None
        self.scenarios = load_scenarios()
        self.motor_profiles = load_motor_profiles()
        self.signal_maps = load_signal_maps()
        self.signal_map_id = tk.StringVar(value=next(iter(self.signal_maps)))
        self.signal_map_choice = tk.StringVar()
        self.signal_map_status = tk.StringVar()
        self.signal_catalog = None
        self.signal_catalog_name = ''
        self.signal_catalog_path = None
        self.listing_window = None
        self.map_dialog = None
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
        self.status = tk.StringVar(value=self.tr('Add measurement files or a folder.'))
        self.preview_source = tk.StringVar(value=self.tr('Preview: no figure'))
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

    def tr(self, text, **values):
        """Text in the interface language; values fill {name} fields."""
        return translate(text, self.language, **values)

    def _t(self, widget, text, option='text'):
        """Give a widget a translated text that follows language changes."""
        self._translations.append((widget, option, text))
        widget.configure(**{option: self.tr(text)})
        return widget

    def _heading(self, tree, column, text):
        self._translations.append((tree, ('heading', column), text))
        tree.heading(column, text=self.tr(text))

    def choose_language(self, _event=None):
        language = next((code for code, name in LANGUAGES.items()
                         if name == self.language_choice.get()), None)
        if language is not None and language != self.language:
            self.set_language(language)
            save_language(language, SETTINGS_PATH)

    def set_language(self, language):
        """Translate the open window in place; an open link editor is closed."""
        self.language = language
        self.language_choice.set(LANGUAGES[language])
        self.root.title(self.tr('Lab plotter — measurement browser'))
        for widget, option, text in self._translations:
            if isinstance(option, tuple):
                widget.heading(option[1], text=self.tr(text))
            else:
                widget.configure(**{option: self.tr(text)})
        for item, state in self.file_states.items():
            self.files.set(item, 'state', self.tr(state))
        self.render_preview_source()
        if self.map_dialog is not None:
            self.map_dialog.close()
        if self.listing_window is not None and self.listing_window.winfo_exists():
            self.listing_window.destroy()
        self.update_scenario_choices()
        self.filter_signals()
        self.refresh_groups(switch_to_custom=False)
        self.status.set(self.tr('Language changed.'))

    def render_preview_source(self):
        self.preview_source.set(self.tr('Preview: {path}', path=self.preview_path)
                                if self.preview_path is not None else self.tr('Preview: no figure'))

    def set_file_state(self, item, state):
        self.file_states[item] = state
        self.files.set(item, 'state', self.tr(state))

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
        self.scenario_choices[f'{self.tr("Free figure")} [custom]'] = 'custom'
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

    def current_signal_map(self):
        return self.signal_maps[self.signal_map_id.get()]['signals']

    def update_signal_map_choices(self):
        self.signal_map_choices = {f'{entry["name"]} [{map_id}]': map_id
                                   for map_id, entry in self.signal_maps.items()}
        self.signal_map_selector.configure(values=tuple(self.signal_map_choices))
        for label, map_id in self.signal_map_choices.items():
            if map_id == self.signal_map_id.get():
                self.signal_map_choice.set(label)
                break

    def choose_signal_map(self, _event=None):
        map_id = self.signal_map_choices.get(self.signal_map_choice.get())
        if map_id is not None:
            self.signal_map_id.set(map_id)
            self.units_changed()

    def needed_roles(self):
        """Mapped quantities that the current figure and detector actually read."""
        roles = {}
        for trace in self.current_traces():
            if trace['id'] not in self.preset_disabled:
                roles.update(dict.fromkeys(trace.get('roles', ())))
        if self.detector.get() == 'references':
            roles.update(dict.fromkeys(REFERENCE_ROLES))
        order = [role for role, *_ in SIGNAL_ROLES]
        return sorted(roles, key=order.index)

    def refresh_signal_map_status(self):
        if self.signal_catalog is None:
            self.signal_map_status.set(self.tr('Select a measurement: the status shows whether the '
                                               'quantities of the figure are in the recording.'))
            return
        roles = self.needed_roles()
        labels = {role: self.tr(label) for role, label, *_ in SIGNAL_ROLES}

        def missing(signal_map):
            resolved = resolve_signal_map(self.signal_catalog, signal_map)
            return [role for role in roles if resolved[role] is None]

        mapping = self.current_signal_map()
        absent = missing(mapping)
        parts = [self.tr('{name}: {found}/{total} quantities of the figure found.',
                         name=self.signal_catalog_name, found=len(roles) - len(absent), total=len(roles))]
        if absent:
            parts.append(self.tr('Missing: {items}.', items='; '.join(
                f'{labels[role]} ({" / ".join(mapping[role]) or self.tr("not linked")})' for role in absent)))
            better = min(((len(missing(entry['signals'])), entry['name'])
                          for map_id, entry in self.signal_maps.items()
                          if map_id != self.signal_map_id.get()), default=None)
            if better is not None and better[0] < len(absent):
                parts.append(self.tr('Profile "{name}" matches better ({count} missing).',
                                     name=better[1], count=better[0]))
        if self.scene_kind() in ('custom', 'configured'):
            parts.append(self.tr('Recording keys are used directly; @quantities follow the signal map.'))
        self.signal_map_status.set(' '.join(parts))

    def show_signal_listing(self):
        """Window listing every signal of the active recording with its range and block path."""
        if self.signal_catalog is None:
            self.status.set(self.tr('Select a measurement first.'))
            return
        if self.listing_window is not None and self.listing_window.winfo_exists():
            self.listing_window.destroy()
        name = self.signal_catalog_name
        text = signal_listing(self.signal_catalog, name, self.tr)
        window = self.listing_window = tk.Toplevel(self.root)
        window.title(self.tr('Signals — {name}', name=name))
        frame = ttk.Frame(window)
        frame.pack(fill=tk.BOTH, expand=True)
        widget = tk.Text(frame, wrap='none', font='TkFixedFont', width=140, height=40)
        vertical = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=widget.yview)
        horizontal = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=widget.xview)
        widget.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        widget.grid(row=0, column=0, sticky='nsew')
        vertical.grid(row=0, column=1, sticky='ns')
        horizontal.grid(row=1, column=0, sticky='ew')
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        widget.insert('1.0', text)
        widget.configure(state='disabled')  # read-only, but text can still be selected and copied
        buttons = ttk.Frame(window, padding=6)
        buttons.pack(fill=tk.X)
        ttk.Button(buttons, text=self.tr('Save as text…'),
                   command=lambda: self.save_signal_listing(text, window)).pack(side=tk.LEFT)
        ttk.Button(buttons, text=self.tr('Close'), command=window.destroy).pack(side=tk.RIGHT)

    def save_signal_listing(self, text, parent):
        source = self.signal_catalog_path
        target = filedialog.asksaveasfilename(
            parent=parent, defaultextension='.txt', filetypes=[(self.tr('Text files'), '*.txt')],
            initialdir=str(source.parent) if source else None,
            initialfile=f'{Path(self.signal_catalog_name).stem}_signals.txt')
        if not target:
            return
        try:
            Path(target).write_text(text, encoding='utf-8')
        except OSError as error:
            messagebox.showerror(self.tr('Saving failed'), str(error), parent=parent)
            return
        self.status.set(self.tr('Signal list saved: {path}', path=target))

    def edit_signal_map(self):
        if self.map_dialog is not None and self.map_dialog.window.winfo_exists():
            self.map_dialog.window.lift()
            return
        self.map_dialog = SignalMapDialog(self)

    def signal_map_saved(self, map_id):
        """Use a profile just written by this application without a full reload."""
        self.signal_maps = load_signal_maps()
        if self._config_stamps is not None:
            self._config_stamps = (*self._config_stamps[:2], self.config_stamps()[2])
        self.signal_map_id.set(map_id)
        self.update_signal_map_choices()
        self.status.set(self.tr('Signal map "{name}" saved: {file}.',
                                name=self.signal_maps[map_id]['name'], file=SIGNAL_MAP_PATH.name))
        self.units_changed()

    @staticmethod
    def config_stamps():
        def stamp(path):
            try:
                stat = path.stat()
                return stat.st_mtime_ns, stat.st_size
            except OSError:
                return None
        return stamp(SCENARIO_PATH), stamp(MOTOR_PATH), stamp(SIGNAL_MAP_PATH)

    def poll_configuration(self):
        if self.closing:
            return
        stamps = self.config_stamps()
        if not self.busy and not self.batch_active and stamps != self._config_stamps and stamps != self._failed_config_stamps:
            self.reload_configuration(automatic=True)
        self._config_timer = self.root.after(1200, self.poll_configuration)

    def reload_configuration(self, automatic=False):
        if self.busy or self.batch_active:
            self.status.set(self.tr('Settings will be reloaded after the running job.'))
            return
        stamps = self.config_stamps()
        try:
            scenarios = load_scenarios()
            profiles = load_motor_profiles()
            signal_maps = load_signal_maps()
        except ValueError as error:
            self._failed_config_stamps = stamps
            self.status.set(self.tr('Configuration file error; previous settings kept: {error}', error=error))
            if not automatic:
                messagebox.showerror(self.tr('Configuration file error'), str(error), parent=self.root)
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
        old_map = self.signal_maps.get(self.signal_map_id.get())
        self.signal_maps = signal_maps
        if self.signal_map_id.get() not in signal_maps:
            self.signal_map_id.set(next(iter(signal_maps)))
        self.update_signal_map_choices()
        map_changed = old_map != signal_maps[self.signal_map_id.get()]
        self._config_stamps = stamps
        self._failed_config_stamps = None
        self.refresh_groups(switch_to_custom=False)
        parts = [self.tr('Settings updated.')]
        if profile_changed and dirty:
            parts.append(self.tr('Edited motor values were kept; select the profile again to update them.'))
        if changed:
            parts.append(self.tr('Session edits of changed scenarios were reset.'))
        self.status.set(' '.join(parts))
        if (changed or map_changed or (profile_changed and not dirty)) and self.figure is not None:
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
        language_row = ttk.Frame(sidebar)
        language_row.pack(fill=tk.X, pady=(0, 4))
        ttk.Label(language_row, text='Kieli / Language').pack(side=tk.LEFT)
        self.language_choice = tk.StringVar(value=LANGUAGES[self.language])
        language_box = ttk.Combobox(language_row, textvariable=self.language_choice,
                                    values=tuple(LANGUAGES.values()), state='readonly', width=10)
        language_box.pack(side=tk.LEFT, padx=4)
        language_box.bind('<<ComboboxSelected>>', self.choose_language)
        buttons = ttk.Frame(sidebar)
        buttons.pack(fill=tk.X)
        for label, command in [('Files…', self.add_files), ('Folder…', self.add_folder),
                               ('Remove', self.remove_files)]:
            self._t(ttk.Button(buttons, command=command), label).pack(side=tk.LEFT)
        self._t(ttk.Label(sidebar),
                'Ctrl/Shift: multiple selection; Ctrl+A: all; the active row is previewed').pack(anchor=tk.W)
        frame = ttk.Frame(sidebar)
        frame.pack(fill=tk.X)
        self.files = ttk.Treeview(frame, columns=('state',), height=7, selectmode='extended')
        self._heading(self.files, '#0', 'Measurement')
        self._heading(self.files, 'state', 'State')
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

        settings = self._t(ttk.LabelFrame(sidebar, padding=5), 'Figure and time window')
        settings.pack(fill=tk.X, pady=5)
        self.kind_selector = ttk.Combobox(settings, textvariable=self.scenario_choice, state='readonly')
        self.kind_selector.grid(row=0, column=0, columnspan=2, sticky='ew')
        self.kind_selector.bind('<<ComboboxSelected>>', self.choose_scenario)
        self.update_scenario_choices()
        for row, (key, label, value) in enumerate([
                ('pre', 'Before transient (s)', '.2'),
                ('post', 'After last transient (s)', '.2'),
                ('duration', 'Duration (s), empty = automatic', ''),
                ('x_step', 'X major step (s), empty = default', ''),
                ('x_minor_step', 'X minor step (s), empty = default', ''),
                ('offset', 'Time origin (s), empty = automatic', '')], 1):
            self.parameters[key] = tk.StringVar(value=value)
            self._t(ttk.Label(settings), label).grid(row=row, column=0, sticky='w')
            ttk.Entry(settings, textvariable=self.parameters[key], width=12).grid(row=row, column=1)
        self._t(ttk.Checkbutton(settings, variable=self.per_unit, command=self.units_changed),
                'Use p.u. values (all figures)').grid(row=7, column=0, columnspan=2, sticky='w')
        self._t(ttk.Checkbutton(settings, variable=self.minor_grid, command=self.minor_grid_changed),
                'Show minor grid').grid(row=8, column=0, columnspan=2, sticky='w')
        self._t(ttk.Label(settings), 'Angles stay in deg/rad.').grid(row=9, column=0, columnspan=2, sticky='w')
        self._t(ttk.Label(settings), 'Columns (custom / configured scenario)').grid(row=10, column=0, sticky='w')
        self.columns_entry = ttk.Entry(settings, textvariable=self.columns, width=12)
        self.columns_entry.grid(row=10, column=1)
        self._t(ttk.Button(settings, command=self.reload_configuration), 'Reload settings').grid(
            row=11, column=0, columnspan=2, sticky='w')
        appearance = self._t(ttk.LabelFrame(sidebar, padding=5), 'Appearance and palette')
        appearance.pack(fill=tk.X, pady=5)
        self._t(ttk.Label(appearance), 'Style').grid(row=0, column=0, sticky='w')
        style_box = ttk.Combobox(appearance, textvariable=self.style,
                                 values=('screen', 'latex_style'), state='readonly', width=18)
        style_box.grid(row=0, column=1)
        style_box.bind('<<ComboboxSelected>>', self.style_changed)
        self._t(ttk.Label(appearance), 'Palette').grid(row=1, column=0, sticky='w')
        palette_box = ttk.Combobox(appearance, textvariable=self.palette,
                                   values=tuple(PALETTES), state='readonly', width=18)
        palette_box.grid(row=1, column=1)
        palette_box.bind('<<ComboboxSelected>>', lambda _event: self.units_changed())
        self._t(ttk.Label(appearance, wraplength=430),
                'latex_style uses the legacy publication style; it needs a TeX installation and the newtx '
                'packages. The palette is chosen separately.').grid(row=2, column=0, columnspan=2, sticky='w')
        detection = self._t(ttk.LabelFrame(sidebar, padding=5), 'Transient detection')
        detection.pack(fill=tk.X, pady=5)
        detector_box = ttk.Combobox(detection, textvariable=self.detector,
                                    values=('references', 'spikes'), state='readonly', width=20)
        detector_box.grid(row=0, column=0, columnspan=2, sticky='ew')
        detector_box.bind('<<ComboboxSelected>>', lambda _event: self.units_changed())
        self._t(ttk.Label(detection), 'Spike signal').grid(row=1, column=0, sticky='w')
        self.spike_selector = ttk.Combobox(detection, textvariable=self.spike_signal,
                                           values=self.scalar_role_keys(), state='readonly', width=30)
        self.spike_selector.grid(row=1, column=1)
        self._t(ttk.Label(detection), 'Amplitude threshold (raw units)').grid(row=2, column=0, sticky='w')
        ttk.Entry(detection, textvariable=self.spike_threshold, width=12).grid(row=2, column=1)
        self._t(ttk.Label(detection, wraplength=430),
                'references: changes of the linked speed and load-torque references (default wm_ref, '
                'tauLref/TL_ref). spikes: |y − median of the whole signal| > threshold in the selected signal '
                '(also an @quantity). Detection uses the original data, not p.u. conversions. '
                'No events → whole recording.').grid(row=3, column=0, columnspan=2, sticky='w')
        motor = self._t(ttk.LabelFrame(sidebar, padding=5), 'Motor — same values for all selected files')
        motor.pack(fill=tk.X)
        self._t(ttk.Label(motor), 'Motor profile').grid(row=0, column=0, sticky='w')
        self.profile_choice = tk.StringVar()
        self.profile_selector = ttk.Combobox(motor, textvariable=self.profile_choice,
                                             state='readonly', width=23)
        self.profile_selector.grid(row=0, column=1, columnspan=2, sticky='ew')
        self.profile_selector.bind('<<ComboboxSelected>>', self.choose_profile)
        profile = self.motor_profiles[self.profile_id.get()]
        labels = [('U_nom', 'U RMS, line-to-line (V)'), ('I_nom', 'I RMS (A)'),
                  ('f_nom', 'f (Hz)'), ('P_nom', 'P (W)'),
                  ('tau_nom', 'T rated (Nm)'), ('n_p', 'Pole pairs')]
        for index, (key, label) in enumerate(labels):
            self.parameters[key] = tk.StringVar(value=str(profile[key]))
            row, col = divmod(index, 3)
            self._t(ttk.Label(motor), label).grid(row=1 + row*2, column=col)
            ttk.Entry(motor, textvariable=self.parameters[key], width=12).grid(row=2 + row*2, column=col)
        self.profile_baseline = {key: self.parameters[key].get() for key, _ in labels}
        self.update_profile_choices()
        self._t(ttk.Label(motor), 'Torque display').grid(row=5, column=0, sticky='w')
        torque_selector = ttk.Combobox(motor, textvariable=self.torque_mode,
                                      values=('per_unit', 'rated'), state='readonly', width=12)
        torque_selector.grid(row=5, column=1, sticky='w')
        torque_selector.bind('<<ComboboxSelected>>', self.torque_mode_changed)
        self._t(ttk.Label(motor, wraplength=430),
                'rated: torques / nominal torque of the motor profile; other quantities follow the p.u. '
                'choice.').grid(row=6, column=0, columnspan=3, sticky='w')
        links = self._t(ttk.LabelFrame(sidebar, padding=5), 'Signal links (dSPACE model)')
        links.pack(fill=tk.X, pady=5)
        self._t(ttk.Label(links), 'Signal map').grid(row=0, column=0, sticky='w')
        self.signal_map_selector = ttk.Combobox(links, textvariable=self.signal_map_choice,
                                                state='readonly', width=30)
        self.signal_map_selector.grid(row=0, column=1, sticky='ew')
        self.signal_map_selector.bind('<<ComboboxSelected>>', self.choose_signal_map)
        self.update_signal_map_choices()
        self._t(ttk.Button(links, command=self.edit_signal_map), 'Edit links…').grid(
            row=1, column=0, columnspan=2, sticky='w')
        ttk.Label(links, textvariable=self.signal_map_status, wraplength=430).grid(
            row=2, column=0, columnspan=2, sticky='w')

        custom = self._t(ttk.LabelFrame(sidebar, padding=5), 'Measured signals / custom subplots')
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
        self.listed_keys = []
        unit_row = ttk.Frame(custom)
        unit_row.pack(fill=tk.X)
        self._t(ttk.Label(unit_row), 'Unit of the recorded signal:').pack(side=tk.LEFT)
        ttk.Combobox(unit_row, textvariable=self.input_unit, values=SIGNAL_UNITS,
                     state='readonly', width=14).pack(side=tk.LEFT)
        self._t(ttk.Button(unit_row, command=self.assign_units), 'Set for selected').pack(side=tk.LEFT)
        self._t(ttk.Button(custom, command=self.show_signal_listing), 'List all signals…').pack(anchor=tk.W)
        action = ttk.Frame(custom)
        action.pack(fill=tk.X)
        for label, command in [('Add subplot', self.add_group),
                               ('Replace selected', self.replace_group),
                               ('Remove subplot', self.remove_group)]:
            self._t(ttk.Button(action, command=command), label).pack(side=tk.LEFT)
        self.group_list = tk.Listbox(custom, height=4, exportselection=False)
        self.group_list.pack(fill=tk.X)
        self._t(ttk.Label(custom, wraplength=420),
                'Ctrl/Shift: several signals on the same axis. @quantities at the top follow the signal map '
                'and work across dSPACE models. raw = no conversion; p.u. (Nm) etc. = already normalized '
                'input. Select the input unit first and press Set for selected.').pack(anchor=tk.W)
        editor = self._t(ttk.LabelFrame(sidebar, padding=5), 'Subplot signals and calculations')
        editor.pack(fill=tk.X, pady=5)
        tree_frame = ttk.Frame(editor)
        tree_frame.pack(fill=tk.X)
        self.trace_tree = ttk.Treeview(tree_frame, columns=('stage', 'scale', 'offset'),
                                      selectmode='browse', height=9)
        self._heading(self.trace_tree, '#0', 'Subplot / signal source')
        self._heading(self.trace_tree, 'stage', 'Current derived processing')
        self._heading(self.trace_tree, 'scale', 'Gain')
        self._heading(self.trace_tree, 'offset', 'Offset')
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
        for label, command in [('Add selected signals', self.add_preset_signals),
                               ('Remove / restore signal', self.toggle_preset_signal),
                               ('Restore signals', self.reset_preset_signals)]:
            self._t(ttk.Button(signal_actions, command=command), label).pack(side=tk.LEFT)
        self._t(ttk.Label(editor, wraplength=430),
                'Preset: select the target subplot here and the recorded signals and their input units above. '
                'A hidden original can be restored with the same button. Added signals use their own time '
                'raster.').pack(anchor=tk.W)
        ttk.Label(editor, textvariable=self.trace_detail, wraplength=430).pack(anchor=tk.W)
        y_row = ttk.Frame(editor)
        y_row.pack(fill=tk.X)
        ttk.Label(y_row, text='Y min').pack(side=tk.LEFT)
        ttk.Entry(y_row, textvariable=self.y_min, width=12).pack(side=tk.LEFT)
        ttk.Label(y_row, text='Y max').pack(side=tk.LEFT)
        ttk.Entry(y_row, textvariable=self.y_max, width=12).pack(side=tk.LEFT)
        y_buttons = ttk.Frame(editor)
        y_buttons.pack(fill=tk.X)
        self._t(ttk.Button(y_buttons, command=self.assign_y_limits), 'Set Y limits').pack(side=tk.LEFT)
        self._t(ttk.Button(y_buttons, command=self.reset_y_limits), 'Default limits').pack(side=tk.LEFT)
        self._t(ttk.Label(editor, wraplength=430),
                'Select a subplot or its signal. Empty limit = default/automatic. Limits are in displayed '
                'units; switching p.u. does not convert the entered numbers.').pack(anchor=tk.W)
        ticks_row = ttk.Frame(editor)
        ticks_row.pack(fill=tk.X)
        self._t(ttk.Label(ticks_row), 'Y major step').pack(side=tk.LEFT)
        ttk.Entry(ticks_row, textvariable=self.y_major_step, width=10).pack(side=tk.LEFT)
        self._t(ttk.Label(ticks_row), 'Y minor step').pack(side=tk.LEFT)
        ttk.Entry(ticks_row, textvariable=self.y_minor_step, width=10).pack(side=tk.LEFT)
        ticks_buttons = ttk.Frame(editor)
        ticks_buttons.pack(fill=tk.X)
        self._t(ttk.Button(ticks_buttons, command=self.assign_y_ticks), 'Set Y steps').pack(side=tk.LEFT)
        self._t(ttk.Button(ticks_buttons, command=self.reset_y_ticks), 'Restore Y steps').pack(side=tk.LEFT)
        self._t(ttk.Label(editor, wraplength=430),
                'Steps are in displayed units. Empty = default; switching p.u. does not convert the set '
                'steps.').pack(anchor=tk.W)
        edit_row = ttk.Frame(editor)
        edit_row.pack(fill=tk.X)
        self._t(ttk.Label(edit_row), 'Gain').pack(side=tk.LEFT)
        ttk.Entry(edit_row, textvariable=self.edit_scale, width=14).pack(side=tk.LEFT)
        self._t(ttk.Label(edit_row), 'Offset').pack(side=tk.LEFT)
        ttk.Entry(edit_row, textvariable=self.edit_offset, width=14).pack(side=tk.LEFT)
        edit_buttons = ttk.Frame(editor)
        edit_buttons.pack(fill=tk.X)
        self._t(ttk.Button(edit_buttons, command=self.assign_transform), 'Set transform').pack(side=tk.LEFT)
        self._t(ttk.Button(edit_buttons, command=self.reset_transform),
                'Reset selected to default').pack(side=tk.LEFT)
        self._t(ttk.Label(editor, wraplength=430),
                'y → gain × y + offset before p.u. conversion. Mechanical → electrical speed: gain n_p '
                '(replaces the current gain). Allowed e.g. 1/n_p, 2*n_p, -1, i_base. The edited axis is '
                'autoscaled.').pack(anchor=tk.W)
        self.refresh_trace_editor()
        export = self._t(ttk.LabelFrame(sidebar, padding=5), 'PDF export')
        export.pack(fill=tk.X)
        self._t(ttk.Label(export), 'Empty = source folder of each measurement').pack(anchor=tk.W)
        ttk.Entry(export, textvariable=self.output).pack(fill=tk.X)
        self._t(ttk.Button(export, command=self.choose_output), 'Output folder…').pack(side=tk.LEFT)
        self._t(ttk.Button(export, command=lambda: self.output.set('')), 'Source folder').pack(side=tk.LEFT)
        actions = ttk.Frame(sidebar)
        actions.pack(fill=tk.X, pady=5)
        self.preview_button = self._t(ttk.Button(actions, command=self.preview), 'Preview')
        self.preview_button.pack(side=tk.LEFT)
        self.export_button = self._t(ttk.Button(actions, command=self.export), 'Save selected as PDF')
        self.export_button.pack(side=tk.LEFT)
        self._t(ttk.Button(actions, command=self.cancel), 'Cancel batch').pack(side=tk.LEFT)
        self._t(ttk.Button(sidebar, command=self.reset_legends),
                'Reset legends of this figure type').pack(anchor=tk.W)
        self._t(ttk.Label(sidebar, wraplength=430),
                "Drag legends in the preview (zoom/pan off). Positions are kept in this session's PDF "
                'batches.').pack(anchor=tk.W)
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
                item = self.files.insert('', tk.END, text=path.name)
                self.set_file_state(item, 'Added')
                self.path_ids[item] = path
                first = first or item
        if first:
            self.files.selection_set(first)
            self.files.focus(first)

    def add_files(self):
        paths = filedialog.askopenfilenames(filetypes=[(self.tr('MAT measurements'), '*.mat *.MAT'),
                                                       (self.tr('All files'), '*.*')])
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
            self.file_states.pop(item, None)
            self.files.delete(item)
        self.signal_keys = []
        self.signal_catalog = None
        self.filter_signals()
        self.refresh_signal_map_status()

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
                return data.catalog()

        def loaded(catalog):
            if self.active_path() == path:
                keys = catalog.keys()
                self.signal_keys = keys
                self.signal_catalog = catalog
                self.signal_catalog_name = path.name
                self.signal_catalog_path = path
                self.spike_selector.configure(values=[*self.scalar_role_keys(), *keys])
                self.filter_signals()
                self.refresh_signal_map_status()
                if self.map_dialog is not None and self.map_dialog.window.winfo_exists():
                    self.map_dialog.set_catalog(catalog, path.name)
                self.status.set(self.tr('{name}: {count} signals. Select a figure or build subplots.',
                                        name=path.name, count=len(keys)))
            else:
                self.selection_changed()
        self.start_job(read_keys, loaded)

    @staticmethod
    def scalar_role_keys():
        return [f'@{role}' for role, *_ in SIGNAL_ROLES if role not in VECTOR_ROLES]

    def signal_entries(self):
        """(key, shown text, search text) of the @quantities and then the recording keys."""
        resolved = (resolve_signal_map(self.signal_catalog, self.current_signal_map())
                    if self.signal_catalog is not None else {})
        entries = []
        for role, label, *_ in SIGNAL_ROLES:
            key = f'@{role}'
            text = key if self.signal_catalog is None else f'{key} → {resolved[role] or "✗"}'
            entries.append((key, text, f'{text} {self.tr(label)}'))
        return entries + [(key, key, key) for key in self.signal_keys]

    def selected_signal_keys(self):
        return [self.listed_keys[index] for index in self.signals.curselection()]

    def filter_signals(self):
        selected = set(self.selected_signal_keys())
        self.signals.delete(0, tk.END)
        self.listed_keys = []
        query = self.search.get().casefold()
        for key, text, searched in self.signal_entries():
            if query in searched.casefold():
                self.signals.insert(tk.END, text)
                self.listed_keys.append(key)
                if key in selected:
                    self.signals.selection_set(tk.END)

    def current_traces(self):
        scene_id = self.kind.get()
        kind = self.scene_kind(scene_id)
        traces = subplot_traces('custom' if kind == 'configured' else kind, self.groups,
                                self.current_signal_map(), unlinked=self.tr('({role} not linked)'))
        for trace in traces:
            native_id = trace['id']
            trace['title'] = (self.tr('Subplot {number}', number=trace['panel'] + 1)
                              if kind in ('custom', 'configured') else self.tr(trace['title']))
            if scene_id != kind or kind == 'configured':
                trace['id'] = self._native_ui_id(scene_id, native_id)
            if kind == 'configured' and trace['panel'] < len(self.scene_panel_metadata[scene_id]):
                trace['title'] = self.scene_panel_metadata[scene_id][trace['panel']]['title']
            if kind == 'hfsi' and native_id == 'hfsi:4:2' and self.torque_mode.get() == 'rated':
                trace['stage'] = self.tr('recorded p.u.; converted to rated torque')
                trace['scale'] = '1'
        if kind not in ('custom', 'configured'):
            titles = {trace['panel']: trace['title'] for trace in traces}
            for added in self.preset_added[scene_id]:
                traces.append({'id': added['id'], 'panel': added['panel'],
                               'title': titles[added['panel']], 'source': added['key'],
                               'stage': self.tr('added, own time raster of the recording'),
                               'scale': '1', 'offset': '0', 'divisor': '', 'roles': ()})
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
            name = trace['source'] + (' ' + self.tr('[off]') if trace['id'] in self.preset_disabled else '')
            self.trace_tree.insert(parents[panel], tk.END, iid=trace['id'], text=name,
                                   values=(trace['stage'], change['scale'], change['offset']))
        self.refresh_signal_map_status()
        if select_id in self.trace_records or select_id in self.panel_records:
            self.trace_tree.selection_set(select_id)
            self.trace_selected()
        else:
            self.y_min.set('')
            self.y_max.set('')
            self.y_major_step.set('')
            self.y_minor_step.set('')
            self.trace_detail.set(self.tr('Select a subplot for its Y limits or a signal for its calculations.'))

    def add_preset_signals(self):
        kind = self.kind.get()
        panel_key = self.selected_panel_key()
        if self.scene_kind() in ('custom', 'configured') or panel_key is None:
            self.status.set(self.tr('First select a target subplot of a preset, or its signal.'))
            return
        signals = self.selected_signal_keys()
        if not signals:
            self.status.set(self.tr('Select the signals to add from the signal list.'))
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
            self.status.set(self.tr('The selected signals are already in that subplot.'))

    def toggle_preset_signal(self):
        kind = self.kind.get()
        selected = self.trace_tree.selection()
        if self.scene_kind() in ('custom', 'configured') or not selected or selected[0] not in self.trace_records:
            self.status.set(self.tr('Select a single signal of a preset.'))
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
            self.status.set(self.tr('Select a preset to restore.'))
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
            self.trace_detail.set(self.tr('{title}: edit the Y-axis limits.',
                                          title=self.panel_records[selected[0]]['title']))
            return
        trace = self.trace_records[selected[0]]
        change = self.transform_overrides.get(trace['id'], trace)
        self.edit_scale.set(change['scale'])
        self.edit_offset.set(change['offset'])
        self.trace_detail.set(self.tr('{title}: {source}. Derived processing: {stage}. Default gain: {scale}.',
                                      title=trace['title'], source=trace['source'],
                                      stage=trace['stage'] or self.tr('none'), scale=trace['scale']))

    def assign_y_limits(self):
        key = self.selected_panel_key()
        if key is None:
            return
        try:
            values = [entry.get().strip().replace(',', '.') for entry in (self.y_min, self.y_max)]
            bounds = tuple(float(value) if value else None for value in values)
            if any(value is not None and not np.isfinite(value) for value in bounds):
                raise ValueError(self.tr('Y limits must be finite'))
            if bounds[0] is not None and bounds[1] is not None and bounds[1] <= bounds[0]:
                raise ValueError(self.tr('Y max must be greater than Y min'))
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
                raise ValueError(self.tr('Y steps must be positive and finite'))
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
        selected = self.selected_signal_keys()
        if selected:
            key = selected[0]
            unit = self.scenarios.get(self.kind.get(), {}).get('units', {}).get(key, role_unit(key))
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
        for key in self.selected_signal_keys():
            self.signal_units[key] = unit
        self.refresh_groups(switch_to_custom=False)
        self.units_changed()

    def refresh_groups(self, switch_to_custom=True):
        self.group_list.delete(0, tk.END)
        for index, group in enumerate(self.groups, 1):
            names = [f'{key} [{self.signal_units.get(key, role_unit(key))}]' for key in group]
            self.group_list.insert(tk.END, f'{index}: ' + ', '.join(names))
        if switch_to_custom and self.scene_kind() not in ('custom', 'configured'):
            self.kind.set('custom')
        self.refresh_trace_editor()

    def add_group(self):
        keys = self.selected_signal_keys()
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
                    'title': self.tr('Subplot {number}', number=len(self.groups)),
                    'signals': [{'key': key, 'unit': self.signal_units.get(key, role_unit(key)),
                                 'scale': '1', 'offset': '0'} for key in keys]})
            self.refresh_groups()

    def replace_group(self):
        indices = self.group_list.curselection()
        keys = self.selected_signal_keys()
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
                    {'key': key, 'unit': self.signal_units.get(key, role_unit(key)), 'scale': '1', 'offset': '0'}
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
                raise ValueError(self.tr('{key}: the value must be finite', key=key))
            return result
        motor = {key: number(key) for key in ('U_nom', 'I_nom', 'f_nom', 'P_nom', 'tau_nom', 'n_p')}
        if any(value <= 0 for value in motor.values()) or not motor['n_p'].is_integer():
            raise ValueError(self.tr('Motor values must be positive; n_p is an integer'))
        motor['n_p'] = int(motor['n_p'])
        torque_mode = self.torque_mode.get()
        if torque_mode not in ('per_unit', 'rated'):
            raise ValueError(self.tr('Select the torque display'))
        pre, post = number('pre'), number('post')
        duration, step = number('duration', True), number('x_step', True)
        minor_step = number('x_minor_step', True)
        if (pre < 0 or post < 0 or (duration is not None and duration <= 0)
                or (step is not None and step <= 0) or (minor_step is not None and minor_step <= 0)):
            raise ValueError(self.tr('Margins ≥ 0; duration and time steps > 0'))
        scene_id = self.kind.get()
        kind = self.scene_kind()
        generic = kind in ('custom', 'configured')
        if generic and not self.groups:
            raise ValueError(self.tr('Add at least one subplot'))
        columns = 1
        if generic:
            try:
                columns = int(self.columns.get().strip())
            except ValueError as error:
                raise ValueError(self.tr('The number of columns must be an integer 1–8')) from error
            if not 1 <= columns <= 8:
                raise ValueError(self.tr('The number of columns must be an integer 1–8'))
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
            raise ValueError(self.tr('Select a known style and palette'))
        detector = self.detector.get()
        if detector not in ('references', 'spikes'):
            raise ValueError(self.tr('Select references or spikes'))
        window = {'pre_transient': pre, 'post_transient': post,
                  'duration': duration, 't_offset': number('offset', True), 'detector': detector}
        if detector == 'spikes':
            signal = self.spike_signal.get().strip()
            if not signal:
                raise ValueError(self.tr('Select the signal for measurement spikes'))
            threshold = float(self.spike_threshold.get().strip().replace(',', '.'))
            if not np.isfinite(threshold) or threshold <= 0:
                raise ValueError(self.tr('The spike threshold must be positive and finite'))
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
                'signal_map': copy.deepcopy(self.current_signal_map()),
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
        self.status.set(self.tr('Error: {error}', error=error))
        messagebox.showerror(self.tr('Plotting failed'), str(error), parent=self.root)

    def preview(self):
        path = self.active_path()
        if not path:
            return
        try:
            options = self.options()
        except ValueError as error:
            self.report_error(error)
            return
        self.status.set(self.tr('Reading {name}…', name=path.name))

        def loaded(payload):
            fig, _ = render_request(payload, options)
            self.show_figure(fig, kind=options['scene_id'])
            self.preview_path = path
            self.render_preview_source()
            self.status.set(self.tr('{name}: {kind}. Drag the legend; switch zoom/pan off while dragging.',
                                    name=path.name, kind=options['kind']))
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
            self.report_error(ValueError(self.tr('Measurements with the same name: use source folders or '
                                                 'rename the files.')))
            return
        existing = sum(target.exists() for target in targets)
        if existing and not messagebox.askyesno(self.tr('Replace PDFs?'),
                                               self.tr('{count} files already exist. Replace them?', count=existing),
                                               parent=self.root):
            return
        self.cancelled = False
        self.batch_active = True
        self.export_next(list(zip(paths, targets)), options, [], [])

    def export_next(self, pending, options, saved, errors):
        if not pending or self.cancelled or self.closing:
            self.batch_active = False
            self.set_busy(False)
            counts = {'saved': len(saved), 'errors': len(errors)}
            self.status.set(self.tr('Batch: {saved} saved, {errors} errors; cancelled.', **counts)
                            if self.cancelled else self.tr('Batch: {saved} saved, {errors} errors.', **counts))
            if errors and not self.closing:
                messagebox.showwarning(self.tr('Batch errors'), '\n'.join(errors), parent=self.root)
            return
        path, target = pending.pop(0)
        self.status.set(self.tr('Saving {name}… ({count} left)', name=path.name, count=len(pending)))

        def update_row(state):
            for item, source in self.path_ids.items():
                if source == path:
                    self.set_file_state(item, state)

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
                update_row('Saved')
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
                if fig is not None:
                    plt.close(fig)
            next_file()

        def failed(error):
            errors.append(f'{path.name}: {error}')
            update_row('Error')
            next_file()
        self.start_job(lambda: load_plot_request(path, options), loaded, failed)

    def cancel(self):
        self.cancelled = True
        if self.busy:
            self.status.set(self.tr('Cancelling after the current file…'))

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
        messagebox.showerror(translate('Configuration file error', load_language(SETTINGS_PATH)),
                             str(error), parent=root)
        root.destroy()
        return
    app.insert_paths(sys.argv[1:])
    root.mainloop()


if __name__ == '__main__':
    main()
