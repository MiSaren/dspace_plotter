"""Read editable JSON plot scenarios and motor profiles without executing them.

``measurement_scenarios.json`` contains a ``scenarios`` array. Each item has
``id``, ``label``, ``kind`` and a filename ``suffix``. Built-in kinds are
``hfsi``, ``overview``, ``injection-overview`` and ``presentation``; optional
``disabled_traces``, ``added_traces`` (id, zero-based panel, key),
``transforms`` (trace id to scale/offset expressions), and ``units`` (signal
key to unit) override the preset. Native IDs are ``kind:panel:line``.
A ``configured`` item instead has ``columns`` and ``panels``: each panel has a
``title`` and nonempty ``signals`` array, with each signal containing a key,
unit, scale and offset. Missing units/scale/offset default to raw/1/0. Signal
keys must be unique within each panel, but may repeat across panels.
Mechanical speed units are ``rad/s (mechanical)``, ``rpm (mechanical)`` and
``p.u. (rad/s mechanical)``; ``rad/s`` is electrical. Flux linkage can also
use ``Vs`` or ``p.u. (Vs)`` in addition to the equivalent Wb units.
Expressions use numbers, base names, +, -, *, / and parentheses only; they are
validated here but evaluated only by the renderer against its selected motor.

``motor_profiles.json`` contains a ``motors`` array with id, name, U_nom,
I_nom, f_nom, P_nom, tau_nom and n_p. The numeric values must be positive and
finite, and n_p must be an integer. Loaders read afresh on every call, return
ID-keyed dictionaries in file order, and never modify the files.

``signal_maps.json`` contains a ``signal_maps`` array linking the quantities
(roles) of the built-in figures to recording keys of one dSPACE model. Each
item has id, name and a ``signals`` object mapping a role from SIGNAL_ROLES to
a key or an ordered list of alternative keys; the first key found in a
recording is used, and an empty list leaves the quantity unlinked. Omitted
roles use the built-in default keys. Without the file, only the built-in
``default`` map exists. Only save_signal_map writes this file.

Wherever a scenario names a recording key (configured signals, added traces,
units), ``@role`` instead selects the key the active signal map links to that
role, e.g. ``@w_m_ref``. A configured ``@role`` signal without a unit uses the
role's recorded unit from SIGNAL_ROLES (ROLE_UNITS).
"""

import ast
import json
import math
import os
import re
import tempfile
from pathlib import Path

SCENARIO_PATH = Path(__file__).with_name('measurement_scenarios.json')
MOTOR_PATH = Path(__file__).with_name('motor_profiles.json')
SIGNAL_MAP_PATH = Path(__file__).with_name('signal_maps.json')

# Quantities read by the built-in figures: (role, description, default keys,
# vector, recorded unit). Roles are the attribute names of load_measurement
# contexts. Vector roles name a 2-element (x + jy) or 3-element (abc) signal
# without [index]. Units are those the presets assume; raw where unspecified.
SIGNAL_ROLES = (
    ('w_m_ref', 'Speed reference (electrical rad/s)', ('wm_ref',), False, 'rad/s'),
    ('w_mest', 'Estimated speed (electrical rad/s)', ('w_m',), False, 'rad/s'),
    ('w_m_meas', 'Measured speed (HFSI figure, default gain 1/2)', ('wm_meas',), False, 'raw'),
    ('theta_m', 'Estimated rotor angle (electrical rad), also dq transform',
     ('theta_m_out',), False, 'raw'),
    ('theta_meas', 'Measured rotor angle (electrical rad)', ('thetam',), False, 'raw'),
    ('psi_sref', 'Flux reference |ψs| (Wb)', ('psis_ref',), False, 'Wb'),
    ('psi_sest', 'Estimated flux ψs (2-vector, Wb)', ('psi_s',), True, 'Wb'),
    ('tau_Mref_lim', 'Limited torque reference (Nm)', ('tauM_ref_lim',), False, 'Nm'),
    ('tau_Mest', 'Estimated torque (Nm)', ('tau_m',), False, 'Nm'),
    ('tau_Lref', 'Load torque reference (p.u.)', ('tauLref', 'TL_ref'), False, 'p.u. (Nm)'),
    ('i_cs', 'Measured currents (abc or 2-vector, A)', ('i_c_abc',), True, 'A'),
    ('u_dc', 'DC voltage (V)', ('u_dc_lim',), False, 'V'),
    ('u_c', 'Realized voltage (2-vector, V)', ('ucs_realized',), True, 'V'),
    ('u_inj', 'Injection voltage (2-vector, V)', ('u_sd_inj',), True, 'V'),
    ('eps_inj', 'Injection error signal (rad)', ('mon[3]',), False, 'raw'),
    ('i_s', 'Stator current (2-vector, A)', ('i_s',), True, 'A'),
    ('i_ref', 'Current reference (2-vector, A)', ('cur_is_ref',), True, 'A'),
    ('w_inj', 'Observer monitor signal mon[4]', ('mon[4]',), False, 'raw'),
)
SIGNAL_ROLE_IDS = tuple(role for role, *_ in SIGNAL_ROLES)
ROLE_UNITS = {role: unit for role, *_, unit in SIGNAL_ROLES}
ROLE_PREFIX = '@'


def signal_role(key):
    """Return the role of an ``@role`` key, or None for a recording key."""
    return key[len(ROLE_PREFIX):] if isinstance(key, str) and key.startswith(ROLE_PREFIX) else None

_UNITS = frozenset(('raw', 'A', 'V', 'rad/s', 'Nm', 'Wb', 'Vs',
                    'rad/s (mechanical)', 'rpm (mechanical)',
                    'p.u. (A)', 'p.u. (V)', 'p.u. (rad/s)',
                    'p.u. (rad/s mechanical)', 'p.u. (Nm)',
                    'p.u. (Wb)', 'p.u. (Vs)'))
_NAMES = frozenset(('n_p', 'sqrt3', 'u_base', 'i_base', 'w_base',
                    'psi_base', 'tau_base', 'tau_nom'))
_PANEL_LINES = {'hfsi': (3, 2, 2, 2, 3, 2),
                'overview': (2, 2, 3, 2, 2),
                'injection-overview': (2, 2, 3, 2, 2),
                'presentation': (2, 2, 3, 2)}
_SAFE_ID = re.compile(r'[A-Za-z0-9_-]+\Z')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _read(path, root):
    try:
        with Path(path).open(encoding='utf-8') as source:
            data = json.load(source, object_pairs_hook=_pairs)
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f'Cannot read {root} configuration: {error}') from error
    _object(data, {root}, {root}, 'root')
    if not isinstance(data[root], list) or not data[root]:
        raise ValueError(f'{root} must be a nonempty list')
    return data[root]


def _object(value, required, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f'{where} must be an object')
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        raise ValueError(f'{where}: missing {sorted(missing)}')
    if unknown:
        raise ValueError(f'{where}: unknown fields {sorted(unknown)}')


def _text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{where} must be a nonempty string')
    return value


def _key(value, where):
    """A recording key, or an @role key naming a known role."""
    _text(value, where)
    role = signal_role(value)
    if role is not None and role not in SIGNAL_ROLE_IDS:
        raise ValueError(f'{where}: unknown quantity role {value!r}')
    return value


def _id(value, where):
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f'{where} must contain only letters, digits, underscores or hyphens')
    return value


def _unit(value, where):
    if not isinstance(value, str) or value not in _UNITS:
        raise ValueError(f'{where}: unknown unit {value!r}')
    return value


def _expression(value, where):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 120:
        raise ValueError(f'{where} must be a scalar expression string of at most 120 characters')
    try:
        tree = ast.parse(value.strip(), mode='eval')
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ValueError(f'{where}: invalid scalar expression') from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError(f'{where}: scalar expression is too complex')

    def valid(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            try:
                return math.isfinite(node.value)
            except OverflowError:
                return False
        if isinstance(node, ast.Name):
            return node.id in _NAMES
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return valid(node.operand)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            return valid(node.left) and valid(node.right)
        return False

    if not valid(tree.body):
        raise ValueError(f'{where}: invalid scalar expression')
    return value


def _transform(value, where):
    _object(value, set(), {'scale', 'offset'}, where)
    return {'scale': _expression(value.get('scale', '1'), f'{where}.scale'),
            'offset': _expression(value.get('offset', '0'), f'{where}.offset')}


def _builtin(item, kind):
    native = {f'{kind}:{panel}:{line}' for panel, count in enumerate(_PANEL_LINES[kind])
              for line in range(count)}
    disabled = item.get('disabled_traces', [])
    if not isinstance(disabled, list):
        raise ValueError('disabled_traces must be a list')
    seen = set()
    for trace_id in disabled:
        _text(trace_id, 'disabled trace ID')
        if trace_id not in native or trace_id in seen:
            raise ValueError(f'Invalid or duplicate disabled trace ID: {trace_id}')
        seen.add(trace_id)
    added = item.get('added_traces', [])
    if not isinstance(added, list):
        raise ValueError('added_traces must be a list')
    ids = set(native)
    for trace in added:
        _object(trace, {'id', 'panel', 'key'}, {'id', 'panel', 'key'}, 'added trace')
        trace_id = _text(trace['id'], 'added trace ID')
        if trace_id in ids:
            raise ValueError(f'Duplicate trace ID: {trace_id}')
        ids.add(trace_id)
        if type(trace['panel']) is not int or not 0 <= trace['panel'] < len(_PANEL_LINES[kind]):
            raise ValueError(f'Invalid added trace panel: {trace["panel"]!r}')
        _key(trace['key'], 'added trace key')
    transforms = item.get('transforms', {})
    if not isinstance(transforms, dict):
        raise ValueError('transforms must be an object')
    normalized = {}
    for trace_id, transform in transforms.items():
        if trace_id not in ids:
            raise ValueError(f'Unknown transform trace ID: {trace_id}')
        normalized[trace_id] = _transform(transform, f'transform {trace_id}')
    units = item.get('units', {})
    if not isinstance(units, dict):
        raise ValueError('units must be an object')
    for key, unit in units.items():
        _key(key, 'unit signal key')
        _unit(unit, f'unit for {key}')
    item.update(disabled_traces=disabled, added_traces=added,
                transforms=normalized, units=units)


def _configured(item):
    columns = item['columns']
    if type(columns) is not int or not 1 <= columns <= 8:
        raise ValueError('columns must be an integer from 1 to 8')
    panels = item['panels']
    if not isinstance(panels, list) or not panels:
        raise ValueError('panels must be a nonempty list')
    for panel in panels:
        _object(panel, {'title', 'signals'}, {'title', 'signals'}, 'panel')
        _text(panel['title'], 'panel title')
        signals = panel['signals']
        if not isinstance(signals, list) or not signals:
            raise ValueError('panel signals must be a nonempty list')
        keys = set()
        for signal in signals:
            _object(signal, {'key'}, {'key', 'unit', 'scale', 'offset'}, 'signal')
            key = _key(signal['key'], 'signal key')
            if key in keys:
                raise ValueError(f'Duplicate signal key in panel: {key}')
            keys.add(key)
            signal['unit'] = _unit(signal.get('unit', ROLE_UNITS.get(signal_role(key), 'raw')),
                                   'signal unit')
            signal['scale'] = _expression(signal.get('scale', '1'), 'signal scale')
            signal['offset'] = _expression(signal.get('offset', '0'), 'signal offset')


def load_scenarios(path=SCENARIO_PATH) -> dict[str, dict]:
    """Read and validate scenarios; later calls see edits to the JSON file."""
    result = {}
    common = {'id', 'label', 'kind', 'suffix'}
    for item in _read(path, 'scenarios'):
        _object(item, common, common | {'disabled_traces', 'added_traces', 'transforms',
                                        'units', 'columns', 'panels'}, 'scenario')
        scenario_id = _id(item['id'], 'scenario id')
        if scenario_id == 'custom' or scenario_id in result:
            raise ValueError(f'Reserved or duplicate scenario ID: {scenario_id}')
        _text(item['label'], 'scenario label')
        _id(item['suffix'], 'scenario suffix')
        kind = item['kind']
        if not isinstance(kind, str):
            raise ValueError('scenario kind must be a string')
        if kind in _PANEL_LINES:
            _object(item, common, common | {'disabled_traces', 'added_traces', 'transforms', 'units'}, 'scenario')
            _builtin(item, kind)
        elif kind == 'configured':
            _object(item, common | {'columns', 'panels'}, common | {'columns', 'panels'}, 'scenario')
            _configured(item)
        else:
            raise ValueError(f'Unknown scenario kind: {kind}')
        result[scenario_id] = item
    return result


def load_motor_profiles(path=MOTOR_PATH) -> dict[str, dict]:
    """Read profiles including id/name; nominal fields can be passed to BaseValues."""
    result = {}
    fields = {'id', 'name', 'U_nom', 'I_nom', 'f_nom', 'P_nom', 'tau_nom', 'n_p'}
    for motor in _read(path, 'motors'):
        _object(motor, fields, fields, 'motor')
        motor_id = _id(motor['id'], 'motor id')
        if motor_id in result:
            raise ValueError(f'Duplicate motor ID: {motor_id}')
        _text(motor['name'], 'motor name')
        for field in fields - {'id', 'name'}:
            value = motor[field]
            if type(value) not in (int, float):
                raise ValueError(f'{field} must be a positive finite number')
            try:
                valid = value > 0 and math.isfinite(value)
            except OverflowError:
                valid = False
            if not valid or (field == 'n_p' and type(value) is not int):
                raise ValueError(f'{field} must be a positive finite {"integer" if field == "n_p" else "number"}')
        result[motor_id] = motor
    return result


def default_signal_map() -> dict:
    """Return the built-in map, matching the keys the presets originally used."""
    return {'id': 'default', 'name': 'Default',
            'signals': {role: list(keys) for role, _, keys, *_ in SIGNAL_ROLES}}


def signal_keys(value, where):
    """Normalize one role's key or alternative keys to a list."""
    keys = [value] if isinstance(value, str) else value
    if not isinstance(keys, list):
        raise ValueError(f'{where} must be a key or a list of keys')
    for key in keys:
        if not isinstance(key, str) or not key.strip() or key != key.strip():
            raise ValueError(f'{where}: keys must be nonempty strings without surrounding spaces')
    if len(set(keys)) != len(keys):
        raise ValueError(f'{where}: duplicate keys')
    return list(keys)


def load_signal_maps(path=SIGNAL_MAP_PATH) -> dict[str, dict]:
    """Read signal maps with every role normalized to a list of keys."""
    if not Path(path).is_file():
        return {'default': default_signal_map()}
    result = {}
    defaults = default_signal_map()['signals']
    for item in _read(path, 'signal_maps'):
        _object(item, {'id', 'name', 'signals'}, {'id', 'name', 'signals'}, 'signal map')
        map_id = _id(item['id'], 'signal map id')
        if map_id in result:
            raise ValueError(f'Duplicate signal map ID: {map_id}')
        _text(item['name'], 'signal map name')
        signals = item['signals']
        if not isinstance(signals, dict):
            raise ValueError(f'signal map {map_id}: signals must be an object')
        unknown = signals.keys() - set(SIGNAL_ROLE_IDS)
        if unknown:
            raise ValueError(f'signal map {map_id}: unknown roles {sorted(unknown)}')
        result[map_id] = {'id': map_id, 'name': item['name'], 'signals': {
            role: signal_keys(signals.get(role, defaults[role]), f'signal map {map_id}, {role}')
            for role in SIGNAL_ROLE_IDS}}
    return result


def _format_signal_maps(entries):
    blocks = []
    for entry in entries:
        roles = ',\n'.join(f'        {json.dumps(role)}: {json.dumps(keys, ensure_ascii=False)}'
                           for role, keys in entry['signals'].items())
        blocks.append('    {\n'
                      f'      "id": {json.dumps(entry["id"])},\n'
                      f'      "name": {json.dumps(entry["name"], ensure_ascii=False)},\n'
                      '      "signals": {\n' + roles + '\n      }\n    }')
    return '{\n  "signal_maps": [\n' + ',\n'.join(blocks) + '\n  ]\n}\n'


def save_signal_map(entry, path=SIGNAL_MAP_PATH):
    """Insert or replace one map by id, keeping the others in file order.

    An invalid existing file is not overwritten. A missing file is created with
    the built-in default map. The new content is validated before it atomically
    replaces the file.
    """
    path = Path(path)
    maps = load_signal_maps(path)
    _object(entry, {'id', 'name', 'signals'}, {'id', 'name', 'signals'}, 'signal map')
    maps[_id(entry['id'], 'signal map id')] = {
        'id': entry['id'], 'name': _text(entry['name'], 'signal map name').strip(),
        'signals': dict(entry['signals'])}
    text = _format_signal_maps(maps.values())
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix='.tmp')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8', newline='\n') as target:
            target.write(text)
        load_signal_maps(temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return load_signal_maps(path)
