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
"""

import ast
import json
import math
import re
from pathlib import Path

SCENARIO_PATH = Path(__file__).with_name('measurement_scenarios.json')
MOTOR_PATH = Path(__file__).with_name('motor_profiles.json')

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
        _text(trace['key'], 'added trace key')
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
        _text(key, 'unit signal key')
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
            key = _text(signal['key'], 'signal key')
            if key in keys:
                raise ValueError(f'Duplicate signal key in panel: {key}')
            keys.add(key)
            signal['unit'] = _unit(signal.get('unit', 'raw'), 'signal unit')
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
