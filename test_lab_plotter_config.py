import copy
import json
import tempfile
import unittest
from pathlib import Path

from lab_plotter_config import (
    ROLE_UNITS,
    SIGNAL_ROLES,
    default_signal_map,
    load_motor_profiles,
    load_scenarios,
    load_signal_maps,
    save_signal_map,
    signal_role,
)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'config.json'
        self.scenario = {'id': 'variant', 'label': 'Variant', 'kind': 'overview', 'suffix': '_v'}
        self.motor = {'id': 'default', 'name': 'Default', 'U_nom': 460, 'I_nom': 8.8,
                      'f_nom': 60, 'P_nom': 5600, 'tau_nom': 29.2, 'n_p': 2}

    def write(self, root, entries):
        self.path.write_text(json.dumps({root: entries}), encoding='utf-8')

    def test_supplied_scenarios_and_motor(self):
        scenarios = load_scenarios()
        self.assertEqual([(key, scenarios[key]['kind'], scenarios[key]['suffix']) for key in
                          ('hfsi', 'overview', 'injection-overview', 'presentation')],
                         [('hfsi', 'hfsi', '_hfsi_2'), ('overview', 'overview', '_1'),
                          ('injection-overview', 'injection-overview', '_2'),
                          ('presentation', 'presentation', '_online_meas')])
        example = scenarios['speed-torque']
        self.assertEqual(example['columns'], 2)
        self.assertEqual([signal['key'] for panel in example['panels'] for signal in panel['signals']],
                         ['@w_m_ref', '@w_mest', '@tau_Mest'])
        self.assertEqual([signal['unit'] for panel in example['panels'] for signal in panel['signals']],
                         ['rad/s', 'rad/s', 'Nm'])
        self.assertEqual(load_motor_profiles()['default'], self.motor)

    def test_reload_order_and_independence(self):
        second = dict(self.scenario, id='other')
        self.write('scenarios', [self.scenario, second])
        first = load_scenarios(self.path)
        self.assertEqual(list(first), ['variant', 'other'])
        first['variant']['added_traces'].append({'id': 'mutated'})
        second['label'] = 'Changed'
        self.write('scenarios', [second])
        reloaded = load_scenarios(self.path)
        self.assertEqual(reloaded['other']['label'], 'Changed')
        self.assertEqual(reloaded['other']['added_traces'], [])
        self.write('motors', [self.motor])
        load_motor_profiles(self.path)['default']['U_nom'] = 1
        self.motor['U_nom'] = 400
        self.write('motors', [self.motor])
        self.assertEqual(load_motor_profiles(self.path)['default']['U_nom'], 400)

    def test_builtin_overrides(self):
        self.scenario.update(disabled_traces=['overview:0:0'],
                             added_traces=[{'id': 'extra', 'panel': 4, 'key': 'wm_ref'}],
                             transforms={'overview:0:1': {'scale': '1 / w_base'},
                                         'extra': {'offset': '-2'}},
                             units={'wm_ref': 'rad/s'})
        self.write('scenarios', [self.scenario])
        result = load_scenarios(self.path)['variant']
        self.assertEqual(result['disabled_traces'], ['overview:0:0'])
        self.assertEqual(result['added_traces'][0]['panel'], 4)
        self.assertEqual(result['transforms']['extra'], {'scale': '1', 'offset': '-2'})
        self.assertEqual(result['units']['wm_ref'], 'rad/s')

    def test_configured_normalization(self):
        scenario = dict(self.scenario, kind='configured', columns=2,
                        panels=[{'title': 'Speed', 'signals': [{'key': 'wm_ref'}]}])
        self.write('scenarios', [scenario])
        self.assertEqual(load_scenarios(self.path)['variant']['panels'][0]['signals'][0],
                         {'key': 'wm_ref', 'unit': 'raw', 'scale': '1', 'offset': '0'})

    def test_configured_signal_keys_unique_per_panel(self):
        scenario = dict(self.scenario, kind='configured', columns=2, panels=[
            {'title': 'First', 'signals': [{'key': 'w_m', 'unit': 'rad/s'}]},
            {'title': 'Second', 'signals': [{'key': 'w_m', 'unit': 'rpm (mechanical)',
                                            'scale': '2'}]}])
        self.write('scenarios', [scenario])
        result = load_scenarios(self.path)['variant']
        self.assertEqual(result['panels'][1]['signals'][0]['scale'], '2')
        scenario['panels'][0]['signals'].append({'key': 'w_m', 'unit': 'raw'})
        self.write('scenarios', [scenario])
        with self.assertRaisesRegex(ValueError, 'Duplicate.*w_m'):
            load_scenarios(self.path)

    def test_mechanical_speed_and_flux_units(self):
        units = ['rad/s (mechanical)', 'rpm (mechanical)',
                 'p.u. (rad/s mechanical)', 'Vs', 'p.u. (Vs)']
        for unit in units:
            with self.subTest(unit=unit):
                builtin = dict(self.scenario, units={'signal': unit})
                configured = dict(self.scenario, id='configured-example', kind='configured',
                                  columns=2, panels=[{'title': 'Panel',
                                                      'signals': [{'key': 'signal', 'unit': unit}]}])
                self.write('scenarios', [builtin, configured])
                result = load_scenarios(self.path)
                self.assertEqual(result['variant']['kind'], 'overview')
                self.assertEqual(result['variant']['units']['signal'], unit)
                self.assertEqual(result['configured-example']['panels'][0]['signals'][0]['unit'], unit)

    def test_invalid_scenarios(self):
        base = self.scenario
        configured = dict(base, kind='configured', columns=2,
                          panels=[{'title': 'x', 'signals': [{'key': 'x'}]}])
        invalid = [dict(base, id='custom'), dict(base, id='../escape'),
                   dict(base, suffix='../escape'), dict(base, kind='custom'),
                   dict(base, unexpected=True), dict(base, disabled_traces=['overview:0:99']),
                   dict(base, disabled_traces=['overview:0:0', 'overview:0:0']),
                   dict(base, added_traces=[{'id': 'overview:0:0', 'panel': 0, 'key': 'x'}]),
                   dict(base, added_traces=[{'id': 'x', 'panel': 5, 'key': 'x'}]),
                   dict(base, added_traces=[{'id': 'x', 'panel': True, 'key': 'x'}]),
                   dict(base, transforms={'unknown': {'scale': '1'}}),
                   dict(base, transforms={'overview:0:0': {'scale': '__import__("os")'}}),
                   dict(base, transforms={'overview:0:0': {'scale': '2 ** 3'}}),
                   dict(base, units={'x': 'bogus'}),
                   dict(configured, columns=True), dict(configured, columns=9),
                   dict(configured, panels=[]),
                   dict(configured, panels=[{'title': 'x', 'signals': []}]),
                   dict(configured, panels=[{'title': 'x', 'signals': [{'key': 'x', 'unit': 'bad'}]}])]
        for scenario in invalid:
            with self.subTest(scenario=scenario):
                self.write('scenarios', [scenario])
                with self.assertRaises(ValueError):
                    load_scenarios(self.path)
        self.write('scenarios', [base, copy.deepcopy(base)])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            load_scenarios(self.path)

    def test_invalid_motor_profiles(self):
        for field, value in [('U_nom', 0), ('I_nom', -1), ('f_nom', float('inf')),
                             ('P_nom', float('nan')), ('tau_nom', True), ('n_p', 2.0),
                             ('n_p', 0), ('name', ''), ('id', '../bad'), ('extra', 1)]:
            with self.subTest(field=field, value=value):
                self.write('motors', [dict(self.motor, **{field: value})])
                with self.assertRaises(ValueError):
                    load_motor_profiles(self.path)
        self.write('motors', [self.motor, self.motor])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_motor_profiles(self.path)

    def test_processing_in_transforms_and_configured_signals(self):
        preset = dict(self.scenario, transforms={'overview:1:1': {'processing': 'real'},
                                                 'overview:0:0': {'scale': '2'}})
        configured = {'id': 'proc', 'label': 'Proc', 'kind': 'configured', 'suffix': '_p', 'columns': 1,
                      'panels': [{'title': 'x', 'signals': [{'key': 'i_s', 'processing': 'magnitude'}]}]}
        self.write('scenarios', [preset, configured])
        result = load_scenarios(self.path)
        self.assertEqual(result['variant']['transforms']['overview:1:1'],
                         {'scale': '1', 'offset': '0', 'processing': 'real'})
        self.assertNotIn('processing', result['variant']['transforms']['overview:0:0'])
        self.assertEqual(result['proc']['panels'][0]['signals'][0]['processing'], 'magnitude')
        for scenario in [dict(preset, transforms={'overview:1:1': {'processing': 'sqrt'}}),
                         dict(configured, panels=[{'title': 'x', 'signals': [{'key': 'a', 'processing': 1}]}])]:
            with self.subTest(scenario=scenario):
                self.write('scenarios', [scenario])
                with self.assertRaisesRegex(ValueError, 'unknown processing'):
                    load_scenarios(self.path)

    def test_role_keys_in_scenarios_and_role_default_units(self):
        configured = {'id': 'roles', 'label': 'Roles', 'kind': 'configured', 'suffix': '_r', 'columns': 1,
                      'panels': [{'title': 'Mixed', 'signals': [
                          {'key': '@tau_Lref'}, {'key': '@w_m_ref', 'unit': 'raw'}, {'key': 'tau_m'}]}]}
        preset = dict(self.scenario, added_traces=[{'id': 'extra', 'panel': 0, 'key': '@w_mest'}],
                      units={'@w_mest': 'rad/s'})
        self.write('scenarios', [configured, preset])
        result = load_scenarios(self.path)
        self.assertEqual([signal['unit'] for signal in result['roles']['panels'][0]['signals']],
                         ['p.u. (Nm)', 'raw', 'raw'])
        self.assertEqual(result['variant']['added_traces'][0]['key'], '@w_mest')
        for scenario in [dict(configured, panels=[{'title': 'x', 'signals': [{'key': '@unknown'}]}]),
                         dict(preset, added_traces=[{'id': 'extra', 'panel': 0, 'key': '@'}]),
                         dict(preset, units={'@nope': 'Nm'})]:
            with self.subTest(scenario=scenario):
                self.write('scenarios', [scenario])
                with self.assertRaisesRegex(ValueError, 'unknown quantity role'):
                    load_scenarios(self.path)
        self.assertEqual(ROLE_UNITS['i_cs'], 'A')
        self.assertEqual(signal_role('@w_m_ref'), 'w_m_ref')
        self.assertIsNone(signal_role('wm_ref'))

    def test_supplied_and_missing_signal_maps_use_builtin_defaults(self):
        default = default_signal_map()
        self.assertEqual(load_signal_maps(), {'default': default})
        self.assertEqual(load_signal_maps(Path(self.directory.name) / 'absent.json'), {'default': default})
        self.assertEqual(list(default['signals']), [role for role, *_ in SIGNAL_ROLES])
        self.assertEqual(default['signals']['tau_Lref'], ['tauLref', 'TL_ref'])

    def test_signal_maps_fill_defaults_normalize_and_reject_invalid(self):
        self.write('signal_maps', [{'id': 'old-model', 'name': 'Old model',
                                    'signals': {'theta_m': 'theta_m', 'eps_inj': [],
                                                'tau_Lref': ['TL_refseq1', 'TL_ref']}}])
        signals = load_signal_maps(self.path)['old-model']['signals']
        self.assertEqual(signals['theta_m'], ['theta_m'])
        self.assertEqual(signals['eps_inj'], [])
        self.assertEqual(signals['tau_Lref'], ['TL_refseq1', 'TL_ref'])
        self.assertEqual(signals['w_m_ref'], ['wm_ref'])
        good = {'id': 'x', 'name': 'X', 'signals': {}}
        for item in [dict(good, signals={'unknown': 'a'}), dict(good, signals={'theta_m': 1}),
                     dict(good, signals={'theta_m': ['']}), dict(good, signals={'theta_m': [' a']}),
                     dict(good, signals={'theta_m': ['a', 'a']}), dict(good, signals=[]),
                     dict(good, id='bad id'), dict(good, name=''), dict(good, extra=1),
                     {'id': 'x', 'name': 'X'}]:
            with self.subTest(item=item):
                self.write('signal_maps', [item])
                with self.assertRaises(ValueError):
                    load_signal_maps(self.path)
        self.write('signal_maps', [good, good])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            load_signal_maps(self.path)

    def test_save_signal_map_inserts_replaces_and_keeps_invalid_files(self):
        path = Path(self.directory.name) / 'maps.json'
        maps = save_signal_map({'id': 'lab-b', 'name': 'Lab B', 'signals': {'theta_m': ['theta_m']}}, path)
        self.assertEqual(list(maps), ['default', 'lab-b'])
        self.assertEqual(maps['lab-b']['signals']['theta_m'], ['theta_m'])
        save_signal_map({'id': 'default', 'name': 'Renamed',
                         'signals': {'w_m_ref': ['Ref/w ref'], 'u_inj': []}}, path)
        maps = load_signal_maps(path)
        self.assertEqual(list(maps), ['default', 'lab-b'])
        self.assertEqual(maps['default']['name'], 'Renamed')
        self.assertEqual(maps['default']['signals']['w_m_ref'], ['Ref/w ref'])
        self.assertEqual(maps['default']['signals']['u_inj'], [])
        self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['signal_maps'][1]['id'], 'lab-b')
        before = path.read_text(encoding='utf-8')
        for entry in [{'id': 'x', 'name': 'X', 'signals': {'unknown': ['a']}},
                      {'id': '../x', 'name': 'X', 'signals': {}}]:
            with self.subTest(entry=entry), self.assertRaises(ValueError):
                save_signal_map(entry, path)
        path.write_text('{invalid', encoding='utf-8')
        with self.assertRaises(ValueError):
            save_signal_map({'id': 'x', 'name': 'X', 'signals': {}}, path)
        self.assertEqual(path.read_text(encoding='utf-8'), '{invalid')
        self.assertNotEqual(before, '{invalid')
        self.assertEqual(sorted(p.name for p in path.parent.iterdir()), ['maps.json'])

    def test_bad_json_and_root(self):
        for text in ['{', '{"scenarios": [], "scenarios": []}',
                     '{"scenarios": [], "unknown": 1}', '{"scenarios": {}}']:
            with self.subTest(text=text):
                self.path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_scenarios(self.path)


if __name__ == '__main__':
    unittest.main()
