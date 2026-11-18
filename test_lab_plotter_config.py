import copy
import json
import tempfile
import unittest
from pathlib import Path

from lab_plotter_config import load_motor_profiles, load_scenarios


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
                         ['wm_ref', 'w_m', 'tau_m'])
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

    def test_bad_json_and_root(self):
        for text in ['{', '{"scenarios": [], "scenarios": []}',
                     '{"scenarios": [], "unknown": 1}', '{"scenarios": {}}']:
            with self.subTest(text=text):
                self.path.write_text(text, encoding='utf-8')
                with self.assertRaises(ValueError):
                    load_scenarios(self.path)


if __name__ == '__main__':
    unittest.main()
