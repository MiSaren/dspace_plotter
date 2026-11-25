"""Focused, recording-independent checks for the standalone plotting core."""
import contextlib
import io
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

import lab_plotter_core as core


class SyntheticData:
    def __init__(self, path="first.mat"):
        self.file_path = Path(path)
        self.raster_names = ["main", "slow"]
        self.t = np.array([2., 3., 4., 5.])
        self.closed = False
        self.signals = {
            "wm_ref": (self.t.copy(), np.array([0., 0., 2., 2.])),
            "tau_m": (np.array([2., 4.]), np.array([10., 20.])),
            "motor/current[2]": (np.array([2., 4., 5.]), np.array([1., 2., 3.])),
            "motor/current[7]": (np.array([2., 3.]), np.array([4., 5.])),
        }

    def keys(self):
        return list(self.signals)

    def signal_info(self, key):
        keys = [k for k in self.keys() if k.startswith(key + "[")] if key == "motor/current" else [key]
        result = []
        for k in keys:
            time, values = self.signals[k]
            result.append({"key": k, "name": k, "path": "Model Root/" + k,
                           "raster": 0 if np.array_equal(time, self.t) else 1,
                           "samples": values.size})
        return result

    def __contains__(self, key):
        return key in self.signals

    def __getitem__(self, key):
        return self.signals[key][1]

    def time(self, key):
        return self.signals[key][0]

    def complex(self, key):
        raise KeyError(key)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


class ScalarExpressionTests(unittest.TestCase):
    def test_named_motor_values_and_arithmetic(self):
        base = core.DEFAULT_BASE
        self.assertEqual(core.scalar_expression('2*n_p + 1/n_p', base), 2*base.n_p + 1/base.n_p)
        self.assertEqual(core.scalar_expression('-(i_base - 2)/2', base), -(base.i - 2)/2)
        for expression in ('n_p.real', 'abs(2)', '[1]', 'True', 'unknown', '2**3',
                           '1/0', '1e309', '', '1+'*80 + '1'):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                core.scalar_expression(expression, base)


class CustomSignalTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_vector_components_use_true_keys_and_their_own_rasters(self):
        data = SyntheticData()
        fig, axes = core.plot_signals(data, [["motor/current"], ["wm_ref", "tau_m"]],
                                      t_offset=2, t_span=(0, 3), x_step=.5)
        self.assertEqual(len(axes), 2)
        lines = axes[0].lines
        self.assertEqual([line.get_label() for line in lines],
                         ["motor/current[2]", "motor/current[7]"])
        np.testing.assert_array_equal(lines[0].get_xdata(), [0, 2, 3])
        np.testing.assert_array_equal(lines[1].get_xdata(), [0, 1])
        np.testing.assert_array_equal(axes[1].lines[1].get_xdata(), [0, 2])
        self.assertEqual(axes[1].get_xlim(), (0, 3))
        np.testing.assert_allclose(np.diff(axes[0].get_xticks()), .5)
        data.signals["motor/current[2]"][1][:] = -100
        np.testing.assert_array_equal(lines[0].get_ydata(), [1, 2, 3])
        fig.canvas.draw()

    def test_multicolumn_grid_preserves_panels_and_bottom_labels(self):
        data = SyntheticData()
        keys = ['wm_ref', 'tau_m', 'motor/current', 'wm_ref', 'tau_m',
                'wm_ref', 'tau_m', 'wm_ref', 'tau_m', 'wm_ref']
        for count, columns in [(3, 1), (3, 2), (6, 2), (5, 3), (2, 3), (10, 3), (2, 8)]:
            with self.subTest(count=count, columns=columns):
                groups = [[key] for key in keys[:count]]
                fig, axes = core.plot_signals(
                    data, groups, columns=columns, t_offset=2, t_span=(0, 3), x_step=.5,
                    units={'wm_ref': 'rad/s', 'tau_m': 'Nm'}, per_unit=True,
                    transforms={'custom:1:tau_m': {'scale': '2', 'offset': '0'}})
                self.assertEqual(axes.ndim, 1)
                self.assertEqual(list(axes), fig.axes)
                self.assertEqual(len(axes), count)
                fig.canvas.draw()
                for index, (ax, group) in enumerate(zip(axes, groups)):
                    bottom = index + columns >= count
                    self.assertEqual(ax.get_xlabel(), 'Time (s)' if bottom else '')
                    if bottom:
                        self.assertTrue(ax.xaxis.label.get_visible())
                    self.assertEqual(bool(ax.get_xticklabels()), bottom)
                    self.assertTrue(ax.get_shared_x_axes().joined(ax, axes[0]))
                    self.assertEqual(ax.get_xlim(), (0, 3))
                    np.testing.assert_allclose(np.diff(ax.get_xticks()), .5)
                    self.assertEqual(ax.get_subplotspec().rowspan.start, index // columns)
                    self.assertEqual(ax.get_subplotspec().colspan.start, index % columns)
                    expected_keys = [s['key'] for s in data.signal_info(group[0])]
                    self.assertEqual([line.get_label().split(' (')[0] for line in ax.lines], expected_keys)
                    self.assertIsNotNone(ax.get_legend())
                    for line, key in zip(ax.lines, expected_keys):
                        np.testing.assert_array_equal(line.get_xdata(), data.time(key) - 2)
                np.testing.assert_allclose(axes[1].lines[0].get_ydata(),
                                           2 * data['tau_m'] / core.DEFAULT_BASE.tau)
                self.assertEqual(axes[1].lines[0].get_label(), 'tau_m (p.u.)')
                np.testing.assert_allclose(axes[0].lines[0].get_ydata(),
                                           data['wm_ref'] / core.DEFAULT_BASE.w)
                if count > 2:
                    for line, key in zip(axes[2].lines, ['motor/current[2]', 'motor/current[7]']):
                        np.testing.assert_array_equal(line.get_ydata(), data[key])
                plt.close(fig)

    def test_invalid_columns_rejected_before_figure_creation(self):
        with patch.object(core.plt, 'subplots') as subplots:
            for columns in (True, False, 0, -1, 9, 1.0, '2', None):
                with self.subTest(columns=columns), self.assertRaises(ValueError):
                    core.plot_signals(SyntheticData(), [['wm_ref']], columns=columns)
            subplots.assert_not_called()

    def test_subplot_creation_failure_does_not_leak_figure(self):
        existing = plt.figure()

        def fail(*args, **kwargs):
            plt.figure()
            raise RuntimeError('subplot creation failed')

        with patch.object(core.plt, 'subplots', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'subplot creation failed'):
                core.plot_signals(SyntheticData(), [['wm_ref']], columns=2)
        self.assertEqual(plt.get_fignums(), [existing.number])

    def test_custom_affine_operations_are_per_subplot_before_normalizing(self):
        data = SyntheticData()
        transforms = {'custom:0:wm_ref': {'scale': 'n_p', 'offset': 'w_base'},
                      'custom:1:wm_ref': {'scale': '-1', 'offset': '0'}}
        fig, axes = core.plot_signals(data, [['wm_ref'], ['wm_ref']], per_unit=True,
                                     units={'wm_ref': 'rad/s'}, transforms=transforms)
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(),
                                   (data['wm_ref']*core.DEFAULT_BASE.n_p + core.DEFAULT_BASE.w)/core.DEFAULT_BASE.w)
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), -data['wm_ref']/core.DEFAULT_BASE.w)
        np.testing.assert_array_equal(data['wm_ref'], [0., 0., 2., 2.])
        plt.close(fig)
        with patch.object(core.plt, 'subplots') as create, self.assertRaises(ValueError):
            core.plot_signals(data, [['wm_ref']],
                              transforms={'custom:0:wm_ref': {'scale': '1/0', 'offset': '0'}})
        create.assert_not_called()

    def test_custom_units_and_already_pu_inputs(self):
        data = SyntheticData()
        base = core.DEFAULT_BASE
        for unit, reference in [('A', base.i), ('V', base.u), ('rad/s', base.w),
                                ('Nm', base.tau), ('Wb', base.psi)]:
            with self.subTest(unit=unit):
                data.signals['physical'] = (data.t.copy(), np.full(4, 2*reference))
                data.signals['normalized'] = (data.t.copy(), np.full(4, 2.))
                units = {'physical': unit, 'normalized': f'p.u. ({unit})'}
                for per_unit in (True, False):
                    fig, axes = core.plot_signals(data, [['physical', 'normalized']],
                                                 units=units, base=base, per_unit=per_unit)
                    expected = 2. if per_unit else 2*reference
                    for line in axes[0].lines:
                        np.testing.assert_allclose(line.get_ydata(), expected)
                        self.assertIn('(p.u.)' if per_unit else f'({unit})', line.get_label())
                    plt.close(fig)
        fig, axes = core.plot_signals(data, [['wm_ref']], units={'wm_ref': 'raw'}, per_unit=True)
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), data['wm_ref'])
        self.assertIn('(raw)', axes[0].lines[0].get_label())
        with patch.object(core.plt, 'subplots') as subplots, self.assertRaises(ValueError):
            core.plot_signals(data, [['wm_ref']], units={'wm_ref': 'unknown'})
        subplots.assert_not_called()

    def test_trace_units_override_component_and_group_units(self):
        data = SyntheticData()
        units = {'custom:0:motor/current': 'V', 'motor/current[2]': 'A',
                 'motor/current': 'Wb'}
        fig, axes = core.plot_signals(data, [['motor/current'], ['motor/current']],
                                      units=units, per_unit=True)
        for line, key in zip(axes[0].lines, ['motor/current[2]', 'motor/current[7]']):
            np.testing.assert_allclose(line.get_ydata(), data[key] / core.DEFAULT_BASE.u)
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(),
                                   data['motor/current[2]'] / core.DEFAULT_BASE.i)
        np.testing.assert_allclose(axes[1].lines[1].get_ydata(),
                                   data['motor/current[7]'] / core.DEFAULT_BASE.psi)
        self.assertEqual([line.get_label() for line in axes[0].lines],
                         ['motor/current[2] (p.u.)', 'motor/current[7] (p.u.)'])
        plt.close(fig)

    def test_mechanical_speed_units_and_flux_alias(self):
        base = core.DEFAULT_BASE
        self.assertAlmostEqual(base.w_M, base.w / base.n_p)
        self.assertAlmostEqual(base.C, 1 / (base.Z * base.w))
        mechanical_speed = 1500 * 2 * np.pi / 60
        data = SyntheticData()
        data.signals['rpm'] = (data.t.copy(), np.full(4, 1500.))
        data.signals['mechanical'] = (data.t.copy(), np.full(4, mechanical_speed))
        data.signals['mechanical_pu'] = (data.t.copy(), np.full(4, mechanical_speed / base.w_M))
        data.signals['electrical'] = (data.t.copy(), np.full(4, mechanical_speed * base.n_p))
        data.signals['flux'] = (data.t.copy(), np.full(4, base.psi * 2))
        data.signals['flux_pu'] = (data.t.copy(), np.full(4, 2.))
        units = {'rpm': 'rpm (mechanical)', 'mechanical': 'rad/s (mechanical)',
                 'mechanical_pu': 'p.u. (rad/s mechanical)', 'electrical': 'rad/s',
                 'flux': 'Vs', 'flux_pu': 'p.u. (Vs)'}
        for per_unit in (False, True):
            fig, axes = core.plot_signals(data, [list(units)], units=units, per_unit=per_unit)
            lines = axes[0].lines
            speed = mechanical_speed / base.w_M if per_unit else mechanical_speed
            for line in lines[:3]:
                np.testing.assert_allclose(line.get_ydata(), speed)
                self.assertTrue(line.get_label().endswith('(p.u.)' if per_unit else '(rad/s)'))
            np.testing.assert_allclose(lines[3].get_ydata(), speed if per_unit else mechanical_speed * base.n_p)
            for line in lines[4:]:
                np.testing.assert_allclose(line.get_ydata(), 2 if per_unit else 2 * base.psi)
                self.assertTrue(line.get_label().endswith('(p.u.)' if per_unit else '(Vs)'))
            plt.close(fig)
        self.assertEqual(core.signal_scale('rad/s', base, False), (1., 'rad/s'))
        self.assertEqual(core.signal_scale('p.u. (rad/s)', base, False), (base.w, 'rad/s'))
        self.assertEqual(core.signal_scale('Wb', base, False), (1., 'Wb'))
        self.assertEqual(core.signal_scale('p.u. (Wb)', base, False), (base.psi, 'Wb'))

    def test_rated_torque_is_independent_of_other_units(self):
        data = SyntheticData()
        base = core.DEFAULT_BASE
        data.signals['torque_pu'] = (data.t.copy(), np.full(4, 2.))
        units = {'tau_m': 'Nm', 'torque_pu': 'p.u. (Nm)', 'wm_ref': 'raw'}
        for per_unit in (True, False):
            fig, axes = core.plot_signals(data, [['tau_m', 'torque_pu', 'wm_ref']],
                                          per_unit=per_unit, units=units, torque_mode='rated',
                                          transforms={'custom:0:tau_m': {'scale': 'n_p', 'offset': 'tau_nom'}})
            np.testing.assert_allclose(axes[0].lines[0].get_ydata(),
                                       (data['tau_m']*base.n_p + base.tau_nom)/base.tau_nom)
            np.testing.assert_allclose(axes[0].lines[1].get_ydata(), 2*base.tau/base.tau_nom)
            np.testing.assert_array_equal(axes[0].lines[2].get_ydata(), data['wm_ref'])
            self.assertEqual([line.get_label() for line in axes[0].lines],
                             ['tau_m (per rated)', 'torque_pu (per rated)', 'wm_ref (raw)'])
            plt.close(fig)

    def test_invalid_groups_and_window_rejected_before_figure_creation(self):
        invalid = [[], [[]], ["wm_ref"], [[""]], [[None]], [["absent"]]]
        with patch.object(core.plt, "subplots") as subplots:
            for groups in invalid:
                with self.subTest(groups=groups), self.assertRaises((ValueError, KeyError)):
                    core.plot_signals(SyntheticData(), groups)
            for options in ({"t_offset": np.nan}, {"t_span": (1, 0)},
                            {"t_span": (0, np.inf)}, {"x_step": 0}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    core.plot_signals(SyntheticData(), [["wm_ref"]], **options)
            subplots.assert_not_called()

    def test_invalid_samples_rejected_before_figure_creation(self):
        data = SyntheticData()
        for time, values in [([0, 0], [1, 2]), ([0, 1], [1]),
                             ([0, np.nan], [1, 2]), ([0, 1], ["a", "b"])]:
            data.signals["bad"] = (np.asarray(time), np.asarray(values))
            with patch.object(core.plt, "subplots") as subplots, self.assertRaises(ValueError):
                core.plot_signals(data, [["bad"]])
            subplots.assert_not_called()

    def test_render_failure_closes_only_created_figure(self):
        existing = plt.figure()
        with patch.object(Axes, "plot", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                core.plot_signals(SyntheticData(), [["wm_ref"]])
        self.assertEqual(plt.get_fignums(), [existing.number])


class MeasurementTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def load(self, data, **options):
        with patch.object(core, "LabDataMat", return_value=data), contextlib.redirect_stdout(io.StringIO()):
            return core.load_measurement(data.file_path, **options)

    def test_contexts_copy_base_arrays_windows_and_close_source(self):
        first_data, second_data = SyntheticData("first.mat"), SyntheticData("second.mat")
        first = self.load(first_data, t_offset=2, t_span=(0, 2), x_step=.5)
        second = self.load(second_data, t_offset=3, t_span=(0, 1), x_step=1)
        self.assertTrue(first_data.closed and second_data.closed)
        np.testing.assert_array_equal(first.tau_Mest, [10, 10, 20, 20])
        np.testing.assert_array_equal(core.plot_window(first)[0], [0, 1, 2, 3])
        np.testing.assert_array_equal(core.plot_window(second)[0], [-1, 0, 1, 2])
        first.tau_Mest[:] = -1
        first.base.tau_nom = -1
        first.t[:] = -1
        first.loaded.append("extra")
        np.testing.assert_array_equal(second.tau_Mest, [10, 10, 20, 20])
        np.testing.assert_array_equal(second.t, [2, 3, 4, 5])
        self.assertEqual(second.base.tau_nom, 29.2)
        self.assertEqual(core.DEFAULT_BASE.tau_nom, 29.2)
        self.assertNotIn("extra", second.loaded)
        self.assertEqual((second.t_offset, second.t_span, second.x_step), (3, (0, 1), 1))

    def test_signal_map_alternatives_shapes_and_reference_detection(self):
        class VectorData(SyntheticData):
            def __contains__(self, key):  # LabDataMat also finds a vector by its base key
                return key == 'motor/current' or super().__contains__(key)

        data = VectorData()
        data.signals['speed_ref'] = (data.t.copy(), np.array([0., 1., 1., 1.]))
        signal_map = {'w_m_ref': 'speed_ref', 'w_mest': ['absent', 'tau_m'],
                      'tau_Mest': 'motor/current', 'eps_inj': [], 'i_cs': 'motor/current'}
        output = io.StringIO()
        with patch.object(core, "LabDataMat", return_value=data), contextlib.redirect_stdout(output):
            context = core.load_measurement(data.file_path, signal_map=signal_map)
        self.assertTrue(data.closed)
        self.assertEqual(context.signal_map['w_m_ref'], 'speed_ref')
        self.assertEqual(context.signal_map['w_mest'], 'tau_m')
        self.assertIsNone(context.signal_map['tau_Mest'])  # a vector cannot feed a scalar role
        self.assertEqual(context.signal_map['i_cs'], 'motor/current')
        self.assertIsNone(context.signal_map['eps_inj'])
        np.testing.assert_array_equal(context.w_m_ref, [0, 1, 1, 1])
        np.testing.assert_array_equal(context.w_mest, [10, 10, 20, 20])
        self.assertTrue(np.all(np.isnan(context.tau_Mest)))
        self.assertTrue(np.all(np.isnan(context.eps_inj)))
        self.assertIn('motor/current for tau_Mest', output.getvalue())
        self.assertNotIn('eps_inj', output.getvalue())
        # The mapped speed reference changes at 3 s, the default wm_ref at 4 s.
        self.assertAlmostEqual(context.t_offset, 2.8)
        self.assertEqual(core.reference_thresholds(data), {'wm_ref': None})
        self.assertEqual(core.reference_thresholds(data, {'w_m_ref': [], 'tau_Lref': ['tau_m']}),
                         {'tau_m': None})
        default = self.load(SyntheticData())
        self.assertAlmostEqual(default.t_offset, 3.8)
        self.assertEqual(default.signal_map['w_m_ref'], 'wm_ref')
        for bad in ({'unknown': 'x'}, {'theta_m': 3}, ['wm_ref'], {'theta_m': ['a', 'a']}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.load(SyntheticData(), signal_map=bad)

    def test_role_keys_in_custom_panels_added_traces_and_spike_detection(self):
        data = SyntheticData()
        signal_map = {'w_m_ref': ['absent', 'wm_ref'], 'tau_Mest': 'tau_m'}
        fig, axes = core.plot_signals(data, [['@w_m_ref', 'tau_m'], ['@tau_Mest']], t_offset=2,
                                      per_unit=True, units={}, signal_map=signal_map)
        self.assertEqual([line.get_label() for line in axes[0].lines], ['wm_ref (p.u.)', 'tau_m (raw)'])
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), np.array([0., 0., 2., 2.])/core.DEFAULT_BASE.w)
        np.testing.assert_allclose(axes[1].lines[0].get_ydata(), np.array([10., 20.])/core.DEFAULT_BASE.tau)
        plt.close(fig)
        fig, axes = core.plot_signals(data, [['@tau_Mest']], units={'@tau_Mest': 'raw'}, per_unit=True)
        np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [10., 20.])
        plt.close(fig)
        before = set(plt.get_fignums())
        for groups, expected in [([['@eps_inj']], 'no signal mon'), ([['@w_m_ref']], 'not linked')]:
            with self.subTest(groups=groups), self.assertRaisesRegex(ValueError, expected):
                core.plot_signals(data, groups, signal_map={'w_m_ref': []})
        self.assertEqual(set(plt.get_fignums()), before)
        self.assertEqual(core.role_key(data, 'tau_m'), 'tau_m')
        self.assertEqual(core.role_key(data, '@tau_Mest'), 'tau_m')

        context = self.load(SyntheticData(), t_offset=2, t_span=(0, 3), extra_signals=['@tau_Mest'])
        self.assertEqual([item[0] for item in context.extra_signals['@tau_Mest']], ['tau_m'])
        fig, axes = core.plot_measurement(context, 'overview', per_unit=True,
                                          added_traces=[{'id': 'extra', 'panel': 2, 'key': '@tau_Mest'}])
        self.assertEqual(axes[2].lines[-1].get_label(), 'tau_m (p.u.)')
        np.testing.assert_allclose(axes[2].lines[-1].get_ydata(), np.array([10., 20.])/context.base.tau)
        plt.close(fig)
        spikes = SyntheticData()
        spikes.signals['measured'] = (spikes.t.copy(), np.array([0., 0., 5., 0.]))
        context = self.load(spikes, detector='spikes', spike_signal='@w_mest', spike_threshold=1.,
                            signal_map={'w_mest': 'measured'})
        self.assertAlmostEqual(context.t_offset, 3.8)
        with self.assertRaisesRegex(ValueError, '@eps_inj'):
            self.load(SyntheticData(), detector='spikes', spike_signal='@eps_inj', spike_threshold=1.)
        traces = core.subplot_traces('custom', [['@w_m_ref', 'tau_m']], signal_map=signal_map)
        self.assertEqual([trace['source'] for trace in traces], ['@w_m_ref → absent / wm_ref', 'tau_m'])
        self.assertEqual([trace['roles'] for trace in traces], [('w_m_ref',), ()])

    def test_trace_sources_and_roles_follow_signal_map(self):
        default = {trace['id']: trace for trace in core.subplot_traces('hfsi')}
        self.assertEqual(default['hfsi:4:2']['source'], 'tauLref / TL_ref')
        self.assertEqual(default['hfsi:3:1']['source'], 'thetam - theta_m_out')
        self.assertEqual(default['hfsi:1:0']['roles'], ('i_cs', 'theta_m'))
        mapped = {trace['id']: trace for trace in core.subplot_traces(
            'hfsi', signal_map={'theta_m': 'Observer/theta', 'eps_inj': []})}
        self.assertEqual(mapped['hfsi:1:0']['source'], 'i_c_abc + Observer/theta')
        self.assertEqual(mapped['hfsi:5:0']['source'], 'Observer/theta (observer)')
        self.assertEqual(mapped['hfsi:3:0']['source'], '(eps_inj not linked)')
        self.assertEqual({key: value['id'] for key, value in mapped.items()},
                         {key: value['id'] for key, value in default.items()})
        used = {role for kind in core.FIGURE_TYPES for trace in core.subplot_traces(kind)
                for role in trace['roles']}
        self.assertLessEqual(used, set(core.SIGNAL_ROLE_IDS))
        self.assertEqual(set(core.SIGNAL_ROLE_IDS) - used, {'i_s', 'i_ref', 'w_inj'})

    def test_extra_signals_are_detached_and_bad_sources_close_recording(self):
        data = SyntheticData()
        context = self.load(data, extra_signals=['motor/current'])
        self.assertTrue(data.closed)
        components = context.extra_signals['motor/current']
        self.assertEqual([item[0] for item in components], ['motor/current[2]', 'motor/current[7]'])
        data.signals['motor/current[2]'][0][:] = -1
        data.signals['motor/current[2]'][1][:] = -1
        np.testing.assert_array_equal(components[0][1], [2, 4, 5])
        np.testing.assert_array_equal(components[0][2], [1, 2, 3])
        gaps = SyntheticData()
        gaps.signals['gaps'] = (np.array([0., 1.]), np.array([1., np.nan]))
        context = self.load(gaps, extra_signals=['gaps'])
        self.assertTrue(np.isnan(context.extra_signals['gaps'][0][2][1]))
        for key, samples in [('missing', None), ('bad', ([0, 0], [1, 2])),
                             ('bad', ([0, 1], [1])),
                             ('bad', ([0, 1], [1, 2j]))]:
            data = SyntheticData()
            if samples:
                data.signals[key] = tuple(np.asarray(sample) for sample in samples)
            with self.subTest(key=key, samples=samples), self.assertRaisesRegex(ValueError, key):
                self.load(data, extra_signals=[key])
            self.assertTrue(data.closed)

    def test_added_signals_units_transforms_raster_and_preset_limits(self):
        context = self.load(SyntheticData(), t_offset=2, t_span=(0, 3),
                            extra_signals=['motor/current'])
        added = [{'id': 'extra:current', 'panel': 0, 'key': 'motor/current'}]
        transforms = {'extra:current': {'scale': 'n_p', 'offset': 'i_base'}}
        for per_unit, unit, factor, display in [(True, 'A', 1/context.base.i, 'p.u.'),
                                                (False, 'A', 1, 'A'),
                                                (None, 'p.u. (A)', context.base.i, 'A'),
                                                (True, 'raw', 1, 'raw')]:
            with self.subTest(per_unit=per_unit, unit=unit):
                baseline, baseline_axes = core.plot_measurement(context, 'hfsi', per_unit=per_unit)
                reference_ylim = baseline_axes[1, 0].get_ylim()
                reference_labels = [text.get_text() for text in baseline_axes[1, 0].get_legend().get_texts()]
                plt.close(baseline)
                fig, axes = core.plot_measurement(context, 'hfsi', per_unit=per_unit,
                                                  added_traces=added, transforms=transforms,
                                                  units={'motor/current': unit})
                ax = axes[0, 0]
                for line, times, values in zip(ax.lines[-2:], ([0, 2, 3], [0, 1]),
                                               ([1, 2, 3], [4, 5])):
                    np.testing.assert_array_equal(line.get_xdata(), times)
                    np.testing.assert_allclose(line.get_ydata(),
                                               (np.asarray(values)*context.base.n_p + context.base.i)*factor)
                    self.assertIn(f'({display})', line.get_label())
                    self.assertGreater(ax.get_ylim()[1], np.max(line.get_ydata()))
                self.assertEqual(ax.get_xlim(), (0, 3))
                self.assertEqual(axes[1, 0].get_ylim(), reference_ylim)
                self.assertEqual([text.get_text() for text in axes[1, 0].get_legend().get_texts()], reference_labels)
                self.assertIn(f'motor/current[2] ({display})',
                              [text.get_text() for text in ax.get_legend().get_texts()])
                fig.canvas.draw()
                plt.close(fig)

    def test_native_removal_keeps_transform_indices_legends_and_empty_panels(self):
        context = self.load(SyntheticData(), t_offset=2, t_span=(0, 3), extra_signals=['tau_m'])
        context.w_m_meas = np.full(4, context.base.w)
        traces = core.subplot_traces('hfsi')
        ids = [trace['id'] for trace in traces if trace['panel'] == 0]
        added = [{'id': 'extra:tau', 'panel': 0, 'key': 'tau_m'}]
        transforms = {ids[2]: {'scale': 'n_p', 'offset': 'w_base'}}
        fig, axes = core.plot_measurement(context, 'hfsi', per_unit=True,
                                          disabled_traces=ids[:2], added_traces=added,
                                          transforms=transforms)
        ax = axes[0, 0]
        self.assertEqual(len(ax.lines), 2)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), context.base.n_p + 1)
        self.assertEqual([text.get_text() for text in ax.get_legend().get_texts()],
                         [line.get_label() for line in ax.lines])
        self.assertEqual(ax.get_legend()._loc, 8)  # preset lower center
        fig.canvas.draw()
        plt.close(fig)
        fig, axes = core.plot_measurement(context, 'hfsi', disabled_traces=ids)
        self.assertEqual(len(axes[0, 0].lines), 0)
        self.assertIsNone(axes[0, 0].get_legend())
        self.assertEqual(axes[0, 0].get_ylim(), (-.2, .2))
        plt.close(fig)

    def test_added_validation_does_not_leak_figures(self):
        context = self.load(SyntheticData(), extra_signals=['tau_m'])
        existing = plt.figure()
        for added in [[{'id': 'x', 'panel': 9, 'key': 'tau_m'}],
                      [{'id': 'x', 'panel': True, 'key': 'tau_m'}],
                      [{'id': 'x', 'panel': 0, 'key': 'absent'}],
                      [{'id': 'x', 'panel': 0, 'key': 'tau_m'},
                       {'id': 'x', 'panel': 1, 'key': 'tau_m'}]]:
            with self.subTest(added=added), self.assertRaises(ValueError):
                core.plot_measurement(context, 'hfsi', added_traces=added)
        with self.assertRaises(ValueError):
            core.plot_measurement(context, 'hfsi', disabled_traces=['bad'])
        with self.assertRaises(ValueError):
            core.plot_measurement(context, 'hfsi', added_traces=[{'id': 'x', 'panel': 0, 'key': 'tau_m'}],
                                  transforms={'x': {'scale': '1/0', 'offset': '0'}})
        self.assertEqual(plt.get_fignums(), [existing.number])

    def test_added_long_trace_retains_units_after_zoom_downsampling(self):
        data = SyntheticData()
        data.signals['long'] = (np.linspace(2, 5, 6000), np.full(6000, 2.))
        context = self.load(data, t_offset=2, t_span=(0, 3), extra_signals=['long'])
        fig, axes = core.plot_measurement(context, 'injection-overview', per_unit=True,
            added_traces=[{'id': 'added', 'panel': 4, 'key': 'long'}],
            units={'long': 'Nm'}, torque_mode='rated',
            transforms={'added': {'scale': 'n_p', 'offset': 'tau_nom'}})
        axes[4].set_xlim(.5, 1.)
        fig.canvas.draw()
        line = axes[4].lines[-1]
        np.testing.assert_allclose(line.get_ydata(), (2*context.base.n_p + context.base.tau_nom)/context.base.tau_nom)
        self.assertTrue(np.all((line.get_xdata() >= .499) & (line.get_xdata() <= 1.001)))
        self.assertEqual(line.get_label(), 'long (per rated)')
        plt.close(fig)

    def test_optional_full_resolution_keeps_out_of_window_values(self):
        data = SyntheticData()
        values = np.ones(6000)
        values[-1] = 100
        data.signals['long'] = (np.linspace(2, 5, 6000), values)
        context = self.load(data, t_offset=2, t_span=(0, .5), extra_signals=['long'])
        fig, axes = core.plot_measurement(context, 'injection-overview',
                                          added_traces=[{'id': 'added', 'panel': 4, 'key': 'long'}],
                                          downsample_output=False)
        try:
            self.assertEqual(len(axes[4].lines[-1].get_ydata()), 6000)
            self.assertEqual(axes[4].lines[-1].get_ydata()[-1], 100)
        finally:
            plt.close(fig)

    def test_rated_torque_for_presets_and_added_signals(self):
        context = self.load(SyntheticData(), t_offset=2, t_span=(0, 3), extra_signals=['tau_m'])
        context.base.tau_nom = 20.
        context.tau_Lref = np.full(4, .5)
        added = [{'id': 'extra:tau', 'panel': 2, 'key': 'tau_m'}]
        for kind in core.FIGURE_TYPES:
            baseline, axes = core.plot_measurement(context, kind, per_unit=True)
            baseline_torque = axes[2, 0] if kind == 'hfsi' else axes[2]
            ydata = [line.get_ydata().copy() for line in baseline_torque.lines]
            limits = baseline_torque.get_ylim()
            plt.close(baseline)
            for per_unit in (True, False, None):
                fig, axes = core.plot_measurement(context, kind, per_unit=per_unit,
                                                  torque_mode='rated')
                torque = axes[2, 0] if kind == 'hfsi' else axes[2]
                for index, (line, values) in enumerate(zip(torque.lines, ydata)):
                    if kind == 'hfsi' and index == 2:
                        expected = context.tau_Lref*context.base.tau/context.base.tau_nom
                    else:
                        expected = values*context.base.tau/context.base.tau_nom
                    np.testing.assert_allclose(line.get_ydata(), expected)
                np.testing.assert_allclose(torque.get_ylim(), np.asarray(limits)*context.base.tau/context.base.tau_nom)
                self.assertIn('per rated', torque.get_ylabel())
                speed = axes[0, 0] if kind == 'hfsi' else axes[0]
                if per_unit is False:
                    self.assertIn('rad/s', speed.get_ylabel())
                plt.close(fig)
        for per_unit in (True, False, None):
            for unit, multiplier in [('Nm', 1), ('p.u. (Nm)', context.base.tau), ('raw', None)]:
                fig, axes = core.plot_measurement(context, 'overview', per_unit=per_unit,
                    torque_mode='rated', added_traces=added, units={'tau_m': unit})
                expected = context.extra_signals['tau_m'][0][2]
                if multiplier is not None:
                    expected = expected * multiplier / context.base.tau_nom
                np.testing.assert_allclose(axes[2].lines[-1].get_ydata(), expected)
                self.assertIn('(raw)' if unit == 'raw' else '(per rated)', axes[2].lines[-1].get_label())
                plt.close(fig)
        fig, axes = core.plot_measurement(context, 'hfsi', torque_mode='rated',
            transforms={'hfsi:4:2': {'scale': 'n_p', 'offset': '1'}})
        np.testing.assert_allclose(axes[2, 0].lines[2].get_ydata(),
            (context.tau_Lref*context.base.n_p + 1)*context.base.tau/context.base.tau_nom)
        plt.close(fig)
        with self.assertRaises(ValueError):
            core.plot_measurement(context, 'hfsi', torque_mode='invalid')

    def test_preset_measured_speed_gain_replaces_legacy_conversion(self):
        context = self.load(SyntheticData(), t_offset=2, t_span=(0, 3))
        context.w_m_meas = np.full(4, context.base.w)
        original = context.w_m_meas.copy()
        traces = core.subplot_traces('hfsi')
        measured = next(trace for trace in traces if trace['source'] == 'wm_meas')
        self.assertEqual(measured['scale'], '1/2')
        defaults, axes = core.plot_measurement(context, 'hfsi', per_unit=True)
        np.testing.assert_allclose(axes[0, 0].lines[2].get_ydata(), .5)
        plt.close(defaults)
        transforms = {measured['id']: {'scale': 'n_p', 'offset': 'w_base'}}
        for per_unit in (True, False):
            with self.subTest(per_unit=per_unit):
                fig, axes = core.plot_measurement(context, 'hfsi', per_unit=per_unit, transforms=transforms)
                expected = context.base.n_p + 1
                if not per_unit:
                    expected *= context.base.w
                np.testing.assert_allclose(axes[0, 0].lines[2].get_ydata(), expected)
                self.assertLess(axes[0, 0].get_ylim()[0], expected)
                self.assertGreater(axes[0, 0].get_ylim()[1], expected)
                np.testing.assert_allclose(axes[0, 0].lines[0].get_ydata(),
                                           context.w_m_ref/context.base.w if per_unit else context.w_m_ref)
                plt.close(fig)
        np.testing.assert_array_equal(context.w_m_meas, original)
        for kind in core.FIGURE_TYPES:
            traces = core.subplot_traces(kind)
            fig, axes = core.plot_measurement(context, kind, transforms={
                traces[0]['id']: {'scale': '1', 'offset': '0'}})
            self.assertEqual(len(traces), sum(len(axis.lines) for axis in axes.flat))
            plt.close(fig)
        before = set(plt.get_fignums())
        with self.assertRaises(ValueError):
            core.plot_measurement(context, 'hfsi', transforms={measured['id']: {'scale': '1/0', 'offset': '0'}})
        self.assertEqual(set(plt.get_fignums()), before)

    def test_preset_checkbox_converts_values_labels_and_limits(self):
        context = self.load(SyntheticData(), t_offset=2, t_span=(0, 3))
        for name in ('psi_sref', 'psi_sest', 'u_inj', 'u_s', 'i_c'):
            setattr(context, name, np.ones(4))
        for kind in core.FIGURE_TYPES:
            with self.subTest(kind=kind):
                pu_fig, pu_axes = core.plot_measurement(context, kind, per_unit=True)
                si_fig, si_axes = core.plot_measurement(context, kind, per_unit=False)
                if kind == 'hfsi':
                    pairs = [(0, 0, context.base.w, 'rad/s'),
                             (1, 0, context.base.psi, 'Wb'),
                             (2, 0, context.base.tau, 'Nm'),
                             (0, 1, context.base.i, 'A')]
                    np.testing.assert_allclose(pu_axes[2, 0].lines[1].get_ydata(),
                                               context.tau_Mest/context.base.tau)
                    self.assertEqual(pu_axes[1, 1].get_ylabel(), si_axes[1, 1].get_ylabel())
                    np.testing.assert_array_equal(pu_axes[2, 1].lines[0].get_ydata(),
                                                  si_axes[2, 1].lines[0].get_ydata())
                else:
                    pairs = [(0, None, context.base.w, 'rad/s'),
                             (1, None, context.base.psi, 'Wb'),
                             (2, None, context.base.tau, 'Nm'),
                             (3, None, context.base.i, 'A')]
                    if kind != 'presentation':
                        pairs.append((4, None, context.base.u, 'V'))
                for row, col, reference, unit in pairs:
                    pu = pu_axes[row] if col is None else pu_axes[row, col]
                    si = si_axes[row] if col is None else si_axes[row, col]
                    self.assertIn('p.u.', pu.get_ylabel())
                    self.assertIn(unit, si.get_ylabel())
                    for pu_line, si_line in zip(pu.lines, si.lines):
                        np.testing.assert_allclose(si_line.get_ydata(),
                                                   np.asarray(pu_line.get_ydata())*reference)
                    np.testing.assert_allclose(si.get_ylim(), np.asarray(pu.get_ylim())*reference)
                pu_fig.canvas.draw()
                si_fig.canvas.draw()
                plt.close(pu_fig)
                plt.close(si_fig)

    def test_zoom_retains_converted_injection_units(self):
        context = self.load(SyntheticData(), t_offset=2, t_span=(0, 3))
        for name, value in vars(context).items():
            if isinstance(value, np.ndarray):
                setattr(context, name, np.tile(value, 1500))
        context.t = np.linspace(2, 5, 6000)
        context.u_inj = np.full(6000, context.base.u)
        fig, axes = core.plot_measurement(context, 'injection-overview', per_unit=True,
                                          transforms={'injection-overview:4:0':
                                                      {'scale': 'n_p', 'offset': 'u_base'}})
        axes[-1].set_xlim(.5, 1.)
        fig.canvas.draw()
        np.testing.assert_allclose(axes[-1].lines[0].get_ydata(), context.base.n_p + 1.)
        self.assertIn('p.u.', axes[-1].get_ylabel())

    def test_dispatch_all_notebook_plots_and_preserve_limits_units(self):
        context = self.load(SyntheticData(), t_offset=2, t_span=(0, 3))
        for kind in core.FIGURE_TYPES:
            with self.subTest(kind=kind):
                fig, axes = core.plot_measurement(context, kind)
                self.assertEqual(axes.flat[-1].get_xlim(), (0, 3))
                if kind == "hfsi":
                    self.assertEqual(axes.shape, (3, 2))
                    self.assertEqual(axes[2, 0].get_ylim(), (-3, 3))
                    self.assertIn("per rated", axes[2, 0].get_ylabel())
                    np.testing.assert_allclose(axes[2, 0].lines[1].get_ydata(),
                                               context.tau_Mest/context.base.tau_nom)
                elif kind == "injection-overview":
                    self.assertEqual(axes[-1].get_ylabel(), "Injection Voltages (V)")
                elif kind == "overview":
                    np.testing.assert_allclose(axes[2].lines[1].get_ydata(),
                                               context.tau_Mest/context.base.tau)
                fig.canvas.draw()
                plt.close(fig)
        with self.assertRaises(ValueError):
            core.plot_measurement(context, "unknown")


if __name__ == "__main__":
    unittest.main()
