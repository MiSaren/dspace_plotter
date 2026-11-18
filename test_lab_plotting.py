import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from lab_plotting import LabDataMat, detect_spikes, detect_transients, measurement_files, transient_window


class ReferenceData:
    def __init__(self, t, **references):
        self.t = np.asarray(t)
        self.references = references

    def __contains__(self, key):
        return key in self.references

    def __getitem__(self, key):
        return np.asarray(self.references[key][1])

    def time(self, key):
        return np.asarray(self.references[key][0])


class TransientWindowTests(unittest.TestCase):
    def window(self, data, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return transient_window(data, **kwargs)

    def test_steps_and_whole_ramp_on_separate_rasters(self):
        data = ReferenceData(
            np.linspace(0, 12, 121),
            wm_ref=([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 0, 0, 1, 2, 3, 3, 3, 3]),
            tauLref=([0, 1, 4, 7, 10, 12], [0, 1, 1, 1, 0, 0]),
        )
        self.assertEqual(detect_transients(*data.references['wm_ref']), [(2.0, 5.0)])
        origin, limits = self.window(data)
        self.assertAlmostEqual(origin, 0.8)
        self.assertAlmostEqual(limits[1], 9.4)
        origin, limits = self.window(data, pre_transient=0.5, duration=4)
        self.assertEqual((origin, limits), (0.5, (0.0, 4.0)))

    def test_threshold_selects_signals_and_ignores_small_changes(self):
        data = ReferenceData(
            range(6),
            wm_ref=(range(6), [0, 0, 1e-9, 2, 2, 2]),
            tauLref=(range(6), [0, 1, 1, 1, 1, 1]),
        )
        origin, limits = self.window(data, thresholds={'wm_ref': None})
        self.assertAlmostEqual(origin, 2.8)
        self.assertAlmostEqual(limits[1], 0.4)
        self.assertEqual(detect_transients(range(6), data['wm_ref'], threshold=3), [])

    def test_missing_reference_uses_legacy_load_reference(self):
        data = ReferenceData(range(6), TL_ref=(range(6), [0, 0, 1, 1, 1, 1]))
        origin, limits = self.window(data)
        self.assertAlmostEqual(origin, 1.8)
        self.assertAlmostEqual(limits[1], 0.4)

    def test_no_transient_falls_back_to_recording_bounds(self):
        for references in ({}, {'wm_ref': ([2, 3, 4, 5], [0, 0, 0, 0])},
                           {'wm_ref': ([2, 3, 4, 5], [np.nan]*4)}):
            with self.subTest(references=references):
                data = ReferenceData([2, 3, 4, 5], **references)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(transient_window(data), (2.0, (0.0, 3.0)))
                self.assertIn('No reference transients', output.getvalue())
                self.assertEqual(self.window(data, duration=1), (2.0, (0.0, 1.0)))

    def test_nan_gaps_do_not_create_changes(self):
        self.assertEqual(detect_transients(range(5), [0, 0, np.nan, 2, 2]), [])

    def test_margins_keep_origin_and_do_not_extrapolate(self):
        data = ReferenceData([0, .1, .2, .3], wm_ref=([0, .1, .2, .3], [0, 1, 1, 0]))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            origin, limits = transient_window(data)
        self.assertAlmostEqual(origin, -.1)
        self.assertAlmostEqual(limits[1], .6)
        self.assertIn('margins remain empty', output.getvalue())
        self.assertEqual(self.window(data, t_offset=0, t_span=[0, .2]), (0., (0., .2)))

    def test_invalid_window_parameters(self):
        data = ReferenceData(range(5))
        for kwargs in ({'pre_transient': -1}, {'post_transient': np.nan},
                       {'duration': 0}, {'duration': -1}, {'duration': np.inf},
                       {'t_offset': np.nan}, {'t_span': [2, 1]},
                       {'t_span': [0, 1], 'duration': 1},
                       {'thresholds': {'wm_ref': -1}},
                       {'t_offset': 10, 'duration': 2}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.window(data, **kwargs)


class MeasurementSpikeTests(unittest.TestCase):
    def test_spikes_use_raw_median_and_own_raster_not_reference_changes(self):
        data = ReferenceData(
            np.arange(11),
            measured=([0, 1, 4, 4.5, 6, 8, 9, 10], [10, 10, 14, 14, 10, 16, 10, 10]),
            wm_ref=(np.arange(11), [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]))
        self.assertEqual(detect_spikes(*data.references['measured'], threshold=3),
                         [(4., 4.5), (8., 8.)])
        with contextlib.redirect_stdout(io.StringIO()):
            reference_origin, _ = transient_window(data)
            origin, limits = transient_window(data, detector='spikes', spike_signal='measured', spike_threshold=3)
            fixed = transient_window(data, detector='spikes', spike_signal='measured', spike_threshold=3,
                                     pre_transient=.5, duration=2)
        self.assertAlmostEqual(reference_origin, 1.8)
        self.assertAlmostEqual(origin, 3.8)
        self.assertAlmostEqual(limits[1], 4.4)
        self.assertEqual(fixed, (3.5, (0., 2.)))
        self.assertEqual(detect_spikes(range(7), [0, 3, np.nan, 3, 0, 0, 0], 2),
                         [(1., 1.), (3., 3.)])

    def test_no_spikes_falls_back_without_switching_detector(self):
        for values in ([10, 10, 10, 10], [np.nan]*4):
            data = ReferenceData(range(4), measured=(range(4), values),
                                 wm_ref=(range(4), [0, 0, 1, 1]))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                result = transient_window(data, detector='spikes', spike_signal='measured', spike_threshold=1)
            self.assertEqual(result, (0., (0., 3.)))
            self.assertIn('No measurement spikes', output.getvalue())
        data = ReferenceData(range(4), measured=(range(4), [0, 0, 3, 0]))
        for options in ({'spike_signal': 'absent', 'spike_threshold': 1},
                        {'spike_signal': 'measured', 'spike_threshold': None},
                        {'spike_signal': 'measured', 'spike_threshold': 0},
                        {'spike_signal': 'measured', 'spike_threshold': -1},
                        {'spike_signal': 'measured', 'spike_threshold': np.nan},
                        {'spike_signal': '', 'spike_threshold': 1}):
            with self.subTest(options=options), self.assertRaises((ValueError, KeyError)):
                transient_window(data, detector='spikes', **options)
        with self.assertRaises(ValueError):
            transient_window(data, detector='other')
        for values in ([[1, 2], [3, 4]], [1+1j, 2+2j], [1]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                detect_spikes([0, 1], values, 1)


class MeasurementFilesTests(unittest.TestCase):
    def test_malformed_measurement_closes_file_after_loading_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'malformed.mat'
            with h5py.File(source, 'w') as handle:
                handle.create_group('recording')
            opened = h5py.File(source, 'r')
            with patch('lab_plotting.h5py.File', return_value=opened):
                with self.assertRaises(KeyError):
                    LabDataMat(source)
            self.assertFalse(opened.id.valid)

    def test_single_file_and_nonrecursive_case_insensitive_batch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('b.mat', 'A.MAT', 'ignore.pdf'):
                (root / name).touch()
            (root / 'nested').mkdir()
            (root / 'nested' / 'hidden.mat').touch()
            source = root / 'b.mat'
            self.assertEqual(measurement_files(source.with_suffix('')), [source.resolve()])
            self.assertEqual(measurement_files(source, plot_all=True),
                             [(root / 'A.MAT').resolve(), source.resolve()])
            self.assertEqual(measurement_files(root / 'A.MAT'), [(root / 'A.MAT').resolve()])

    def test_data_directory_fallback_keeps_source_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'unique_measurement.mat'
            source.touch()
            with patch('lab_plotting.DATA_DIR', root):
                self.assertEqual(measurement_files('unique_measurement'), [source.resolve()])
            with self.assertRaises(FileNotFoundError):
                LabDataMat._resolve(root / 'missing', None)


class NotebookWorkflowTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        self.plt = plt
        self.existing_figures = set(plt.get_fignums())
        self.style = plt.rc_context()
        self.style.__enter__()
        self.addCleanup(self.style.__exit__, None, None, None)
        self.addCleanup(self.close_new_figures)
        self.namespace = {}
        notebook = Path(__file__).with_name('Lab_plotter.ipynb')
        for cell in json.loads(notebook.read_text(encoding='utf-8'))['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = '\n'.join(line for line in ''.join(cell['source']).splitlines()
                               if not line.startswith('%'))
            tree = ast.parse(source)
            # Load definitions without executing diagnostics or the save workflow.
            nodes = [node for node in tree.body if isinstance(
                node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(notebook), 'exec'),
                 self.namespace)
        base = self.namespace['BaseValues'](460, 8.8, 60, 5600, 29.2, 2)
        self.namespace.update(
            base=base, pre_transient=.2, post_transient=.2, duration=None,
            thresholds=None, t_offset=None, t_span=None, x_step=.1,
            plot_all=False, figure_types=('hfsi',), output_dir=None)
        self.namespace['set_latex_style'] = self.namespace['set_screen_style']

    def close_new_figures(self):
        for number in set(self.plt.get_fignums()) - self.existing_figures:
            self.plt.close(number)

    def measurement_context(self, source, motor_base, **options):
        t = np.linspace(0, 1, 11)
        values = dict(file_path=Path(source), base=motor_base, t=t,
                      t_offset=.2, t_span=(0., .6), x_step=options['x_step'])
        for key in ('w_m_ref', 'w_mest', 'w_m_meas', 'psi_sref', 'psi_sest',
                    'tau_Mref_lim', 'tau_Mest', 'tau_Lref', 'theta_err_M',
                    'eps_inj_M', 'theta_mest', 'theta_m'):
            values[key] = np.ones(t.size)
        values['i_c'] = np.ones(t.size, dtype=complex)
        return SimpleNamespace(**values)

    def test_single_saves_beside_source_and_preserves_interactive_figure(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'measurement.mat'
            source.touch()
            self.namespace.update(meas=source, load_measurement=self.measurement_context)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(self.plt, 'show') as show:
                results = self.namespace['run_plotting']()
            self.assertTrue(results[0]['success'], results)
            target = source.with_name('measurement_hfsi_2.pdf')
            self.assertEqual(results[0]['saved'], [target])
            self.assertGreater(target.stat().st_size, 1000)
            self.assertEqual(results[0]['t_offset'], .2)
            self.assertEqual(results[0]['t_span'], (0., .6))
            self.assertEqual(len(set(self.plt.get_fignums()) - self.existing_figures), 1)
            show.assert_called_once()

    def test_batch_continues_after_failure_and_closes_only_its_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('a.mat', 'b.mat', 'c.mat'):
                (root / name).touch()
            sentinel = self.plt.figure()

            def load(source, motor_base, **options):
                if source.name == 'b.mat':
                    self.plt.figure()
                    raise ValueError('invalid measurement')
                return self.measurement_context(source, motor_base, **options)

            self.namespace.update(meas=root / 'a.mat', plot_all=True,
                                  output_dir=root / 'figures', load_measurement=load)
            with contextlib.redirect_stdout(io.StringIO()), patch.object(self.plt, 'show') as show:
                results = self.namespace['run_plotting']()
            self.assertEqual([item['source'].name for item in results],
                             ['a.mat', 'b.mat', 'c.mat'])
            self.assertEqual([item['success'] for item in results], [True, False, True])
            self.assertIn('invalid measurement', results[1]['error'])
            self.assertEqual(set(self.plt.get_fignums()), self.existing_figures | {sentinel.number})
            for item in (results[0], results[2]):
                self.assertEqual(item['saved'][0].parent, root / 'figures')
                self.assertGreater(item['saved'][0].stat().st_size, 1000)
            show.assert_not_called()


if __name__ == '__main__':
    unittest.main()
