import json
import tempfile
import time
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from matplotlib.backend_bases import MouseEvent

import lab_plotter_app as app_module
import lab_plotter_config as config_module


class PlotterAppTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = app_module.PlotterApp(self.root)
        self.app.files.unbind('<<TreeviewSelect>>')
        self.addCleanup(self.app.close)
        self.figure_numbers = set(app_module.plt.get_fignums())

    def wait_idle(self):
        deadline = time.monotonic() + 10
        while self.app.busy or self.app.batch_active:
            self.root.update()
            if time.monotonic() > deadline:
                self.fail('GUI operation did not complete')
            time.sleep(.01)
        self.root.update()

    @staticmethod
    def small_figure(_payload, _options):
        figure, axes = app_module.plt.subplots()
        axes.plot([0, 1], [0, 1])
        return figure, axes

    def test_scenarios_profiles_reload_and_preserve_manual_motor_edits(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scenarios_path = root / 'scenarios.json'
            motors_path = root / 'motors.json'
            scenarios = json.loads(config_module.SCENARIO_PATH.read_text(encoding='utf-8'))
            motors = json.loads(config_module.MOTOR_PATH.read_text(encoding='utf-8'))
            scenarios['scenarios'].append({
                'id': 'variant', 'label': 'Variant', 'kind': 'hfsi', 'suffix': '_variant',
                'disabled_traces': ['hfsi:0:1'],
                'added_traces': [{'id': 'torque', 'panel': 0, 'key': 'tau_m'}],
                'transforms': {'hfsi:0:2': {'scale': 'n_p', 'offset': '0'},
                               'torque': {'scale': '2', 'offset': '1'}},
                'units': {'tau_m': 'Nm'}})
            motors['motors'].append({'id': 'other', 'name': 'Other', 'U_nom': 400, 'I_nom': 5,
                                     'f_nom': 50, 'P_nom': 2200, 'tau_nom': 10, 'n_p': 2})
            scenarios_path.write_text(json.dumps(scenarios), encoding='utf-8')
            motors_path.write_text(json.dumps(motors), encoding='utf-8')
            with patch.object(app_module, 'SCENARIO_PATH', scenarios_path), \
                    patch.object(app_module, 'MOTOR_PATH', motors_path), \
                    patch.object(app_module, 'load_scenarios', side_effect=lambda: config_module.load_scenarios(scenarios_path)), \
                    patch.object(app_module, 'load_motor_profiles', side_effect=lambda: config_module.load_motor_profiles(motors_path)):
                self.app.reload_configuration()
                self.assertIn('Variant [variant]', self.app.scenario_choices)
                self.app.kind.set('variant')
                options = self.app.options()
                self.assertEqual(options['kind'], 'hfsi')
                self.assertEqual(options['suffix'], '_variant')
                self.assertEqual(options['disabled_traces'], {'hfsi:0:1'})
                self.assertEqual(options['transforms']['hfsi:0:2']['scale'], 'n_p')
                added = options['added_traces'][0]
                self.assertEqual(options['transforms'][added['id']]['scale'], '2')
                self.assertEqual(options['units']['tau_m'], 'Nm')
                self.assertIn('[pois]', self.app.trace_tree.item('variant:0:1', 'text'))
                self.app.apply_profile('other')
                self.assertEqual(self.app.options()['base'].U_nom, 400)
                self.app.parameters['U_nom'].set('411')
                motors['motors'][1]['U_nom'] = 420
                motors_path.write_text(json.dumps(motors), encoding='utf-8')
                self.app.reload_configuration()
                self.assertEqual(self.app.parameters['U_nom'].get(), '411')
                self.assertIn('säilytettiin', self.app.status.get())
                self.app.apply_profile('other')
                self.assertEqual(self.app.options()['base'].U_nom, 420)
                before = self.app.scenarios.copy()
                scenarios_path.write_text('{invalid', encoding='utf-8')
                with patch.object(app_module.messagebox, 'showerror') as error:
                    self.app.reload_configuration()
                error.assert_called_once()
                self.assertEqual(self.app.scenarios, before)
                self.assertEqual(self.app.kind.get(), 'variant')

    def test_configured_scenario_multicolumn_units_and_automatic_reload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'scenarios.json'
            motors_path = root / 'motors.json'
            content = json.loads(config_module.SCENARIO_PATH.read_text(encoding='utf-8'))
            path.write_text(json.dumps(content), encoding='utf-8')
            motors_path.write_text(config_module.MOTOR_PATH.read_text(encoding='utf-8'), encoding='utf-8')
            with patch.object(app_module, 'SCENARIO_PATH', path), \
                    patch.object(app_module, 'MOTOR_PATH', motors_path), \
                    patch.object(app_module, 'load_scenarios', side_effect=lambda: config_module.load_scenarios(path)), \
                    patch.object(app_module, 'load_motor_profiles', side_effect=lambda: config_module.load_motor_profiles(motors_path)):
                self.app.reload_configuration()
                self.app.kind.set('speed-torque')
                options = self.app.options()
                self.assertEqual(options['kind'], 'configured')
                self.assertEqual(options['columns'], 2)
                self.assertEqual(options['groups'], [['wm_ref', 'w_m'], ['tau_m']])
                self.assertEqual(options['units']['custom:1:tau_m'], 'Nm')
                self.assertEqual(options['panel_titles'], ['Speed', 'Torque'])
                self.app.columns.set('3')
                self.app.kind.set('hfsi')
                self.app.kind.set('speed-torque')
                self.assertEqual(self.app.options()['columns'], 3)
                self.app.columns.set('9')
                with self.assertRaises(ValueError):
                    self.app.options()
                self.app.signal_units['tau_m'] = 'A'
                self.assertEqual(self.app.signal_units['tau_m'], 'A')
                scene = next(item for item in content['scenarios'] if item['id'] == 'speed-torque')
                scene['label'] = 'Updated scenario'
                scene['columns'] = 1
                scene['panels'][1]['signals'][0]['scale'] = 'n_p'
                path.write_text(json.dumps(content), encoding='utf-8')
                self.root.after_cancel(self.app._config_timer)
                self.app.poll_configuration()
                self.assertIn('Updated scenario [speed-torque]', self.app.scenario_choices)
                options = self.app.options()
                self.assertEqual(options['columns'], 1)
                self.assertEqual(options['units']['custom:1:tau_m'], 'Nm')
                self.assertEqual(options['transforms']['custom:1:tau_m']['scale'], 'n_p')
                self.app.trace_tree.selection_set('speed-torque:1:tau_m')
                self.app.trace_selected()
                self.app.edit_scale.set('4')
                self.app.edit_offset.set('0')
                self.app.assign_transform()
                self.app.reset_transform()
                self.assertEqual(self.app.options()['transforms']['custom:1:tau_m']['scale'], 'n_p')

    def test_configured_panel_removal_keeps_units_titles_and_transforms_aligned(self):
        self.app.kind.set('speed-torque')
        self.app.axis_limits['speed-torque:1'] = (-1., 1.)
        self.app.group_list.selection_set(0)
        self.app.remove_group()
        options = self.app.options()
        self.assertEqual(options['groups'], [['tau_m']])
        self.assertEqual(options['panel_titles'], ['Torque'])
        self.assertEqual(options['units']['custom:0:tau_m'], 'Nm')
        self.assertEqual(options['transforms']['custom:0:tau_m']['scale'], '1')
        self.assertEqual(options['y_limits'], {0: (-1., 1.)})
        self.assertEqual(self.app.kind.get(), 'speed-torque')
        self.app.signal_keys = ['wm_ref']
        self.app.filter_signals()
        self.app.signals.selection_set(0)
        self.app.add_group()
        options = self.app.options()
        self.assertEqual(options['groups'], [['tau_m'], ['wm_ref']])
        self.assertEqual(options['panel_titles'], ['Torque', 'Alikuvaaja 2'])
        self.assertEqual(self.app.kind.get(), 'speed-torque')

    def test_preview_header_tracks_rendered_file_not_selection_or_batch_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / 'first.mat', root / 'second.mat']
            for path in paths:
                path.touch()
            self.app.insert_paths(paths)
            self.app.select_all()
            items = self.app.files.get_children()
            self.app.files.focus(items[1])
            self.assertEqual(self.app.preview_source.get(), 'Esikatselu: ei kuvaa')
            with patch.object(app_module, 'load_plot_request', return_value=None), \
                    patch.object(app_module, 'render_request', side_effect=self.small_figure):
                self.app.preview()
                self.wait_idle()
                self.assertEqual(self.app.preview_source.get(), f'Esikatselu: {paths[1]}')
                self.app.files.focus(items[0])
                self.assertEqual(self.app.preview_source.get(), f'Esikatselu: {paths[1]}')
                self.app.export()
                self.wait_idle()
                self.assertEqual(self.app.preview_source.get(), f'Esikatselu: {paths[1]}')
                self.assertIn('2 tallennettu', self.app.status.get())
                self.app.preview()
                self.wait_idle()
                self.assertEqual(self.app.preview_source.get(), f'Esikatselu: {paths[0]}')
            self.app.files.focus(items[1])
            with patch.object(app_module, 'load_plot_request', side_effect=ValueError('read failed')), \
                    patch.object(app_module.messagebox, 'showerror'):
                self.app.preview()
                self.wait_idle()
            self.assertEqual(self.app.preview_source.get(), f'Esikatselu: {paths[0]}')
            previous = self.app.figure
            with patch.object(app_module, 'load_plot_request', return_value=None), \
                    patch.object(app_module, 'render_request', side_effect=self.small_figure), \
                    patch.object(app_module.FigureCanvasTkAgg, 'draw', side_effect=RuntimeError('draw failed')), \
                    patch.object(app_module.messagebox, 'showerror'):
                self.app.preview()
                self.wait_idle()
            self.assertIs(self.app.figure, previous)
            self.assertEqual(self.app.preview_source.get(), f'Esikatselu: {paths[0]}')
            self.assertEqual(set(app_module.plt.get_fignums()), self.figure_numbers | {previous.number})

    def test_custom_snapshot_expands_components_and_preserves_rasters(self):
        data = MagicMock()
        data.t = np.array([0., .1, .2, .3])
        values = {'vector[0]': np.array([1., 2.]), 'vector[1]': np.array([3., 4.])}
        times = {'vector[0]': np.array([0., .2]), 'vector[1]': np.array([.1, .3])}
        data.__getitem__.side_effect = values.__getitem__
        data.time.side_effect = times.__getitem__
        data.signal_info.return_value = [{'key': 'vector[0]'}, {'key': 'vector[1]'}]
        self.app.kind.set('custom')
        self.app.groups = [['vector']]
        options = self.app.options()
        with patch.object(app_module, 'LabDataMat') as reader, \
                patch.object(app_module, 'transient_window', return_value=(.1, (0., .2))):
            reader.return_value.__enter__.return_value = data
            payload = app_module.load_plot_request(Path('example.mat'), options)
            reader.return_value.__exit__.assert_called_once()
        values['vector[0]'][0] = 999
        figure, axes = app_module.render_request(payload, options)
        try:
            self.assertEqual([line.get_label() for line in axes[0].lines],
                             ['vector[0] (raw)', 'vector[1] (raw)'])
            np.testing.assert_allclose(axes[0].lines[0].get_xdata(), [-.1, .1])
            np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [1., 2.])
            np.testing.assert_allclose(axes[0].lines[1].get_xdata(), [0., .2])
        finally:
            app_module.plt.close(figure)

    def test_custom_groups_and_numeric_validation(self):
        self.app.signal_keys = ['wm_ref', 'w_m', 'tau_m']
        self.app.filter_signals()
        self.app.signals.selection_set(0, 1)
        self.app.add_group()
        self.assertEqual(self.app.groups, [['wm_ref', 'w_m']])
        self.app.parameters['duration'].set('0,5')
        options = self.app.options()
        self.assertEqual(options['window']['duration'], .5)
        self.app.groups[0].append('tau_m')
        self.assertEqual(options['groups'], [['wm_ref', 'w_m']])
        self.app.parameters['x_step'].set('-1')
        with self.assertRaises(ValueError):
            self.app.options()

    def test_drag_legend_is_preserved_for_redraw_export_and_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'one.mat'
            source.touch()
            self.app.insert_paths([source])

            def make_figure(_payload, options):
                fig, axes = app_module.plt.subplots(2, 1)
                for axis in axes:
                    axis.plot([0, 1], [0, 1], label='signal')
                    axis.legend(loc='upper right')
                app_module.apply_legend_positions(fig, options['legend_positions'])
                return fig, axes

            with patch.object(app_module, 'load_plot_request', return_value=None), \
                    patch.object(app_module, 'render_request', side_effect=make_figure) as render:
                self.app.preview()
                self.wait_idle()
                canvas = self.app.canvas
                legend = self.app.figure.axes[0].get_legend()
                before = legend.get_window_extent(canvas.get_renderer())
                x, y = before.x0 + before.width/2, before.y0 + before.height/2
                for name, dx, dy in [('button_press_event', 0, 0),
                                     ('motion_notify_event', -35, -30),
                                     ('button_release_event', -35, -30)]:
                    event = MouseEvent(name, canvas, x + dx, y + dy, button=1)
                    canvas.callbacks.process(name, event)
                self.assertIn('hfsi:0', self.app.legend_positions)
                self.assertNotIn('hfsi:1', self.app.legend_positions)
                position = self.app.legend_positions['hfsi:0']
                self.app.toolbar.zoom()
                moved = legend.get_window_extent(canvas.get_renderer())
                zx, zy = moved.x0 + moved.width/2, moved.y0 + moved.height/2
                for name, dx in [('button_press_event', 0), ('motion_notify_event', -10),
                                 ('button_release_event', -10)]:
                    canvas.callbacks.process(name, MouseEvent(name, canvas, zx + dx, zy, button=1))
                self.app.toolbar.zoom()
                self.assertEqual(self.app.legend_positions['hfsi:0'], position)
                self.app.preview()
                self.wait_idle()
                canvas = self.app.canvas
                canvas.draw()
                axis = self.app.figure.axes[0]
                bounds = axis.get_legend().get_window_extent(canvas.get_renderer())
                np.testing.assert_allclose(axis.transAxes.inverted().transform((bounds.x0, bounds.y0)),
                                           position, atol=1e-5)
                self.app.export()
                self.wait_idle()
                self.assertEqual(render.call_args.args[1]['legend_positions'], {0: position})
                self.assertTrue((Path(folder) / 'one_hfsi_2.pdf').exists())
                self.app.reset_legends()
                self.wait_idle()
                self.assertEqual(self.app.options()['legend_positions'], {})

    def test_minor_grid_and_individual_tick_controls(self):
        self.app.trace_tree.selection_set('panel:hfsi:0')
        self.app.trace_selected()
        self.app.y_major_step.set('0,5')
        self.app.y_minor_step.set('0,1')
        self.app.assign_y_ticks()
        self.app.trace_tree.selection_set('panel:hfsi:1')
        self.app.trace_selected()
        self.app.y_major_step.set('2')
        self.app.assign_y_ticks()
        self.app.trace_tree.selection_set('panel:hfsi:0')
        self.app.trace_selected()
        self.assertEqual(self.app.y_major_step.get(), '0.5')
        self.app.parameters['x_step'].set('.4')
        self.app.parameters['x_minor_step'].set('.1')
        self.app.minor_grid.set(True)
        self.app.minor_grid_changed()
        options = self.app.options()
        self.assertEqual(options['y_steps'], {0: (.5, .1), 1: (2., None)})
        self.assertEqual(options['x_step'], .4)
        self.assertEqual(options['x_minor_step'], .1)

        figure, axes = app_module.plt.subplots(2, 1, sharex=True)
        try:
            for axis in axes:
                axis.set_xlim(0, 2)
                axis.set_ylim(-2, 2)
                axis.xaxis.set_major_locator(app_module.MultipleLocator(options['x_step']))
                axis.grid(True, which='major')
            app_module.apply_grid_and_ticks(figure, minor_grid=True, x_minor_step=.1,
                                            y_steps=options['y_steps'])
            figure.canvas.draw()
            np.testing.assert_allclose(np.diff(axes[0].get_yticks()), .5)
            np.testing.assert_allclose(np.diff(axes[1].get_yticks()), 2.)
            np.testing.assert_allclose(np.diff(axes[0].get_xticks()), .4)
            self.assertTrue(any(tick.gridline.get_visible() for tick in axes[0].xaxis.get_minor_ticks()))
            self.assertTrue(any(tick.gridline.get_visible() for tick in axes[0].yaxis.get_minor_ticks()))
            np.testing.assert_allclose(axes[0].yaxis.get_minor_locator().tick_values(-.3, .3),
                                       np.arange(-.4, .5, .1), atol=1e-10)
            app_module.apply_grid_and_ticks(figure, minor_grid=False, x_minor_step=.1,
                                            y_steps=options['y_steps'])
            self.assertFalse(any(tick.gridline.get_visible() for tick in axes[0].xaxis.get_minor_ticks()))
            self.assertFalse(any(tick.gridline.get_visible() for tick in axes[0].yaxis.get_minor_ticks()))
            self.assertTrue(any(tick.gridline.get_visible() for tick in axes[0].yaxis.get_major_ticks()))
            app_module.apply_grid_and_ticks(figure, minor_grid=True, x_minor_step=None,
                                            y_steps={0: (.5, None)})
            self.assertTrue(any(tick.gridline.get_visible() for tick in axes[0].yaxis.get_minor_ticks()))
        finally:
            app_module.plt.close(figure)

        self.app.y_major_step.set('0')
        with patch.object(app_module.messagebox, 'showerror') as error:
            self.app.assign_y_ticks()
        error.assert_called_once()
        self.assertEqual(self.app.axis_tick_steps['hfsi:0'], (.5, .1))
        self.app.reset_y_ticks()
        self.assertNotIn('hfsi:0', self.app.axis_tick_steps)
        self.app.parameters['x_minor_step'].set('-1')
        with self.assertRaises(ValueError):
            self.app.options()
        self.app.kind.set('custom')
        self.assertFalse(self.app.minor_grid.get())
        self.app.kind.set('hfsi')
        self.assertTrue(self.app.minor_grid.get())

    def test_autoscale_uses_complete_visible_data_and_manual_bounds_win(self):
        fig, axes = app_module.plt.subplots(5, 1)
        try:
            axes[0].plot([0, 1, 20], [0, 2, 50])
            axes[0].set_xlim(0, 1)
            axes[0].set_ylim(0, 1)
            axes[0].plot([0, 1], [-99, -99], visible=False)
            axes[1].plot([0, 1, 10], [1, -3, 7])
            axes[1].set_xlim(0, 1)
            axes[1].set_ylim(-1, 1)
            axes[2].plot([0, 1], [0, 0])
            axes[3].plot([0, 1], [np.nan, np.inf])
            axes[3].set_ylim(-5, 5)
            axes[4].plot([0, 1], [5, 6])
            app_module.autoscale_y(fig)
            self.assertAlmostEqual(axes[0].get_ylim()[0], -.05)
            self.assertGreater(axes[0].get_ylim()[1], 50)
            self.assertLess(axes[1].get_ylim()[0], -3)
            self.assertGreater(axes[1].get_ylim()[1], 7)
            self.assertAlmostEqual(axes[2].get_ylim()[0], -.05)
            self.assertGreater(axes[2].get_ylim()[1], 0)
            self.assertEqual(axes[3].get_ylim(), (-5., 5.))
            app_module.apply_y_limits(fig, {0: (1., 4.), 4: (None, 8.)})
            self.assertEqual(axes[0].get_ylim(), (1., 4.))
            self.assertEqual(axes[4].get_ylim(), (-.05, 8.))
            self.assertEqual(axes[0].get_xlim(), (0., 1.))
        finally:
            app_module.plt.close(fig)

    def test_autoscale_limits_survive_preview_downsampling_and_zoom(self):
        fig, axes = app_module.plt.subplots(2, 1, sharex=True)
        try:
            x = np.linspace(0, 10, 6000)
            positive = np.ones(x.size)
            signed = np.ones(x.size)
            positive[-1], signed[-1] = 100, -50
            axes[0].plot(x, positive)
            axes[1].plot(x, signed)
            axes[0].set_xlim(0, 1)
            app_module.autoscale_y(fig)
            expected = [axis.get_ylim() for axis in axes]
            self.assertEqual(expected[0][0], -.05)
            self.assertGreater(expected[0][1], 100)
            self.assertLess(expected[1][0], -50)
            app_module.downsample(fig)
            fig.canvas.draw()
            axes[0].set_xlim(.1, .2)
            fig.canvas.draw()
            for axis, limits in zip(axes, expected):
                np.testing.assert_allclose(axis.get_ylim(), limits)
        finally:
            app_module.plt.close(fig)

    def test_render_preserves_full_recording_for_autoscaling(self):
        options = self.app.options()
        options['y_limits'] = {1: (-2., 3.)}

        def preset(_payload, _kind, **kwargs):
            self.assertFalse(kwargs['downsample_output'])
            figure, axes = app_module.plt.subplots(2, 1, sharex=True)
            axes[0].plot([0, 1, 10], [0, 1, 100])
            axes[1].plot([0, 1, 10], [1, 2, 50])
            axes[0].set_xlim(0, 1)
            return figure, axes

        with patch.object(app_module, 'plot_measurement', side_effect=preset):
            figure, axes = app_module.render_request(None, options)
        try:
            self.assertEqual(axes[0].get_ylim()[0], -.05)
            self.assertGreater(axes[0].get_ylim()[1], 100)
            self.assertEqual(axes[1].get_ylim(), (-2., 3.))
            self.assertEqual(axes[0].get_xlim(), (0., 1.))
        finally:
            app_module.plt.close(figure)

    def test_manual_axis_bounds_palette_and_legacy_latex_style(self):
        self.app.trace_tree.selection_set('panel:hfsi:0')
        self.app.trace_selected()
        self.app.y_min.set('-0,5')
        self.app.y_max.set('0,5')
        self.app.assign_y_limits()
        self.app.palette.set('okabe_ito')
        options = self.app.options()

        def preset(_payload, _kind, **_options):
            figure, axes = app_module.plt.subplots(3, 2)
            for axis in axes.flat:
                axis.plot([0, 1], [0, 1], color='C1')
                axis.set_ylim(-2, 2)
            return figure, axes

        with patch.object(app_module, 'plot_measurement', side_effect=preset):
            figure, axes = app_module.render_request(None, options)
        try:
            self.assertEqual(axes[0, 0].get_ylim(), (-.5, .5))
            np.testing.assert_allclose(axes[0, 1].get_ylim(), (-.05, 1.05))
            expected = app_module.to_rgba(app_module.PALETTES['okabe_ito'][1])
            self.assertEqual(axes[0, 0].lines[0].get_color(), expected)
            figure.canvas.draw()
            with self.assertRaises(ValueError):
                app_module.apply_y_limits(figure, {0: (3., None)})
        finally:
            app_module.plt.close(figure)
        self.app.y_min.set('2')
        self.app.y_max.set('1')
        with patch.object(app_module.messagebox, 'showerror') as error:
            self.app.assign_y_limits()
        error.assert_called_once()
        self.assertEqual(options['y_limits'], {0: (-.5, .5)})
        self.app.reset_y_limits()
        self.assertEqual(self.app.options()['y_limits'], {})
        self.app.style.set('latex_style')
        self.app.palette.set('screen')
        self.app.style_changed()
        self.assertEqual(self.app.palette.get(), 'okabe_ito')
        with patch.object(app_module, 'plot_measurement', side_effect=preset), \
                patch.object(app_module, 'set_latex_style', wraps=app_module.set_latex_style) as latex:
            figure, axes = app_module.render_request(None, self.app.options())
        try:
            latex.assert_called_once_with(palette='okabe_ito')
            self.assertTrue(axes[0, 0].yaxis.label.get_usetex())
        finally:
            app_module.plt.close(figure)

    def test_spike_selector_passes_raw_signal_threshold_to_loader(self):
        self.app.kind.set('custom')
        self.app.groups = [['reference']]
        self.app.refresh_groups()
        self.app.detector.set('spikes')
        with self.assertRaises(ValueError):
            self.app.options()
        self.app.spike_signal.set('measured')
        self.app.spike_threshold.set('3,0')
        options = self.app.options()
        data = MagicMock()
        data.t = np.arange(11)
        values = {'measured': np.array([10, 10, 10, 10, 14, 14, 10, 10, 16, 10, 10]),
                  'reference': np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
        data.__contains__.side_effect = values.__contains__
        data.__getitem__.side_effect = values.__getitem__
        data.time.return_value = data.t
        data.signal_info.return_value = [{'key': 'reference'}]
        with patch.object(app_module, 'LabDataMat') as reader:
            reader.return_value.__enter__.return_value = data
            snapshot, origin, limits = app_module.load_plot_request('measurement.mat', options)
            reader.return_value.__exit__.assert_called_once()
        self.assertAlmostEqual(origin, 3.8)
        self.assertAlmostEqual(limits[1], 4.4)
        np.testing.assert_array_equal(snapshot['reference'], values['reference'])
        self.assertEqual(options['window']['spike_threshold'], 3.)
        self.app.spike_threshold.set('0')
        with self.assertRaises(ValueError):
            self.app.options()
        self.app.detector.set('references')
        self.assertNotIn('spike_threshold', self.app.options()['window'])

    def test_preset_signals_add_remove_restore_and_batch_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ('a.mat', 'b.mat'):
                (root / name).touch()
            self.app.insert_paths([root / 'a.mat', root / 'b.mat'])
            self.app.select_all()
            self.app.signal_keys = ['wm_meas', 'tau_m']
            self.app.filter_signals()
            self.app.signals.selection_set(0)
            self.app.input_unit.set('rad/s')
            self.app.assign_units()
            self.assertEqual(self.app.kind.get(), 'hfsi')
            self.app.trace_tree.selection_set('panel:hfsi:0')
            self.app.add_preset_signals()
            self.assertEqual(len(self.app.preset_added['hfsi']), 1)
            added = self.app.preset_added['hfsi'][0]
            self.assertEqual((added['panel'], added['key']), (0, 'wm_meas'))
            self.assertEqual(self.app.trace_tree.selection(), (added['id'],))
            self.app.edit_scale.set('n_p')
            self.app.edit_offset.set('0')
            self.app.assign_transform()
            self.app.trace_tree.selection_set('hfsi:0:1')
            self.app.toggle_preset_signal()
            self.assertIn('hfsi:0:1', self.app.preset_disabled)
            self.assertIn('[pois]', self.app.trace_tree.item('hfsi:0:1', 'text'))
            options = self.app.options()
            self.assertEqual(options['added_traces'], [added])
            self.assertEqual(options['disabled_traces'], {'hfsi:0:1'})
            self.assertEqual(options['transforms'][added['id']]['scale'], 'n_p')
            self.assertEqual(options['units']['wm_meas'], 'rad/s')
            with patch.object(app_module, 'load_plot_request', return_value=None), \
                    patch.object(app_module, 'render_request', side_effect=self.small_figure) as render:
                self.app.export()
                self.wait_idle()
                self.assertEqual(render.call_count, 2)
                for call in render.call_args_list:
                    self.assertEqual(call.args[1]['disabled_traces'], {'hfsi:0:1'})
                    self.assertEqual(call.args[1]['added_traces'], [added])
            self.app.toggle_preset_signal()
            self.assertNotIn('hfsi:0:1', self.app.preset_disabled)
            self.app.trace_tree.selection_set(added['id'])
            self.app.toggle_preset_signal()
            self.assertEqual(self.app.preset_added['hfsi'], [])
            self.assertNotIn(added['id'], self.app.transform_overrides)
            self.assertEqual(options['added_traces'], [added])

    def test_preset_added_trace_units_and_restore_are_scoped_per_plot_type(self):
        self.app.signal_keys = ['tau_m', 'wm_meas']
        self.app.filter_signals()
        self.app.signals.selection_set(0, 1)
        self.app.trace_tree.selection_set('panel:hfsi:0')
        self.app.add_preset_signals()
        self.assertEqual([item['key'] for item in self.app.preset_added['hfsi']],
                         ['tau_m', 'wm_meas'])
        self.app.trace_tree.selection_set('panel:hfsi:0')
        self.app.add_preset_signals()
        self.assertEqual(len(self.app.preset_added['hfsi']), 2)
        self.app.trace_tree.selection_set('panel:hfsi:1')
        self.app.add_preset_signals()
        self.assertEqual(len(self.app.preset_added['hfsi']), 4)
        self.app.trace_tree.selection_set('hfsi:1:0')
        self.app.toggle_preset_signal()
        self.app.kind.set('overview')
        self.app.refresh_trace_editor()
        self.assertEqual(self.app.options()['added_traces'], [])
        self.app.trace_tree.selection_set('overview:0:0')
        self.app.toggle_preset_signal()
        self.app.kind.set('hfsi')
        self.app.refresh_trace_editor()
        self.assertEqual(self.app.options()['disabled_traces'], {'hfsi:1:0'})
        self.app.reset_preset_signals()
        self.assertEqual(self.app.options()['added_traces'], [])
        self.assertEqual(self.app.options()['disabled_traces'], set())
        self.assertIn('overview:0:0', self.app.preset_disabled)

    def test_rated_torque_uses_selected_editable_motor_profile(self):
        self.app.torque_mode.set('rated')
        self.app.torque_mode_changed()
        self.assertEqual(self.app.trace_records['hfsi:4:2']['scale'], '1')
        self.assertIn('nimellismomenttiin', self.app.trace_records['hfsi:4:2']['stage'])
        self.assertFalse(hasattr(self.app, 'rated_entry'))
        profile = dict(self.app.motor_profiles[self.app.profile_id.get()])
        profile.update(id='other', name='Other', tau_nom=12.5)
        self.app.motor_profiles['other'] = profile
        self.app.apply_profile('other')
        self.app.per_unit.set(False)
        options = self.app.options()
        self.assertEqual(options['torque_mode'], 'rated')
        self.assertFalse(options['per_unit'])
        self.assertEqual(options['base'].tau_nom, 12.5)
        self.app.parameters['tau_nom'].set('17')
        self.assertEqual(options['base'].tau_nom, 12.5)
        self.assertEqual(self.app.options()['base'].tau_nom, 17.)
        for value in ('', '0', '-1', 'nan', 'inf', 'invalid'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.app.parameters['tau_nom'].set(value)
                self.app.options()
        self.app.apply_profile('default')
        self.assertEqual(self.app.options()['base'].tau_nom, 29.2)

    def test_preset_trace_editor_snapshots_gain_for_preview_and_export(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'speed.mat'
            source.touch()
            self.app.insert_paths([source])
            trace = next(item for item in self.app.trace_records.values() if item['source'] == 'wm_meas')
            self.app.trace_tree.selection_set(trace['id'])
            self.app.trace_selected()
            self.assertEqual(self.app.edit_scale.get(), '1/2')
            self.app.edit_scale.set('n_p')
            self.app.edit_offset.set('0.5')
            self.app.assign_transform()
            expected = {'scale': 'n_p', 'offset': '0.5'}
            self.app.axis_limits['hfsi:0'] = (-1., 1.)
            self.app.axis_tick_steps['hfsi:0'] = (.5, .1)
            self.app.parameters['x_minor_step'].set('.05')
            self.app.palette.set('okabe_ito')
            captured = self.app.options()
            self.assertEqual(captured['transforms'][trace['id']], expected)
            with patch.object(app_module, 'load_plot_request', return_value=None), \
                    patch.object(app_module, 'render_request', side_effect=self.small_figure) as render:
                self.app.preview()
                self.wait_idle()
                self.app.export()
                self.wait_idle()
                self.assertEqual(render.call_args.args[1]['transforms'][trace['id']], expected)
                self.assertEqual(render.call_args.args[1]['y_limits'], {0: (-1., 1.)})
                self.assertEqual(render.call_args.args[1]['y_steps'], {0: (.5, .1)})
                self.assertEqual(render.call_args.args[1]['x_minor_step'], .05)
                self.assertTrue(render.call_args.args[1]['minor_grid'])
                self.assertEqual(render.call_args.args[1]['palette'], 'okabe_ito')
                self.app.trace_tree.selection_set(trace['id'])
                self.app.reset_transform()
                self.wait_idle()
            self.assertEqual(captured['transforms'][trace['id']], expected)
            self.assertNotIn(trace['id'], self.app.options()['transforms'])
            self.app.edit_scale.set('1/0')
            with patch.object(app_module.messagebox, 'showerror') as error:
                self.app.assign_transform()
            error.assert_called_once()
            self.assertNotIn(trace['id'], self.app.transform_overrides)
            self.assertTrue((Path(folder) / 'speed_hfsi_2.pdf').exists())

    def test_custom_group_removal_remaps_operations_and_replacement_resets_only_its_group(self):
        self.app.groups = [['speed'], ['speed'], ['other']]
        self.app.scene_groups['custom'] = self.app.groups
        self.app.refresh_groups()
        self.app.transform_overrides = {
            'custom:0:speed': {'scale': '1', 'offset': '0'},
            'custom:1:speed': {'scale': 'n_p', 'offset': '0'},
            'custom:2:other': {'scale': '3', 'offset': '0'},
        }
        self.app.axis_limits = {'custom:0': (-1., 1.), 'custom:1': (-2., 2.), 'custom:2': (-3., 3.)}
        self.app.axis_tick_steps = {'custom:0': (1., None), 'custom:1': (2., .2),
                                    'custom:2': (3., .3), 'hfsi:0': (.1, None)}
        self.app.legend_positions = {'custom:0': (.1, .2), 'custom:1': (.3, .4),
                                     'custom:2': (.5, .6), 'hfsi:0': (.7, .8)}
        self.app.group_list.selection_set(0)
        self.app.remove_group()
        self.assertEqual(self.app.axis_limits, {'custom:0': (-2., 2.), 'custom:1': (-3., 3.)})
        self.assertEqual(self.app.axis_tick_steps, {'custom:0': (2., .2), 'custom:1': (3., .3),
                                                    'hfsi:0': (.1, None)})
        self.assertEqual(self.app.legend_positions, {'custom:0': (.3, .4), 'custom:1': (.5, .6),
                                                     'hfsi:0': (.7, .8)})
        self.assertEqual(self.app.transform_overrides['custom:0:speed']['scale'], 'n_p')
        self.assertEqual(self.app.transform_overrides['custom:1:other']['scale'], '3')
        self.app.signal_keys = ['replacement']
        self.app.filter_signals()
        self.app.signals.selection_set(0)
        self.app.group_list.selection_set(0)
        self.app.replace_group()
        self.assertNotIn('custom:0:speed', self.app.transform_overrides)
        self.assertIn('custom:1:other', self.app.transform_overrides)
        self.assertEqual(self.app.groups, [['replacement'], ['other']])
        self.assertEqual(self.app.axis_limits, {'custom:1': (-3., 3.)})
        self.assertEqual(self.app.axis_tick_steps, {'custom:1': (3., .3), 'hfsi:0': (.1, None)})
        self.assertEqual(self.app.legend_positions, {'custom:1': (.5, .6), 'hfsi:0': (.7, .8)})

    def test_unit_checkbox_refreshes_preview_and_applies_to_export(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'current.mat'
            source.touch()
            self.app.insert_paths([source])
            self.app.signal_keys = ['current']
            self.app.filter_signals()
            self.app.signals.selection_set(0)
            self.app.add_group()
            self.app.input_unit.set('A')
            self.app.assign_units()
            self.assertIn('[A]', self.app.group_list.get(0))
            self.assertTrue(self.app.options()['per_unit'])
            options = self.app.options()
            reference = options['base'].i
            snapshot = app_module.SignalSnapshot(np.array([0., 1.]),
                {'current': np.array([reference, 2*reference])},
                {'current': np.array([0., 1.])}, {'current': [{'key': 'current'}]})
            payload = (snapshot, 0., (0., 1.))
            with patch.object(app_module, 'load_plot_request', return_value=payload), \
                    patch.object(app_module, 'render_request', wraps=app_module.render_request) as render:
                self.app.preview()
                self.wait_idle()
                np.testing.assert_allclose(self.app.figure.axes[0].lines[0].get_ydata(), [1., 2.])
                self.assertIn('(p.u.)', self.app.figure.axes[0].lines[0].get_label())
                self.app.per_unit.set(False)
                self.app.units_changed()
                self.wait_idle()
                np.testing.assert_allclose(self.app.figure.axes[0].lines[0].get_ydata(),
                                           [reference, 2*reference])
                self.assertIn('(A)', self.app.figure.axes[0].lines[0].get_label())
                self.app.export()
                self.wait_idle()
                self.assertFalse(render.call_args.args[1]['per_unit'])
                self.assertEqual(render.call_args.args[1]['units'], {'current': 'A'})
            self.assertTrue((Path(folder) / 'current_custom.pdf').is_file())
            captured = self.app.options()
            self.app.signal_units['current'] = 'V'
            self.assertEqual(captured['units'], {'current': 'A'})

    def test_preview_embeds_zoom_toolbar_and_export_preserves_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'one.mat'
            source.touch()
            self.app.insert_paths([source])
            with patch.object(app_module, 'load_plot_request', return_value=None), \
                    patch.object(app_module, 'render_request', side_effect=self.small_figure):
                self.app.preview()
                self.wait_idle()
                self.assertIsNotNone(self.app.toolbar)
                self.assertIsNotNone(self.app.canvas)
                preview_number = self.app.figure.number
                self.app.export()
                self.wait_idle()
            self.assertTrue((Path(folder) / 'one_hfsi_2.pdf').is_file())
            self.assertEqual(set(app_module.plt.get_fignums()), self.figure_numbers | {preview_number})
            self.assertIn('1 tallennettu, 0 virhettä', self.app.status.get())

    def test_batch_continues_after_read_error_and_honors_output_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / name for name in ('a.mat', 'bad.mat', 'c.mat')]
            for source in paths:
                source.touch()
            self.app.insert_paths(paths)
            self.app.files.selection_set(self.app.files.get_children())
            self.app.output.set(str(root / 'pdf'))

            def load(source, _options):
                if source.name == 'bad.mat':
                    raise KeyError('missing signal')
                return None

            with patch.object(app_module, 'load_plot_request', side_effect=load), \
                    patch.object(app_module, 'render_request', side_effect=self.small_figure), \
                    patch.object(app_module.messagebox, 'showwarning') as warning:
                self.app.export()
                self.wait_idle()
            self.assertTrue((root / 'pdf' / 'a_hfsi_2.pdf').is_file())
            self.assertTrue((root / 'pdf' / 'c_hfsi_2.pdf').is_file())
            self.assertFalse((root / 'pdf' / 'bad_hfsi_2.pdf').exists())
            self.assertEqual(set(app_module.plt.get_fignums()), self.figure_numbers)
            self.assertIn('2 tallennettu, 1 virhettä', self.app.status.get())
            warning.assert_called_once()

    def test_failed_pdf_write_preserves_existing_export(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'a.mat'
            source.touch()
            target = Path(folder) / 'a_hfsi_2.pdf'
            target.write_bytes(b'previous PDF')
            self.app.insert_paths([source])

            def broken_figure(payload, options):
                figure, axes = self.small_figure(payload, options)
                figure.savefig = MagicMock(side_effect=OSError('disk write failed'))
                return figure, axes

            with patch.object(app_module, 'load_plot_request', return_value=None), \
                    patch.object(app_module, 'render_request', side_effect=broken_figure), \
                    patch.object(app_module.messagebox, 'askyesno', return_value=True), \
                    patch.object(app_module.messagebox, 'showwarning'):
                self.app.export()
                self.wait_idle()
            self.assertEqual(target.read_bytes(), b'previous PDF')
            self.assertEqual(set(Path(folder).iterdir()), {source, target})
            self.assertEqual(set(app_module.plt.get_fignums()), self.figure_numbers)
            self.assertIn('0 tallennettu, 1 virhettä', self.app.status.get())

    def test_cancel_stops_before_next_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = [root / name for name in ('a.mat', 'b.mat')]
            for source in paths:
                source.touch()
            self.app.insert_paths(paths)
            self.app.files.selection_set(self.app.files.get_children())
            with patch.object(app_module, 'load_plot_request', return_value=None) as load, \
                    patch.object(app_module, 'render_request', side_effect=self.small_figure):
                self.app.export()
                self.app.cancel()
                self.wait_idle()
            load.assert_called_once()
            self.assertTrue((root / 'a_hfsi_2.pdf').is_file())
            self.assertFalse((root / 'b_hfsi_2.pdf').exists())
            self.assertIn('keskeytetty', self.app.status.get())


if __name__ == '__main__':
    unittest.main()
