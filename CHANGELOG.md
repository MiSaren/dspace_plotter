# Changelog

All notable changes to this project are listed here. Versions follow [Semantic Versioning](https://semver.org/). Before 1.0, a minor version (0.x.0) may change configuration file formats, and a patch version (0.x.y) contains fixes only.

## [0.1.0] - 2026-11-25

First tagged release. The original idea and the earlier Jupyter notebook version are by [Hannu Hartikainen](https://github.com/HannuHar).

### Plotting

- Browse dSPACE ControlDesk recordings (MATLAB v7.3 / HDF5 `.mat`), preview them, and export selected files to PDF in a batch.
- Use the built-in `hfsi`, `overview`, `injection-overview`, and `presentation` figures, or define multi-column `configured` scenarios in `measurement_scenarios.json`.
- Select motor profiles from `motor_profiles.json`, and show quantities in SI units, per-unit values, or values relative to rated torque.
- Set time windows, subplot limits, tick spacing, minor grids, palettes, a LaTeX publication style, and draggable legends.
- Detect transients from reference changes or from measurement spikes.

### Signal names and quantities

- Signal maps in `signal_maps.json` link the quantities of the built-in figures to the signal names of a dSPACE model. Recording keys no longer need to be edited in `lab_plotter_core.py`.
- The link editor lists the signals of the active recording with their Simulink block paths. It shows whether each quantity is found, missing, or ambiguous, and saves links to an existing or new profile.
- A status line names the quantities of the current figure that are missing from the recording, and suggests a better-matching profile when one exists.
- `@role` keys, such as `@w_m_ref`, select a quantity through the signal map in configured scenarios, added traces, units, spike detection, and custom subplots. They default to the recorded unit of the role.
- **List all signals** shows every signal with its value range, sample count, and block path, and can save the list as text.

### Processing and comparison

- The derived processing of each trace can be changed in the subplot tree or with `"processing"` in the configuration. Options include magnitude, real part, and imaginary part of vectors; wrapping, unwrapping, and degrees for angles; and a time derivative for custom signals.
- **Compare side by side** previews two measurements with the same settings. By default, corresponding axes share limits, zoom, and pan.

### Interface

- The interface is available in Finnish and English and can be switched while the application runs. The choice is stored for each user in `lab_plotter_settings.json`, which Git ignores.

### Changes from the pre-release versions

- The HFSI angle-error and angle panels use the measured rotor angle (`thetam`).
- The `speed-torque` example scenario uses `@role` keys. With the default signal map it plots the same signals as before.
- Preset panel titles in `lab_plotter_core.subplot_traces` are now in English; the application translates them.

[0.1.0]: https://github.com/MiSaren/dspace_plotter/releases/tag/v0.1.0
