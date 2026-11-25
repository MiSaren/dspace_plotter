# dSPACE Plotter

A local desktop application for exploring dSPACE ControlDesk measurements saved as MATLAB v7.3 / HDF5 `.mat` files. No Jupyter notebook or web service is required. The interface is available in Finnish and English.

## Features

- Browse measurement files and folders, preview signals, and export selected files to PDF.
- Use built-in plots or configure multi-column scenarios with signal-specific units and calculations.
- Select motor profiles and display quantities in SI, per-unit, or rated-torque values.
- Link the quantities of the built-in plots to the signal names of your dSPACE model in a link editor, and save the links as reusable profiles.
- Set time windows, subplot limits, tick spacing, grid visibility, colors, and draggable legends.
- Change the derived processing of each trace, such as magnitude or real part of a vector, or angle wrapping.
- Compare two measurements side by side with shared axis limits and zoom.
- Detect transients from reference changes or measurement deviations.

## Quick start

Tested on Windows with Python 3.13.7. The Python installation must include Tkinter. From PowerShell in the project directory:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r lab_plotter_requirements.txt
.\.venv\Scripts\python.exe .\lab_plotter_app.py
```

Use files from trusted sources, and verify signal units and motor data before interpreting results.

## Using the application

Choose the interface language with **Kieli / Language** at the top of the sidebar. The window is translated immediately. The choice is stored for your user account in `lab_plotter_settings.json`, which Git ignores; without that file the interface starts in Finnish. Button names below are given in English, with the Finnish name in parentheses.

1. Add files with **Files…** (**Tiedostot…**) or a folder with **Folder…** (**Kansio…**). Select the active file and choose **Preview** (**Esikatsele**) to preview it.
2. Select a signal-map profile that matches your dSPACE model and check its status line (see [Signal maps](#signal-maps)). Select a motor profile and verify its values. Profile fields can be edited for the current session. Rated torque uses the displayed nominal torque in Nm.
3. Select multiple files with Ctrl or Shift and choose **Save selected as PDF** (**Tallenna valitut PDF**) to export them. PDFs are saved beside their source files by default; existing files require confirmation before replacement.
4. To compare two measurements, select both with Ctrl and choose **Compare side by side** (**Vertaa vierekkäin**). Both are drawn with the same figure type and settings, each with its own toolbar. With **Same axis limits and zoom when comparing** (**Samat akselirajat ja zoom vertailussa**) switched on, which is the default, corresponding axes use common limits and zooming or panning one figure also changes the other. Setting changes redraw the comparison, and **Preview** returns to a single figure.

Y axes are automatically scaled from each visible curve's **entire recording**, including data outside the displayed time window. If all finite values on an axis are nonnegative, the lower limit is −0.05 in the displayed units. Manual subplot bounds override the automatic limits.

Drag legends in the preview with toolbar zoom and pan switched off. Legend positions, units, axis limits, and other interface edits apply to previews and batch PDFs for the current session; they are not saved between launches.

## Plot scenarios

Edit [`measurement_scenarios.json`](./measurement_scenarios.json) to add or update scenarios. Saved changes are reloaded automatically while the application is running; **Reload settings** (**Lataa asetukset uudelleen**) reloads them manually. Reloading waits for an active preview or batch export to finish. Session edits to changed scenarios are reset. If the JSON is invalid, the application reports the error and keeps the previous valid configuration. The scenario and motor files must be valid at startup, as must `signal_maps.json` when it exists.

Each item in `scenarios` has a unique `id`, a menu `label`, a `kind`, and a PDF filename `suffix`. IDs and suffixes may contain only letters, numbers, underscores, and hyphens. For example, this complete configuration defines two columns:

```json
{
  "scenarios": [
    {
      "id": "my_test",
      "label": "My test",
      "kind": "configured",
      "suffix": "_my_test",
      "columns": 2,
      "panels": [
        {"title": "Speed", "signals": [
          {"key": "@w_m_ref"},
          {"key": "wm_meas", "unit": "rad/s (mechanical)", "scale": "1", "offset": "0"}
        ]},
        {"title": "Torque", "signals": [
          {"key": "@tau_Mest", "scale": "1", "offset": "0"}
        ]}
      ]
    }
  ]
}
```

To add the example to an existing file, insert its scenario object into the existing `scenarios` array and preserve valid JSON syntax. `columns` accepts integers from 1 to 8; panels are placed row by row and unused cells are omitted. A signal key must be unique within a panel, but can appear in multiple panels. Each signal keeps its own time raster. Missing signals are reported per file; a batch continues with the remaining files.

A key that starts with `@` names a quantity role instead of a recording key, for example `@w_m_ref` or `@tau_Mest` above. The selected [signal map](#signal-maps) decides which recorded signal the role uses, so the same scenario works for different dSPACE models. When a role signal has no `unit`, the role's recorded unit is used: for example, `rad/s` for speeds, `Nm` for torques, `p.u. (Nm)` for the load-torque reference, and `raw` for angles. Vector roles, such as `@i_cs`, plot one line per component. The legend shows the recording key that was used. `@` keys also work in `added_traces`, in `units`, as the spike-detection signal, and in custom subplots built in the application, where they are listed at the top of the signal list.

### Signal units and calculations

`unit` describes the **recorded signal**, not its desired output. `raw` leaves values unchanged. Supported units are:

- `A`, `V`, `Nm`, `Wb`, `Vs`
- `rad/s` for electrical speed; `rad/s (mechanical)` and `rpm (mechanical)` for mechanical speed
- `p.u. (A)`, `p.u. (V)`, `p.u. (Nm)`, `p.u. (Wb)`, `p.u. (Vs)`, `p.u. (rad/s)`, and `p.u. (rad/s mechanical)` for already-normalized values

Mechanical rpm is converted to rad/s in SI display. `Wb` and `Vs` are equivalent here. Signals with different quantities can share an axis, but checking their scale and unit compatibility is the user's responsibility.

Each signal is first given its derived processing and then transformed as `scale * y + offset` **before** unit conversion. Expressions support numbers, parentheses, `+ - * /`, and the names `n_p`, `sqrt3`, `u_base`, `i_base`, `w_base`, `psi_base`, `tau_base`, and `tau_nom`; Python calls are not allowed. A signal marked `rad/s (mechanical)` is normalized using the mechanical base speed. Do not also multiply by pole pairs unless that is intentional.

### Derived processing

Select a trace in the subplot tree and choose its **Processing** (**Käsittely**). The list offers only the operations that suit that trace:

| Trace | Operations (`processing` id) |
| --- | --- |
| Vector quantity in a preset, such as estimated flux, voltage, or dq current | magnitude `magnitude`, real part `real`, imaginary part `imag` |
| Angle in a preset, such as rotor angle or angle error | `none`, wrap to (−π, π] `wrap`, `unwrap`, `degrees`, wrap to (−180°, 180°] `wrap_degrees`, wrapped mechanical degrees `wrap_mech_degrees` (÷ `n_p`) |
| Other preset quantity, such as speed or torque | `none`, absolute value `abs` |
| Custom or configured signal | `none`, `abs`, the angle operations, and time derivative `derivative`; for a 2- or 3-element vector also `magnitude`, `real`, `imag`, and angle `angle` |

Preset traces keep the quantity of their axis, so, for example, the angle of a normalized flux is not offered. Vector operations combine the components of a custom vector into one trace (`|i_c_abc|`, `Re(…)`, `Im(…)`, `∠…`); 3-element signals are treated as abc space vectors. Angle results are shown in `rad` or `deg` without unit conversion, and a derivative is shown per second. In configuration files, set `"processing"` in an item of `transforms` or in a configured signal, for example `{"overview:1:1": {"processing": "real"}}`. A transform's missing `scale` is `1`, which replaces the preset gain.

### Built-in scenarios

The `hfsi`, `overview`, `injection-overview`, and `presentation` implementations are defined in the application. Their configuration `kind` selects the implementation. Customize them with `disabled_traces`, `added_traces`, `transforms`, and `units`. Native trace IDs use `kind:panel:line`, with zero-based indices. For example, an `added_traces` item can be `{"id":"new", "panel":0, "key":"wm_meas"}`; use `{"new":{"scale":"n_p","offset":"0"}}` in `transforms` and `{"wm_meas":"rad/s"}` in `units`. Their quantities are read through the selected [signal map](#signal-maps). Added traces and `configured` scenarios use recording keys directly, or `@role` keys that follow the signal map. Use a `configured` scenario for a fully file-defined layout. The subplot tree also shows legacy model-specific transforms; verify that they suit your measurement model.

## Signal maps

Recorded signal names depend on the dSPACE model, so the built-in plots read their quantities (for example the speed reference, estimated flux, or phase currents) through a **signal map**: a profile that links each quantity to a key in the `.mat` file. Signal maps are stored in [`signal_maps.json`](./signal_maps.json). The included `default` profile contains the keys the plots originally used.

To link the signals of your own model:

1. Select a measurement file. The **Signal links** (**Signaalien linkitys**) status line shows how many quantities of the current plot were found and names the missing ones. If another profile matches the file better, the status line names it.
2. Choose **Edit links…** (**Muokkaa linkityksiä…**). For each quantity, the editor shows the plots that use it, the linked key, and whether that key exists in the active file. Choose a key from the file's signal list or type one. The list and the status show the Simulink block path each signal comes from (for example `CTRL/Vector Control/Sensorless SpeedFlux Observer/MATLAB Function`), which helps when the signal names themselves are not descriptive. Vector quantities, such as currents and flux, list 2- and 3-element vectors without their `[index]`. Leave a row empty to unlink a quantity; it is then plotted empty. If a name occurs under several block paths, the status lists the alternatives so that you can choose the correct one.
   **List all signals…** (**Listaa kaikki signaalit…**), also available below the signal list of the main window, opens a table of every signal in the active recording with its key, value range, number of samples, and block path. The table can be copied or saved as a text file.
3. Choose **Save as new profile…** (**Tallenna uutena profiilina…**) to create a profile for your model, or **Save to profile** (**Tallenna profiiliin**) to replace the links of the profile being edited.

Rows you do not change keep any saved alternatives. Saving a profile does not reset other session settings. The selected profile applies to previews and PDF batches, and the file is reloaded automatically like the other configuration files.

Missing quantities are reported per file; the remaining traces are still plotted.

The file can also be edited by hand. Each entry in `signal_maps` has a unique `id`, a `name`, and a `signals` object. Each entry in `signals` maps a role to a key or to an ordered list of alternative keys; the first alternative present with the correct shape is used, and `[]` unlinks the quantity. Roles that are omitted use the default keys. For example:

```json
{
  "signal_maps": [
    {
      "id": "lab-b",
      "name": "Lab B model",
      "signals": {
        "theta_m": "theta_m",
        "tau_Lref": ["load_torque_ref", "TL_ref"],
        "eps_inj": []
      }
    }
  ]
}
```

Role names are listed in the editor and in `SIGNAL_ROLES` in [`lab_plotter_config.py`](./lab_plotter_config.py). Profiles saved from the application list every role explicitly. This means that later changes to the default keys do not affect them.

## Motor profiles

Edit [`motor_profiles.json`](./motor_profiles.json) to add entries to `motors`. Each entry needs a unique `id`, a `name`, and `U_nom`, `I_nom`, `f_nom`, `P_nom`, `tau_nom`, and `n_p`. Values must be positive and `n_p` must be an integer. The included `default` profile is an **example**, not a universal motor rating. Selecting a profile replaces the displayed fields. Changed profiles reload automatically, but fields edited manually in the current session are not overwritten automatically.

## Units and base values

Base values follow [motulator's `BaseValues` calculation](https://github.com/Aalto-Electric-Drives/motulator/blob/ba5ed869f3c8bf1884f126b8515e3655e1bcd6a2/motulator/common/utils/_utils.py#L203-L291): nominal voltage is line-to-line RMS (V), nominal current is RMS (A), frequency is in Hz, power in W, torque in Nm, and `n_p` is the number of pole pairs. The peak-value bases are:

| Quantity | Base value |
| --- | --- |
| Voltage | `u_base = sqrt(2/3) * U_nom` |
| Current | `i_base = sqrt(2) * I_nom` |
| Electrical angular speed | `w_base = 2 * pi * f_nom` |
| Mechanical angular speed | `w_base / n_p` |
| Flux linkage | `psi_base = u_base / w_base` |
| Torque | `tau_base = n_p * (1.5 * u_base * i_base) / w_base` |

Rated torque is not the same as base torque. Per-unit values use the corresponding base; rated-torque mode uses the profile's nominal torque. Electrical and mechanical rotor speeds satisfy `w_m = n_p * w_M` ([motulator model](https://github.com/Aalto-Electric-Drives/motulator/blob/ba5ed869f3c8bf1884f126b8515e3655e1bcd6a2/motulator/drive/model/_machine.py#L180-L190)). A filename does not establish whether a recorded speed is electrical or mechanical.

## Transient detection and plotting styles

- `references` detects changes in the speed and load-torque references linked in the selected signal map (by default `wm_ref` and `tauLref` / `TL_ref`).
- `spikes` detects deviations from the entire selected signal's median that exceed a user-provided threshold in recorded units. It does not determine settling time.

If no events are found, the full recording is used. The `latex_style` publication style requires a separate TeX installation with `newtxtext` and `newtxmath`; `screen` does not.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_lab_plotter_config test_lab_plotter_core test_lab_plotter_app test_lab_plotter_i18n test_lab_plotting.CatalogTests test_lab_plotting.TransientWindowTests test_lab_plotting.MeasurementSpikeTests test_lab_plotting.MeasurementFilesTests -q
```

## License and acknowledgements

The original idea for this plotter and its earlier Jupyter notebook version were created by [Hannu Hartikainen](https://github.com/HannuHar). Many thanks to Hannu for this foundation.

This project is licensed under the [MIT License](./LICENSE). It includes plotting code modified from [motulator](https://github.com/Aalto-Electric-Drives/motulator), also licensed under the MIT License. See [third-party notices](./THIRD_PARTY_NOTICES.md) for its copyright and license text.

## Sharing data

Do not include confidential measurements, generated PDFs, or virtual environments when sharing the project. Review the configuration files before publishing them. The included [`.gitignore`](./.gitignore) excludes common local artifacts and measurement files; it does not remove files already tracked by Git.
