# dSPACE Plotter

A local desktop application for exploring dSPACE ControlDesk measurements saved as MATLAB v7.3 / HDF5 `.mat` files. No Jupyter notebook or web service is required. The interface is currently in Finnish.

## Features

- Browse measurement files and folders, preview signals, and export selected files to PDF.
- Use built-in plots or configure multi-column scenarios with signal-specific units and calculations.
- Select motor profiles and display quantities in SI, per-unit, or rated-torque values.
- Set time windows, subplot limits, tick spacing, grid visibility, colors, and draggable legends.
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

1. Add files with **Tiedostot…** or a folder with **Kansio…**. Select the active file and choose **Esikatsele** to preview it.
2. Select a motor profile and verify its values. Profile fields can be edited for the current session. Rated torque uses the displayed nominal torque in Nm.
3. Select multiple files with Ctrl or Shift and choose **Tallenna valitut PDF** to export them. PDFs are saved beside their source files by default; existing files require confirmation before replacement.

Y axes are automatically scaled from each visible curve's **entire recording**, including data outside the displayed time window. If all finite values on an axis are nonnegative, the lower limit is −0.05 in the displayed units. Manual subplot bounds override the automatic limits.

Drag legends in the preview with toolbar zoom and pan switched off. Legend positions, units, axis limits, and other interface edits apply to previews and batch PDFs for the current session; they are not saved between launches.

## Plot scenarios

Edit [`measurement_scenarios.json`](./measurement_scenarios.json) to add or update scenarios. Saved changes are reloaded automatically while the application is running; **Lataa asetukset uudelleen** reloads them manually. Reloading waits for an active preview or batch export to finish. Session edits to changed scenarios are reset. If the JSON is invalid, the application reports the error and keeps the previous valid configuration. Both configuration files must be valid at startup.

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
          {"key": "wm_ref", "unit": "rad/s", "scale": "1", "offset": "0"},
          {"key": "wm_meas", "unit": "rad/s (mechanical)", "scale": "1", "offset": "0"}
        ]},
        {"title": "Torque", "signals": [
          {"key": "tau_m", "unit": "Nm", "scale": "1", "offset": "0"}
        ]}
      ]
    }
  ]
}
```

To add the example to an existing file, insert its scenario object into the existing `scenarios` array and preserve valid JSON syntax. `columns` accepts integers from 1 to 8; panels are placed row by row and unused cells are omitted. A signal key must be unique within a panel, but can appear in multiple panels. Each signal keeps its own time raster. Missing signals are reported per file; a batch continues with the remaining files.

### Signal units and calculations

`unit` describes the **recorded signal**, not its desired output. `raw` leaves values unchanged. Supported units are:

- `A`, `V`, `Nm`, `Wb`, `Vs`
- `rad/s` for electrical speed; `rad/s (mechanical)` and `rpm (mechanical)` for mechanical speed
- `p.u. (A)`, `p.u. (V)`, `p.u. (Nm)`, `p.u. (Wb)`, `p.u. (Vs)`, `p.u. (rad/s)`, and `p.u. (rad/s mechanical)` for already-normalized values

Mechanical rpm is converted to rad/s in SI display. `Wb` and `Vs` are equivalent here. Signals with different quantities can share an axis, but checking their scale and unit compatibility is the user's responsibility.

Each signal is transformed as `scale * y + offset` **before** unit conversion. Expressions support numbers, parentheses, `+ - * /`, and the names `n_p`, `sqrt3`, `u_base`, `i_base`, `w_base`, `psi_base`, `tau_base`, and `tau_nom`; Python calls are not allowed. A signal marked `rad/s (mechanical)` is normalized using the mechanical base speed. Do not also multiply by pole pairs unless that is intentional.

### Built-in scenarios

The `hfsi`, `overview`, `injection-overview`, and `presentation` implementations are defined in the application. Their configuration `kind` selects the implementation. Customize them with `disabled_traces`, `added_traces`, `transforms`, and `units`. Native trace IDs use `kind:panel:line`, with zero-based indices. For example, an `added_traces` item can be `{"id":"new", "panel":0, "key":"wm_meas"}`; use `{"new":{"scale":"n_p","offset":"0"}}` in `transforms` and `{"wm_meas":"rad/s"}` in `units`. Use a `configured` scenario for a fully file-defined layout. The subplot tree also shows legacy model-specific transforms; verify that they suit your measurement model.

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

- `references` detects changes in `wm_ref` and `tauLref` / `TL_ref`.
- `spikes` detects deviations from the entire selected signal's median that exceed a user-provided threshold in recorded units. It does not determine settling time.

If no events are found, the full recording is used. The `latex_style` publication style requires a separate TeX installation with `newtxtext` and `newtxmath`; `screen` does not.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest test_lab_plotter_config test_lab_plotter_core test_lab_plotter_app test_lab_plotting.TransientWindowTests test_lab_plotting.MeasurementSpikeTests test_lab_plotting.MeasurementFilesTests -q
```

## License and acknowledgements

This project is licensed under the [MIT License](./LICENSE). It includes plotting code modified from [motulator](https://github.com/Aalto-Electric-Drives/motulator), also licensed under the MIT License. See [third-party notices](./THIRD_PARTY_NOTICES.md) for its copyright and license text.

## Sharing data

Do not include confidential measurements, generated PDFs, or virtual environments when sharing the project. Review the configuration files before publishing them. The included [`.gitignore`](./.gitignore) excludes common local artifacts and measurement files; it does not remove files already tracked by Git.
