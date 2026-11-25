"""User-interface languages: English source texts and their Finnish translations.

translate(text, language) returns the Finnish text for 'fi' when the catalog
has one and the English source text otherwise; keyword values fill {name}
fields. Recording keys, units, scenario names and configuration identifiers
(references, per_unit, latex_style, ...) are not translated.

The chosen language is a per-user preference in lab_plotter_settings.json next
to this file (ignored by Git). A missing or unreadable file selects Finnish.
"""
import json
import os
import tempfile
from pathlib import Path

SETTINGS_PATH = Path(__file__).with_name('lab_plotter_settings.json')
LANGUAGES = {'fi': 'Suomi', 'en': 'English'}
DEFAULT_LANGUAGE = 'fi'
# Interface texts that read the same in both languages.
NEUTRAL = frozenset(('I RMS (A)', 'f (Hz)', 'P (W)'))


def translate(text, language, **values):
    result = FINNISH.get(text, text) if language == 'fi' else text
    return result.format(**values) if values else result


def load_language(path=SETTINGS_PATH):
    try:
        with Path(path).open(encoding='utf-8') as source:
            language = json.load(source).get('language')
    except (OSError, ValueError, AttributeError):
        return DEFAULT_LANGUAGE
    return language if language in LANGUAGES else DEFAULT_LANGUAGE


def save_language(language, path=SETTINGS_PATH):
    """Store the language, keeping other settings; failures are not fatal."""
    if language not in LANGUAGES:
        raise ValueError(f'Unknown language: {language}')
    path = Path(path)
    try:
        with path.open(encoding='utf-8') as source:
            settings = json.load(source)
        if not isinstance(settings, dict):
            settings = {}
    except (OSError, ValueError):
        settings = {}
    settings['language'] = language
    temporary = None
    try:
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.stem, suffix='.tmp')
        with os.fdopen(handle, 'w', encoding='utf-8') as target:
            json.dump(settings, target, indent=2)
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        return False
    return True


FINNISH = {
    # Main window
    'Lab plotter — measurement browser': 'Lab plotter — mittausselain',
    'Add measurement files or a folder.': 'Lisää mittaustiedostoja tai kansio.',
    'Preview: no figure': 'Esikatselu: ei kuvaa',
    'Preview: {path}': 'Esikatselu: {path}',
    'Language changed.': 'Kieli vaihdettu.',
    'Free figure': 'Vapaa kuva',
    'Files…': 'Tiedostot…',
    'Folder…': 'Kansio…',
    'Remove': 'Poista',
    'Ctrl/Shift: multiple selection; Ctrl+A: all; the active row is previewed':
        'Ctrl/Shift: monivalinta; Ctrl+A: kaikki; aktiivinen rivi esikatseluun',
    'Measurement': 'Mittaus',
    'State': 'Tila',
    'Added': 'Lisätty',
    'Saved': 'Tallennettu',
    'Error': 'Virhe',
    'MAT measurements': 'MAT-mittaukset',
    'All files': 'Kaikki',
    'Figure and time window': 'Kuva ja aikaikkuna',
    'Before transient (s)': 'Ennen transienttia (s)',
    'After last transient (s)': 'Viimeisen transientin jälkeen (s)',
    'Duration (s), empty = automatic': 'Pituus (s), tyhjä = automaattinen',
    'X major step (s), empty = default': 'X pääjako (s), tyhjä = oletus',
    'X minor step (s), empty = default': 'X apujako (s), tyhjä = oletus',
    'Time origin (s), empty = automatic': 'Nollakohta (s), tyhjä = autom.',
    'Use p.u. values (all figures)': 'Käytä p.u.-arvoja (kaikki kuvat)',
    'Show minor grid': 'Näytä apuruudukko (minor grid)',
    'Angles stay in deg/rad.': 'Kulmat säilyvät deg/rad-yksiköissä.',
    'Columns (custom / configured scenario)': 'Palstat (custom / asetusskenaario)',
    'Reload settings': 'Lataa asetukset uudelleen',
    'Appearance and palette': 'Ulkoasu ja väripaletti',
    'Style': 'Tyyli',
    'Palette': 'Paletti',
    'latex_style uses the legacy publication style; it needs a TeX installation and the newtx '
    'packages. The palette is chosen separately.':
        'latex_style käyttää vanhaa julkaisutyyliä; vaatii TeX-asennuksen ja newtx-paketit. '
        'Paletti valitaan erikseen.',
    'Transient detection': 'Transienttitunnistus',
    'Spike signal': 'Piikkisignaali',
    'Amplitude threshold (raw units)': 'Amplitudikynnys (raakayksikkö)',
    'references: changes of the linked speed and load-torque references (default wm_ref, '
    'tauLref/TL_ref). spikes: |y − median of the whole signal| > threshold in the selected signal '
    '(also an @quantity). Detection uses the original data, not p.u. conversions. '
    'No events → whole recording.':
        'references: linkitettyjen nopeus- ja kuormamomenttiohjeiden muutokset (oletus wm_ref, '
        'tauLref/TL_ref). spikes: |y − koko signaalin mediaani| > kynnys valitussa mittaussignaalissa '
        '(myös @-suure). Tunnistus käyttää alkuperäistä dataa, ei p.u.-muunnoksia. '
        'Ei tapahtumia → koko tallenne.',
    'Motor — same values for all selected files': 'Moottori — samat arvot kaikille valituille',
    'Motor profile': 'Moottoriprofiili',
    'U RMS, line-to-line (V)': 'U RMS, pääjännite (V)',
    'T rated (Nm)': 'T nimellinen (Nm)',
    'Pole pairs': 'Napaparit',
    'Torque display': 'Momentin esitystapa',
    'rated: torques / nominal torque of the motor profile; other quantities follow the p.u. choice.':
        'rated: momentit / moottoriprofiilin nimellismomentti; muut suureet noudattavat p.u.-valintaa.',
    'Signal links (dSPACE model)': 'Signaalien linkitys (dSPACE-malli)',
    'Signal map': 'Linkitysprofiili',
    'Edit links…': 'Muokkaa linkityksiä…',
    'Measured signals / custom subplots': 'Mittaussignaalit / custom-alikuvaajat',
    'Unit of the recorded signal:': 'Tallennetun signaalin yksikkö:',
    'Set for selected': 'Aseta valituille',
    'Add subplot': 'Lisää alikuvaaja',
    'Replace selected': 'Korvaa valittu',
    'Remove subplot': 'Poista alikuvaaja',
    'Ctrl/Shift: several signals on the same axis. @quantities at the top follow the signal map and '
    'work across dSPACE models. raw = no conversion; p.u. (Nm) etc. = already normalized input. '
    'Select the input unit first and press Set for selected.':
        'Ctrl/Shift: useita signaaleja samalle akselille. Listan alun @-suureet seuraavat '
        'linkitysprofiilia ja toimivat eri dSPACE-malleissa. raw = ei muunnosta; p.u. (Nm) ym. = jo '
        'normalisoitu syöte. Valitse ensin syötteen yksikkö ja paina Aseta valituille.',
    'Subplot signals and calculations': 'Alikuvaajien signaalit ja laskutoimitukset',
    'Subplot / signal source': 'Alikuvaaja / signaalilähde',
    'Current derived processing': 'Nykyinen johdettu käsittely',
    'Gain': 'Kerroin',
    'Offset': 'Lisäys',
    'Add selected signals': 'Lisää valitut signaalit',
    'Remove / restore signal': 'Poista / palauta signaali',
    'Restore signals': 'Palauta signaalit',
    'Preset: select the target subplot here and the recorded signals and their input units above. '
    'A hidden original can be restored with the same button. Added signals use their own time raster.':
        'Esiasetus: valitse tästä kohdealikuvaaja sekä yllä mittaussignaalit ja niiden syöttöyksiköt. '
        'Piilotetun alkuperäisen voi palauttaa samalla painikkeella. '
        'Lisätyt signaalit käyttävät tallenteen omaa aikarasteria.',
    'Set Y limits': 'Aseta Y-rajat',
    'Default limits': 'Oletusrajat',
    'Select a subplot or its signal. Empty limit = default/automatic. Limits are in displayed units; '
    'switching p.u. does not convert the entered numbers.':
        'Valitse alikuvaaja tai sen signaali. Tyhjä raja = oletus/autom. Rajojen luvut ovat näkyvissä '
        'yksiköissä; p.u.-vaihto ei muunna syötettyjä lukuja.',
    'Y major step': 'Y pääjako',
    'Y minor step': 'Y apujako',
    'Set Y steps': 'Aseta Y-jaot',
    'Restore Y steps': 'Palauta Y-jaot',
    'Steps are in displayed units. Empty = default; switching p.u. does not convert the set steps.':
        'Jaot ovat näkyvissä yksiköissä. Tyhjä = oletus; p.u.-vaihto ei muunna asetettuja jakoja.',
    'Set transform': 'Aseta muunnos',
    'Reset selected to default': 'Palauta valittu oletukseen',
    'y → gain × y + offset before p.u. conversion. Mechanical → electrical speed: gain n_p (replaces '
    'the current gain). Allowed e.g. 1/n_p, 2*n_p, -1, i_base. The edited axis is autoscaled.':
        'y → kerroin × y + lisäys ennen p.u.-muunnosta. Mekaaninen → sähköinen nopeus: kerroin n_p '
        '(korvaa nykyisen kertoimen). Sallittu esim. 1/n_p, 2*n_p, -1, i_base. '
        'Muokatun akselin y-rajaus on automaattinen.',
    'PDF export': 'PDF-tallennus',
    'Empty = source folder of each measurement': 'Tyhjä = kunkin mittauksen lähdekansio',
    'Output folder…': 'Tallennuskansio…',
    'Source folder': 'Lähdekansio',
    'Preview': 'Esikatsele',
    'Save selected as PDF': 'Tallenna valitut PDF',
    'Cancel batch': 'Keskeytä eräajo',
    'Reset legends of this figure type': 'Palauta tämän kuvatyypin legendit',
    "Drag legends in the preview (zoom/pan off). Positions are kept in this session's PDF batches.":
        'Raahaa legendoja esikatselussa (zoom/siirto pois). Sijainnit säilyvät tämän istunnon '
        'PDF-eräajossa.',
    # Trace editor
    'Speed': 'Nopeus',
    'Currents': 'Virrat',
    'Flux': 'Vuo',
    'Angle error': 'Kulmavirhe',
    'Torque': 'Momentti',
    'Angle': 'Kulma',
    'Voltages': 'Jännitteet',
    'Injection voltages': 'Injektiojännitteet',
    'Subplot {number}': 'Alikuvaaja {number}',
    '({role} not linked)': '({role} ei linkitetty)',
    '[off]': '[pois]',
    'recorded p.u.; converted to rated torque': 'recorded p.u.; muunnetaan nimellismomenttiin',
    'added, own time raster of the recording': 'lisätty, tallenteen oma aikarasteri',
    'Select a subplot for its Y limits or a signal for its calculations.':
        'Valitse alikuvaaja sen Y-rajoja varten tai signaali sen laskutoimituksia varten.',
    '{title}: edit the Y-axis limits.': '{title}: muokkaa Y-akselin rajoja.',
    '{title}: {source}. Derived processing: {stage}. Default gain: {scale}.':
        '{title}: {source}. Johdettu käsittely: {stage}. Oletuskerroin: {scale}.',
    'none': 'ei',
    'First select a target subplot of a preset, or its signal.':
        'Valitse ensin valmiin esiasetuksen kohdealikuvaaja tai sen signaali.',
    'Select the signals to add from the signal list.':
        'Valitse lisättävät mittaussignaalit signaaliluettelosta.',
    'The selected signals are already in that subplot.': 'Valitut signaalit ovat jo kyseisessä alikuvaajassa.',
    'Select a single signal of a preset.': 'Valitse valmiin esiasetuksen yksittäinen signaali.',
    'Select a preset to restore.': 'Valitse valmis esiasetus palautettavaksi.',
    'Y limits must be finite': 'Y-rajojen on oltava äärellisiä',
    'Y max must be greater than Y min': 'Y max on oltava suurempi kuin Y min',
    'Y steps must be positive and finite': 'Y-jaon on oltava positiivinen ja äärellinen',
    # Status and errors
    '{name}: {count} signals. Select a figure or build subplots.':
        '{name}: {count} signaalia. Valitse kuva tai kokoa alikuvaajat.',
    'Settings will be reloaded after the running job.': 'Asetukset ladataan vasta käynnissä olevan työn jälkeen.',
    'Configuration file error; previous settings kept: {error}':
        'Asetustiedoston virhe; aiemmat asetukset säilytetty: {error}',
    'Configuration file error': 'Asetustiedoston virhe',
    'Settings updated.': 'Asetukset päivitetty.',
    'Edited motor values were kept; select the profile again to update them.':
        'Muokatut moottoriarvot säilytettiin; valitse profiili uudelleen päivittääksesi ne.',
    'Session edits of changed scenarios were reset.':
        'Muuttuneiden skenaarioiden istuntokohtaiset muokkaukset nollattiin.',
    '{key}: the value must be finite': '{key}: arvon on oltava äärellinen',
    'Motor values must be positive; n_p is an integer': 'Moottoriarvojen on oltava positiivisia; n_p on kokonaisluku',
    'Select the torque display': 'Valitse momentin esitystapa',
    'Margins ≥ 0; duration and time steps > 0': 'Marginaalit ≥ 0; pituus ja aikajaot > 0',
    'Add at least one subplot': 'Lisää vähintään yksi alikuvaaja',
    'The number of columns must be an integer 1–8': 'Palstojen määrän on oltava kokonaisluku 1–8',
    'Select a known style and palette': 'Valitse tunnettu tyyli ja paletti',
    'Select references or spikes': 'Valitse references tai spikes',
    'Select the signal for measurement spikes': 'Valitse mittauspiikkien signaali',
    'The spike threshold must be positive and finite': 'Piikkikynnyksen on oltava positiivinen ja äärellinen',
    'Error: {error}': 'Virhe: {error}',
    'Plotting failed': 'Plottaus epäonnistui',
    'Reading {name}…': 'Luetaan {name}…',
    '{name}: {kind}. Drag the legend; switch zoom/pan off while dragging.':
        '{name}: {kind}. Raahaa legenda; zoom/siirto pois raahauksen ajaksi.',
    'Measurements with the same name: use source folders or rename the files.':
        'Saman nimiset mittaukset: käytä lähdekansioita tai nimeä tiedostot eri tavoin.',
    'Replace PDFs?': 'Korvataanko PDF:t?',
    '{count} files already exist. Replace them?': '{count} tiedostoa on jo olemassa. Korvataanko?',
    'Batch: {saved} saved, {errors} errors.': 'Eräajo: {saved} tallennettu, {errors} virhettä.',
    'Batch: {saved} saved, {errors} errors; cancelled.':
        'Eräajo: {saved} tallennettu, {errors} virhettä; keskeytetty.',
    'Batch errors': 'Eräajon virheet',
    'Saving {name}… ({count} left)': 'Tallennetaan {name}… (jäljellä {count})',
    'Cancelling after the current file…': 'Keskeytetään nykyisen tiedoston jälkeen…',
    # Signal maps
    'Select a measurement: the status shows whether the quantities of the figure are in the recording.':
        'Valitse mittaus: tila kertoo, löytyvätkö kuvan suureet tallenteesta.',
    '{name}: {found}/{total} quantities of the figure found.': '{name}: {found}/{total} kuvan suureesta löytyi.',
    'Missing: {items}.': 'Puuttuu: {items}.',
    'not linked': 'ei linkitetty',
    'Profile "{name}" matches better ({count} missing).': 'Profiili "{name}" sopii paremmin ({count} puuttuu).',
    'Recording keys are used directly; @quantities follow the signal map.':
        'Tallenteen avaimia käytetään suoraan; @-suureet seuraavat linkitysprofiilia.',
    'Signal map "{name}" saved: {file}.': 'Linkitysprofiili "{name}" tallennettu: {file}.',
    'Signal links — {name} [{id}]': 'Signaalien linkitys — {name} [{id}]',
    'Choose a recording signal for each quantity. Empty = not linked (the quantity is plotted empty).':
        'Valitse kullekin suureelle tallenteen signaali. Tyhjä = ei linkitetty (suure piirtyy tyhjänä).',
    'Quantity': 'Suure',
    'Role': 'Rooli',
    'Used by': 'Käyttö',
    'Recording signal': 'Tallenteen signaali',
    'Status': 'Tila',
    'Save to profile': 'Tallenna profiiliin',
    'Save as new profile…': 'Tallenna uutena profiilina…',
    'Close': 'Sulje',
    'Recording: {name} ({count} signals). The list shows the block path of each signal; vector '
    'quantities list 2–3-element vectors without [index].':
        'Tallenne: {name} ({count} signaalia). Luettelo näyttää kunkin signaalin lohkopolun; '
        'vektorisuureille tarjotaan 2–3-alkioiset vektorit ilman [indeksiä].',
    'List all signals…': 'Listaa kaikki signaalit…',
    'Select a measurement first.': 'Valitse ensin mittaus.',
    'Signals — {name}': 'Signaalit — {name}',
    'Recording {name}: {count} signals': 'Tallenne {name}: {count} signaalia',
    'Key': 'Avain',
    'Range': 'Arvoalue',
    'Samples': 'Näytteitä',
    'Source (block path)': 'Lähde (lohkopolku)',
    'Save as text…': 'Tallenna tekstinä…',
    'Text files': 'Tekstitiedostot',
    'Signal list saved: {path}': 'Signaaliluettelo tallennettu: {path}',
    'No measurement read: select a file in the main window to see its signals and the link status here.':
        'Ei luettua mittausta: valitse tiedosto pääikkunasta, niin sen signaalit ja linkitysten tila '
        'näkyvät tässä.',
    '— not linked': '— ei linkitetty',
    'no recording to check': 'ei tarkistettavaa tallennetta',
    '✓ found': '✓ löytyi',
    '✗ wrong shape: {expected}': '✗ väärä muoto: {expected}',
    '2–3-element vector': '2–3-alkioinen vektori',
    'single signal': 'yksi signaali',
    '✗ ambiguous, choose: {keys}': '✗ moniselitteinen, valitse: {keys}',
    '✗ not in the recording': '✗ ei tallenteessa',
    'alternatives {keys}': 'vaihtoehdot {keys}',
    'Save signal map': 'Tallenna linkitysprofiili',
    'Replace the links of profile "{name}" in {file}?': 'Korvataanko profiilin "{name}" linkitykset tiedostossa {file}?',
    'New signal map': 'Uusi linkitysprofiili',
    'Profile name (e.g. the name of the dSPACE model):': 'Profiilin nimi (esim. dSPACE-mallin nimi):',
    'Saving failed': 'Tallennus epäonnistui',
    'all figures': 'kaikki kuvat',
    'not plotted': 'ei piirretä',
    '{figures} + transients': '{figures} + transientit',
    # Quantity roles (lab_plotter_config.SIGNAL_ROLES)
    'Speed reference (electrical rad/s)': 'Nopeusohje (sähk. rad/s)',
    'Estimated speed (electrical rad/s)': 'Estimoitu nopeus (sähk. rad/s)',
    'Measured speed (HFSI figure, default gain 1/2)': 'Mitattu nopeus (HFSI-kuva, oletuskerroin 1/2)',
    'Estimated rotor angle (electrical rad), also dq transform':
        'Estimoitu roottorikulma (sähk. rad), myös dq-muunnos',
    'Measured rotor angle (electrical rad)': 'Mitattu roottorikulma (sähk. rad)',
    'Flux reference |ψs| (Wb)': 'Vuo-ohje |ψs| (Wb)',
    'Estimated flux ψs (2-vector, Wb)': 'Estimoitu vuo ψs (2-vektori, Wb)',
    'Limited torque reference (Nm)': 'Rajoitettu momenttiohje (Nm)',
    'Estimated torque (Nm)': 'Estimoitu momentti (Nm)',
    'Load torque reference (p.u.)': 'Kuormamomenttiohje (p.u.)',
    'Measured currents (abc or 2-vector, A)': 'Mitatut virrat (abc- tai 2-vektori, A)',
    'DC voltage (V)': 'DC-jännite (V)',
    'Realized voltage (2-vector, V)': 'Toteutunut jännite (2-vektori, V)',
    'Injection voltage (2-vector, V)': 'Injektiojännite (2-vektori, V)',
    'Injection error signal (rad)': 'Injektion virhesignaali (rad)',
    'Stator current (2-vector, A)': 'Staattorivirta (2-vektori, A)',
    'Current reference (2-vector, A)': 'Virtaohje (2-vektori, A)',
    'Observer monitor signal mon[4]': 'Observerin seurantasignaali mon[4]',
}
