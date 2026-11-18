# Lab Plotter

Paikallinen Tk/Matplotlib-sovellus dSPACE ControlDesk -mittausten MATLAB v7.3 / HDF5 `.mat` -tiedostoille. Ei tarvitse Jupyteria eikä verkkopalvelua. Käynnistä aina luotetuista tiedostoista ja tarkista mittaussignaalien yksiköt sekä moottorin tiedot ennen tulosten käyttöä.

## Käynnistys Windowsissa

Testattu Pythonilla 3.13.7. Python-asennuksen tulee sisältää Tkinter. Avaa PowerShell tässä kansiossa:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r lab_plotter_requirements.txt
.\.venv\Scripts\python.exe .\lab_plotter_app.py
```

Esikatsele aktiivinen tiedosto ja valitse Ctrl/Shift-valinnalla useita tiedostoja PDF-eräajoon. PDF:t tallennetaan oletuksena kunkin mittauksen lähdekansioon; aiempien PDF:ien korvaamisesta kysytään. Valitse ensin oikea **moottoriprofiili** ja tarkista sen arvot. Profiilia voi muokata ruudulla tilapäisesti. `rated`-momentti lasketaan moottoriprofiilin näytössä näkyvällä, tarvittaessa muokattavalla nimellismomentilla (Nm).

Y-akselit skaalataan oletuksena kunkin näkyvän käyrän **koko mittauksen** arvoista, myös aikaikkunan ulkopuolelta. Kun kaikki kyseisen akselin äärelliset arvot ovat epänegatiivisia, alaraja on −0,05 näytetyssä yksikössä. Y-rajat voi myös asettaa käsin alikuvaajakohtaisesti; käsin annettu raja ohittaa automaattisen rajan.

Valitse kuvassa legendan sijainti raahaamalla, kun työkalupalkin zoomaus ja siirto ovat pois. Legendat, yksiköt, raja-arvot ja muut käyttöliittymässä tehdyt muutokset säilyvät sovelluksen istunnon ajan, eivät uudelleenkäynnistyksessä. Ne vaikuttavat esikatseluun ja eräajon PDF:iin.

## Skenaariot: `measurement_scenarios.json`

Tiedoston tallennettu muutos päivittää valinnat automaattisesti sovelluksen ollessa käynnissä; **Lataa asetukset uudelleen** tekee tämän myös käsin. Lataus odottaa käynnissä olevan esikatselun/eräajon loppumista. Muuttuneen skenaarion istunnossa tehdyt muutokset nollataan. Virheellisestä JSONista ilmoitetaan ja edelliset kelvolliset asetukset säilyvät; korjaa ja tallenna tiedosto. Käynnistyksessä molempien asetustiedostojen on oltava kelvollisia.

Tiedoston `scenarios`-listaan voi lisätä useita rivejä. `id` ja `suffix` sisältävät vain kirjaimia, numeroita, `_` tai `-`; `id` on yksilöllinen. `suffix` on PDF:n nimen loppuosa. `label` näkyy valikossa. Esimerkiksi seuraavassa skenaariossa on kaksi palstaa ja kaksi signaalia ensimmäisessä alikuvaajassa:

```json
{
  "scenarios": [
    {
      "id": "oma_koe",
      "label": "Oma koe",
      "kind": "configured",
      "suffix": "_oma_koe",
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

Esimerkki on kokonainen asetustiedosto; jos lisäät sen nykyiseen listaan, säilytä JSON-syntaksi. `columns` on 1–8, paneelit sijoitetaan riveittäin, ja tyhjät solut jätetään pois. Samassa paneelissa signaaliavaimen tulee olla yksilöllinen, mutta sama avain voi esiintyä eri paneeleissa. Signaalit luetaan kukin omalta aikarasteriltaan. Valitun tiedoston puuttuvasta signaalista ilmoitetaan; eräajossa muut tiedostot käsitellään edelleen.

Yksiköt kertovat **tallennetun signaalin** yksikön. `raw` jättää arvot ennalleen; muut vaihtoehdot ovat `A`, `V`, `rad/s` (sähköinen), `rad/s (mechanical)`, `rpm (mechanical)`, `Nm`, `Wb`, `Vs` sekä vastaavat esinormalisoidut `p.u. (A)`, `p.u. (V)`, `p.u. (rad/s)`, `p.u. (rad/s mechanical)`, `p.u. (Nm)`, `p.u. (Wb)` ja `p.u. (Vs)`. Mekaaninen rpm muunnetaan SI-esityksessä rad/s-arvoksi. `Wb` ja `Vs` ovat tässä saman suuruisia. Eri suureita voi asettaa samalle akselille, mutta niiden asteikkojen ja yksiköiden yhteensopivuus on käyttäjän vastuulla.

Laskenta tehdään järjestyksessä `scale * y + offset`, **ennen** yksikkömuunnosta. Sallitut lausekkeet ovat lukuja, sulkeita, `+ - * /` ja nimet `n_p`, `sqrt3`, `u_base`, `i_base`, `w_base`, `psi_base`, `tau_base`, `tau_nom`; ei Python-kutsuja. Jos asetat mekaanisen nopeuden yksiköksi `rad/s (mechanical)`, se normalisoidaan mekaanisella perusnopeudella; älä kerro sitä lisäksi napapariluvulla, ellei se ole tarkoituksesi.

Vanhojen `hfsi`, `overview`, `injection-overview` ja `presentation` -kuvien toteutukset ovat sovelluskoodissa; niiden asetustiedoston `kind` valitsee toteutuksen. Niihin voi asettaa `disabled_traces`, `added_traces`, `transforms` ja `units`. Alkuperäisten käyrien tunnisteet ovat `kind:panel:line`, indeksit alkavat nollasta. `added_traces`-alkio on esimerkiksi `{"id":"new", "panel":0, "key":"wm_meas"}`; `transforms` kuvaa tunnistetta muotoon `{"new":{"scale":"n_p","offset":"0"}}`, ja `units` esim. `{"wm_meas":"rad/s"}`. Täysin tiedostossa kuvattu asettelu tehdään `configured`-skenaariolla. Sovelluksen puussa näkyvät myös valmiiden kuvien mallikohtaiset aiemmat muunnokset; älä oleta niiden vastaavan kaikkia mittausmalleja.

## Moottorit: `motor_profiles.json`

`motors`-listaan voi lisätä yksilöllisen `id`-tunnisteen ja `name`-nimen sekä arvot `U_nom`, `I_nom`, `f_nom`, `P_nom`, `tau_nom` ja `n_p`. Arvojen on oltava positiivisia ja `n_p` kokonaisluku. Mukana oleva `default` on esimerkki, **ei yleispätevä moottorin nimellisarvo**. Profiilin valinta korvaa näytössä olevat arvot. Jos tiedosto muuttuu, muutetut profiilit päivittyvät; käyttäjän istunnossa käsin muokkaamia arvoja ei ylikirjoiteta automaattisesti.

## Motulator-yksikkökäytännöt

Yhteensopivat perussuureet noudattavat [motulatorin BaseValues-laskentaa](https://github.com/Aalto-Electric-Drives/motulator/blob/ba5ed869f3c8bf1884f126b8515e3655e1bcd6a2/motulator/common/utils/_utils.py#L203-L291): nimellisjännite on pääjännitteen tehollisarvo (V), nimellisvirta tehollisarvo (A), taajuus Hz, teho W, momentti Nm ja `n_p` napapariluku. Perusarvot ovat huippuarvoisia: `u_base = sqrt(2/3)*U_nom`, `i_base = sqrt(2)*I_nom`, `w_base = 2*pi*f_nom`, mekaaninen nopeusperusta `w_base/n_p`, `psi_base = u_base/w_base` ja `tau_base = n_p*(1.5*u_base*i_base)/w_base`. Nimellismomentti ja perusmomentti eivät ole sama asia. SI-esitys käyttää rad/s, A, V, Nm ja Vs/Wb; p.u.-esitys jakaa vastaavalla perusarvolla. Sähköisen ja mekaanisen nopeuden suhde on `w_m = n_p*w_M` ([motulatorin malli](https://github.com/Aalto-Electric-Drives/motulator/blob/ba5ed869f3c8bf1884f126b8515e3655e1bcd6a2/motulator/drive/model/_machine.py#L180-L190)). Tiedoston nimestä ei päätellä, onko tallennettu nopeus sähköinen vai mekaaninen.

## Transientit ja riippuvuudet

`references` etsii muutoksia ohjeista `wm_ref` ja `tauLref` / `TL_ref`. `spikes` käyttää käyttäjän valitsemaa signaalia ja raakayksiköissä annettua kynnystä: poikkeama koko signaalin mediaanista ylittää kynnyksen. Tämä ei ole asettumisajan tunnistus. Ilman tapahtumia käytetään tallennetta kokonaisuudessaan. LaTeX-julkaisutyyli vaatii erillisen TeX-asennuksen sekä `newtxtext`- ja `newtxmath`-paketit; `screen` ei vaadi niitä.

## Testit

```powershell
.\.venv\Scripts\python.exe -m unittest test_lab_plotter_config test_lab_plotter_core test_lab_plotter_app test_lab_plotting.TransientWindowTests test_lab_plotting.MeasurementSpikeTests test_lab_plotting.MeasurementFilesTests -q
```

Älä sisällytä luottamuksellisia mittauksia, PDF:iä tai virtuaaliympäristöä, kun jaat sovelluskansion. Tarkista jaettavien asetustiedostojen sisältö ennen lähettämistä.
