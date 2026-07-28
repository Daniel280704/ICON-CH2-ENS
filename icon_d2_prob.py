import os
import sys
import time
import json
import requests
import urllib3
import pytz
import bz2
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import xarray as xr

import earthkit.data
import earthkit.plots
from earthkit.plots.styles import Style
from earthkit.data import config

warnings.filterwarnings('ignore')
urllib3.disable_warnings()
config.set("cache-policy", "temporary")

FILE_LAST_HOUR = "ultima_ora_icond2_prob.txt" 
RUN_DURATION = 48 # Esteso a 48h
START_DELAY = 1

def scarica_pioggia_icon_d2(dt_run_utc, ore_list, max_retries=3):
    """
    Scarica rain_gsp e rain_con dal server DWD OpenData, decomprime i .bz2 in GRIB2 temporanei
    e li carica in due Dataset xarray separati.
    """
    run_hour_syn = dt_run_utc.hour          
    run_hour = f"{run_hour_syn:02d}"
    date_hour = dt_run_utc.strftime('%Y%m%d%H')
    tmp_gsp = []
    tmp_con = []

    def _download_one(url: str, max_retries: int):
        for tentativo in range(max_retries):
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()

                fd, temp_path = tempfile.mkstemp(suffix=".grib2")
                with os.fdopen(fd, 'wb') as f_out:
                    decompressor = bz2.BZ2Decompressor()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f_out.write(decompressor.decompress(chunk))
                return temp_path
            except Exception as e:
                if tentativo == max_retries - 1:
                    raise e
                time.sleep(10 * (tentativo + 1))

    for h in ore_list:
        step_idx = max(0, h - 1)   # h=1 -> 000, h=2 -> 001, ...
        step_str = f"{step_idx:03d}"
        
        # Prefisso corretto: icon-d2-eps_
        url_gsp = f"https://opendata.dwd.de/weather/nwp/icon-d2-eps/grib/{run_hour}/rain_gsp/icon-d2-eps_germany_regular-lat-lon_single-level_{date_hour}_{step_str}_2d_rain_gsp.grib2.bz2"
        url_con = f"https://opendata.dwd.de/weather/nwp/icon-d2-eps/grib/{run_hour}/rain_con/icon-d2-eps_germany_regular-lat-lon_single-level_{date_hour}_{step_str}_2d_rain_con.grib2.bz2"

        try:
            p_gsp = _download_one(url_gsp, max_retries)
            tmp_gsp.append(p_gsp)
        except Exception as e:
            print(f"    💥 Fallimento definitivo rain_gsp ora {h}: {e}")
            raise

        try:
            p_con = _download_one(url_con, max_retries)
            tmp_con.append(p_con)
        except Exception as e:
            print(f"    💥 Fallimento definitivo rain_con ora {h}: {e}")
            raise

    ds_gsp = earthkit.data.from_source("file", tmp_gsp).to_xarray()
    ds_con = earthkit.data.from_source("file", tmp_con).to_xarray()

    if 'number' in ds_gsp.dims:
        ds_gsp = ds_gsp.rename({'number': 'eps'})
    if 'number' in ds_con.dims:
        ds_con = ds_con.rename({'number': 'eps'})

    return ds_gsp, ds_con, (tmp_gsp + tmp_con)

def get_latest_dwd_run():
    """
    Calcola l'ultimo run ICON-D2 EPS verificandolo direttamente sul server DWD.
    Gestisce i run ogni 3 ore (00, 03, 06, 09, 12, 15, 18, 21).
    """
    now = datetime.now(timezone.utc)
    # I file escono con circa 1.5 - 2 ore di ritardo. Usiamo 2.5 ore di margine di sicurezza.
    dt_safe = now - timedelta(hours=2, minutes=30)
    
    # Arrotondamento ai blocchi di 3 ore
    run_hour_syn = (dt_safe.hour // 3) * 3  
    dt_run = dt_safe.replace(hour=run_hour_syn, minute=0, second=0, microsecond=0)

    for attempt in range(3):
        date_hour = dt_run.strftime('%Y%m%d%H')
        run_hour_str = f"{dt_run.hour:02d}"

        # Testiamo lo step 000 con il nome corretto
        url_test = f"https://opendata.dwd.de/weather/nwp/icon-d2-eps/grib/{run_hour_str}/rain_gsp/icon-d2-eps_germany_regular-lat-lon_single-level_{date_hour}_000_2d_rain_gsp.grib2.bz2"
        print(f"  🔍 Controllo server DWD per run {run_hour_str}Z...")

        try:
            r = requests.head(url_test, timeout=10)
            if r.status_code == 200:
                print(f"  🟢 File trovato! Il run {run_hour_str}Z è online.")
                nome_run = dt_run.strftime("%H") + "Z"

                if os.path.exists(FILE_LAST_HOUR):
                    with open(FILE_LAST_HOUR, "r") as f:
                        ultimo_salvato = f.read().strip()
                    if date_hour <= ultimo_salvato:
                        print(f"  ✅ Run ICON-D2 EPS {nome_run} già elaborato in precedenza (Ultimo in archivio: {ultimo_salvato}).")
                        return False, "", None

                with open(FILE_LAST_HOUR, "w") as f:
                    f.write(date_hour)

                return True, nome_run, dt_run
            else:
                print(f"  ⚠️ Run {run_hour_str}Z non ancora disponibile (Errore HTTP {r.status_code})")
        except Exception as e:
            print(f"  ❌ Errore di connessione testando {run_hour_str}Z: {e}")

        # Se non è ancora online o dà 404, scaliamo al run di 3 ore prima
        dt_run -= timedelta(hours=3)

    return False, "", None

def invia_album_telegram(file_paths: list, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    thread_id = os.getenv("TELEGRAM_THREAD_ID_5685")

    if not token or not chat_id: return

    if len(file_paths) == 1:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        payload = {"chat_id": chat_id, "caption": caption}
        if thread_id: payload["message_thread_id"] = thread_id

        try:
            with open(file_paths[0], "rb") as photo:
                requests.post(url, data=payload, files={"photo": photo})
        except Exception as e:
            print(f"Errore invio singola foto: {e}")
        return

    url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
    media = []
    files = {}

    for idx, path in enumerate(file_paths):
        media.append({
            "type": "photo",
            "media": f"attach://photo_{idx}",
            "caption": caption if idx == 0 else ""
        })
        files[f"photo_{idx}"] = open(path, "rb")

    payload = {"chat_id": chat_id, "media": json.dumps(media)}
    if thread_id: payload["message_thread_id"] = thread_id

    try:
        requests.post(url, data=payload, files=files)
        print(f"📸 Album Telegram inviato con successo ({len(file_paths)} mappe).")
    except Exception as e:
        print(f"Errore invio album Telegram: {e}")
    finally:
        for f in files.values():
            f.close()

def raggruppa_in_blocchi(dt_run_local: datetime) -> dict:
    blocchi = {}
    for h in range(1, 49): # Portato a 48h
        dt_target = dt_run_local + timedelta(hours=h)
        date_str = dt_target.date().strftime("%Y-%m-%d")
        hour = dt_target.hour

        if hour == 0:
            date_str = (dt_target.date() - timedelta(days=1)).strftime("%Y-%m-%d")
            b_name = "18-24"
        elif 1 <= hour <= 6: b_name = "00-06"
        elif 7 <= hour <= 12: b_name = "06-12"
        elif 13 <= hour <= 18: b_name = "12-18"
        else: b_name = "18-24"

        key = f"{date_str} (Fascia {b_name})"
        if key not in blocchi:
            blocchi[key] = []
        blocchi[key].append(h)
    return blocchi

def genera_album_orari(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    blocchi = raggruppa_in_blocchi(dt_run_local)

    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    my_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    my_colors = ["#a0e6ff", "#00a0ff", "#00ff00", "#ffff00", "#ffaa00", "#ff0000", "#cc0000", "#ff00ff", "#800080"]

    domain = [xmin, xmax, ymin, ymax]
    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)

    prov_feature = None
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    lats = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92]
    lons = [7.68,  7.55,  8.20,  8.61,  8.42,  8.61,  8.05,  8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    for block_name, ore_list in blocchi.items():
        print(f"\nGenerazione album probabilità: {block_name}")
        lead_times_needed = list(ore_list)
        if ore_list[0] > 1:
            lead_times_needed.insert(0, ore_list[0] - 1)

        try:
            print(f"  ⬇️  Scarico dati RAIN_GSP + RAIN_CON dal DWD per {len(lead_times_needed)} ore...")
            rain_gsp_xr, rain_con_xr, tmp_files = scarica_pioggia_icon_d2(dt_run_utc, lead_times_needed)
            print(f"  ✅ Dati scaricati e decodificati.")
        except Exception as e:
            print(f"  ❌ Salto il blocco {block_name}. Errore: {e}")
            continue

        percorsi_foto = []

        for h in ore_list:
            step_timedelta = np.timedelta64(h, 'h')

            try:
               gsp_now = rain_gsp_xr.sel(step=step_timedelta).rain_gsp
               con_now = rain_con_xr.sel(step=step_timedelta).rain_con
               prec_diff = gsp_now + con_now
            except KeyError:
                print(f"Ora {h} non trovata, salto.")
                continue

            prob_xr = (prec_diff >= 0.5).astype(float).mean(dim="eps") * 100
            prob_xr = prob_xr.where(prob_xr >= 10)

            chart = earthkit.plots.Map(domain=domain)

            chart.grid_cells(prob_xr, style=Style(colors=my_colors, levels=my_levels))

            chart.ax.add_feature(regions_feature)
            if prov_feature: chart.ax.add_feature(prov_feature)
            else: chart.borders()

            chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())
            for lon, lat, sigla in zip(lons, lats, sigle):
                chart.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                chart.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

            start_local = dt_run_local + timedelta(hours=h-1)
            end_local = dt_run_local + timedelta(hours=h)
            str_valida = f"{start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M del %d/%m')}"

            title = f"ICON-D2 EPS - Probabilità Pioggia >= 0.5 mm/h (%)\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')} | {str_valida}"
            chart.title(title)
            chart.legend(label="Probabilità (%)")

            filename = f"oraria_{h}.png"
            chart.save(filename)
            percorsi_foto.append(filename)
            plt.close(chart.fig)

        caption_album = f"ICON-D2 EPS: Probabilità Pioggia oraria >= 0.5 mm\n{block_name}\nRun {nome_run}"
        invia_album_telegram(percorsi_foto, caption_album)

        for f in percorsi_foto + tmp_files:
            if os.path.exists(f): os.remove(f)
        del rain_gsp_xr, rain_con_xr
        time.sleep(15)

def main():
    print("Cerco l'ultimo run completo ICON-D2 EPS direttamente sul server DWD...")
    is_new, nome_run, dt_run_utc = get_latest_dwd_run()

    if is_new:
        print(f"🚀 Lancio generazione Probabilità Orarie ICON-D2 per il RUN {nome_run} ({dt_run_utc.strftime('%Y-%m-%d %H:%M')})")
        genera_album_orari(dt_run_utc, nome_run)
    else:
        print("Nessun nuovo run trovato. Uscita.")

if __name__ == "__main__":
    main()
