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

LATITUDE = 45.07
LONGITUDE = 7.54
FILE_LAST_HOUR = "ultima_ora_icond2_prob.txt" 
RUN_DURATION = 27 # ICON-D2 si ferma a 27h
START_DELAY = 1

def scarica_variabile_icon_d2(dt_run_utc, ore_list, max_retries=3):
    """Scarica i file .bz2 dal server DWD OpenData, li decomprime in GRIB2 e li carica in xarray."""
    run_hour = dt_run_utc.strftime('%H')
    date_hour = dt_run_utc.strftime('%Y%m%d%H')
    
    file_temporanei = []
    
    for h in ore_list:
        step_str = f"{h:03d}"
        url = f"https://opendata.dwd.de/weather/nwp/icon-d2-eps/grib/{run_hour}/tot_prec/icon-d2-eps_germany_icosahedral_single-level_{date_hour}_{step_str}_2d_tot_prec.grib2.bz2"
        
        for tentativo in range(max_retries):
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                
                # Decomprimo il bz2 on the fly e scrivo un GRIB2 temporaneo
                fd, temp_path = tempfile.mkstemp(suffix=".grib2")
                with os.fdopen(fd, 'wb') as f_out:
                    decompressor = bz2.BZ2Decompressor()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f_out.write(decompressor.decompress(chunk))
                
                file_temporanei.append(temp_path)
                break
            except Exception as e:
                if tentativo == max_retries - 1:
                    print(f"    💥 Fallimento definitivo per ora {h} dopo {max_retries} tentativi: {e}")
                    raise e
                time.sleep(10 * (tentativo + 1))
                
    # Carico tutti i grib temporanei in un unico Dataset xarray
    ds = earthkit.data.from_source("file", file_temporanei).to_xarray()
    
    # Rinomino la dimensione dei membri per uniformità se necessario (cfgrib usa 'number')
    if 'number' in ds.dims:
        ds = ds.rename({'number': 'eps'})
        
    return ds, file_temporanei

def estrai_limiti_run(hourly_data: dict, ref_param: str, utc_offset_sec: int) -> tuple[bool, str, datetime]:
    times = hourly_data.get("time", [])
    mean_vals = hourly_data.get(ref_param, [])
    if not times or not mean_vals: return False, "", None

    end_idx = -1
    for i in range(len(mean_vals) - 1, -1, -1):
        if mean_vals[i] is not None:
            end_idx = i
            break

    if end_idx == -1: return False, "", None

    ultima_ora_valida_str = times[end_idx]

    dt_end_local = datetime.fromisoformat(ultima_ora_valida_str)
    dt_end_utc = dt_end_local - timedelta(seconds=utc_offset_sec)
    dt_run_utc_naive = dt_end_utc - timedelta(hours=RUN_DURATION)
    dt_start_utc = dt_run_utc_naive + timedelta(hours=START_DELAY)

    dt_start_local = dt_start_utc + timedelta(seconds=utc_offset_sec)
    start_time_str = dt_start_local.strftime("%Y-%m-%dT%H:%M")
    nome_run = dt_run_utc_naive.strftime("%H") + "Z"

    try:
        start_idx = times.index(start_time_str)
    except ValueError:
        return False, "", None

    expected_points = RUN_DURATION - START_DELAY + 1
    actual_points = end_idx - start_idx + 1

    if actual_points < expected_points:
        print(f"⏳ Run {nome_run} in caricamento... ({actual_points}/{expected_points} ore)")
        return False, "", None

    if os.path.exists(FILE_LAST_HOUR):
        with open(FILE_LAST_HOUR, "r") as f:
            ultima_ora_salvata = f.read().strip()
        if ultima_ora_valida_str <= ultima_ora_salvata:
            print(f"✅ Run ICON-D2 EPS {nome_run} (Probabilità) già elaborato (Ultimo blocco: {ultima_ora_valida_str}).")
            return False, "", None

    with open(FILE_LAST_HOUR, "w") as f:
        f.write(ultima_ora_valida_str)

    dt_run_utc = dt_run_utc_naive.replace(tzinfo=timezone.utc)
    return True, nome_run, dt_run_utc

def fetch_dati_con_retry() -> dict:
    URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m",
        "models": "icon_d2",
        "timezone": "Europe/Rome",
        "past_days": 1,
        "forecast_days": 3 
    }
    for _ in range(3):
        try:
            r = requests.get(URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            time.sleep(15)
    return {}

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
    for h in range(1, 28): # Limitato a 27h
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
            print(f"  ⬇️  Scarico dati TOT_PREC dal DWD per {len(lead_times_needed)} ore...")
            tot_prec_xr, tmp_files = scarica_variabile_icon_d2(dt_run_utc, lead_times_needed)
            print(f"  ✅ Dati scaricati e decodificati.")
        except Exception as e:
            print(f"  ❌ Salto il blocco {block_name}. Errore: {e}")
            continue

        percorsi_foto = []

        for h in ore_list:
            step_timedelta = np.timedelta64(h, 'h')
            step_prev_timedelta = np.timedelta64(h-1, 'h')
            
            try:
                if h == 1:
                    prec_diff = tot_prec_xr.sel(step=step_timedelta).tot_prec
                else:
                    prec_diff = tot_prec_xr.sel(step=step_timedelta).tot_prec - tot_prec_xr.sel(step=step_prev_timedelta).tot_prec
            except KeyError:
                print(f"Ora {h} non trovata, salto.")
                continue

            prob_xr = (prec_diff >= 0.5).astype(float).mean(dim="eps") * 100
            prob_xr = prob_xr.where(prob_xr >= 10)

            chart = earthkit.plots.Map(domain=domain)
            
            # Passando direttamente la variabile Xarray, earthkit mappa la griglia nativa DWD
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
        del tot_prec_xr
        time.sleep(15)

def main():
    print("Cerco l'ultimo run completo ICON-D2 via Open-Meteo per le probabilità...")
    data = fetch_dati_con_retry()
    if not data: sys.exit(0)

    hourly = data.get("hourly", {})
    utc_offset = data.get("utc_offset_seconds", 0)
    is_new, nome_run, dt_run_utc = estrai_limiti_run(hourly, "temperature_2m", utc_offset)

    if is_new:
        print(f"🚀 Lancio generazione Probabilità Orarie ICON-D2 per il RUN {nome_run} ({dt_run_utc})")
        genera_album_orari(dt_run_utc, nome_run)
    else:
        print("Nessun nuovo run completo trovato. Uscita.")

if __name__ == "__main__":
    main()
