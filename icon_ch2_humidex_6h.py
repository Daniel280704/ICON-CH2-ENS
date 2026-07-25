import os
import sys
import time
import json
import requests
import urllib3
import pytz
import gc
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader

import earthkit.plots
from earthkit.plots.geo import bounds, domains
from earthkit.plots.styles import Style
from earthkit.data import config

from meteodatalab import ogd_api
from meteodatalab.operators import regrid
from rasterio.crs import CRS

warnings.filterwarnings('ignore')
urllib3.disable_warnings()
config.set("cache-policy", "temporary")

LATITUDE = 45.07
LONGITUDE = 7.54
FILE_LAST_HOUR = "ultima_ora_icon_ch2_humidex.txt"
RUN_DURATION = 120
START_DELAY = 1

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
            print(f"✅ Run ICON-CH2 {nome_run} Humidex già elaborato (Ultimo blocco: {ultima_ora_valida_str}).")
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
        "models": "meteoswiss_icon_ch2_ensemble_mean",
        "timezone": "Europe/Rome",
        "past_days": 1,
        "forecast_days": 6 
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
    thread_id = os.getenv("TELEGRAM_THREAD_ID_34") # Aggiornato al Thread 34
    
    if not token or not chat_id: return
    
    if len(file_paths) == 1:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        payload = {"chat_id": chat_id, "caption": caption}
        if thread_id: payload["message_thread_id"] = thread_id
        try:
            with open(file_paths[0], "rb") as photo:
                requests.post(url, data=payload, files={"photo": photo})
        except Exception as e: print(f"Errore invio singola foto: {e}")
        return

    url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
    media = []
    files = {}
    
    for idx, path in enumerate(file_paths):
        media.append({"type": "photo", "media": f"attach://photo_{idx}", "caption": caption if idx == 0 else ""})
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

def calcola_humidex(t_kelvin, td_kelvin):
    # Conversione in Celsius
    t_c = t_kelvin - 273.15
    td_c = td_kelvin - 273.15
    
    # Calcolo Tensione di Vapore (e) con formula di Tetens
    e = 6.112 * np.exp((17.67 * td_c) / (td_c + 243.5))
    
    # Calcolo Humidex
    humidex = t_c + (5.0 / 9.0) * (e - 10.0)
    
    # Se l'aria è secca, l'Humidex può risultare inferiore alla temperatura reale.
    # In quel caso, il disagio termico percepito è semplicemente pari alla temperatura reale.
    return np.where(humidex < t_c, t_c, humidex)

def genera_mappe_humidex(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    
    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    nx, ny = 300, 300
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

    # Scala dei colori per l'Humidex
    # <27: Normal, 27-29: Cautela, 30-34: Disagio, 35-39: Forte Disagio, 40-45: Pericolo, >45: Estremo
    my_levels = [20, 27, 30, 35, 40, 45, 55]
    my_colors = ["#a4f0b7", "#ffff00", "#ffcc00", "#ff6600", "#cc0000", "#800080"]
    domain = domains.Domain.from_bbox(bbox=bounds.BoundingBox(xmin, xmax, ymin, ymax, ccrs.Geodetic()), name="Piemonte")

    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    
    # Ciclo su blocchi di 6 ore
    for start_h in range(1, 120, 6):
        end_h = min(start_h + 5, 120)
        lead_times = [f"P{h//24}DT{h%24}H" for h in range(start_h, end_h + 1)]
        
        print(f"\nElaborazione blocco Humidex: +{start_h}h a +{end_h}h")
        
        req_t = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="T_2M", ref_time=dt_run_utc, perturbed=True, lead_time=lead_times)
        req_td = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="TD_2M", ref_time=dt_run_utc, perturbed=True, lead_time=lead_times)
        
        try:
            t_data = ogd_api.get_from_ogd(req_t)
            td_data = ogd_api.get_from_ogd(req_td)
            
            # Calcolo Humidex per ogni ora nel blocco da 6h
            humidex_hourly = calcola_humidex(t_data, td_data)
            
            # Ricava il picco massimo su 6 ore e media sugli scenari
            hx_max_6h = humidex_hourly.max(dim="lead_time")
            hx_final = hx_max_6h.mean(dim="eps")
            
        except Exception as e:
            print(f"Salto il blocco +{start_h}h/+{end_h}h causa errore download: {e}")
            continue

        hx_geo = regrid.iconremap(hx_final.squeeze(), destination)

        chart = earthkit.plots.Map(domain=domain)
        chart.grid_cells(hx_geo, x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))

        chart.ax.add_feature(regions_feature)
        chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())

        start_local = dt_run_local + timedelta(hours=start_h)
        end_local = dt_run_local + timedelta(hours=end_h)
        str_valida = f"{start_local.strftime('%H:%M del %d/%m')} - {end_local.strftime('%H:%M del %d/%m')}"

        title = f"ICON-CH2 EPS - Indice Disagio (Humidex)\nPicco Max nel blocco 6h | Run: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')}\n{str_valida}"
        chart.title(title)
        chart.legend(label="Humidex")

        filename = f"humidex_{start_h}_{end_h}.png"
        chart.save(filename)
        
        caption_album = f"Humidex 6h (Media Massimi)\nValido: {str_valida}\nRun {nome_run}"
        invia_album_telegram([filename], caption_album)
        
        if os.path.exists(filename): os.remove(filename)
        del t_data, td_data, humidex_hourly, hx_max_6h, hx_final, hx_geo
        gc.collect()
        time.sleep(15)

def main():
    print("Cerco l'ultimo run completo ICON-CH2 via Open-Meteo per Humidex...")
    data = fetch_dati_con_retry()
    if not data: sys.exit(0)
        
    hourly = data.get("hourly", {})
    utc_offset = data.get("utc_offset_seconds", 0)
    is_new, nome_run, dt_run_utc = estrai_limiti_run(hourly, "temperature_2m", utc_offset)
    
    if is_new:
        print(f"🚀 Lancio generazione Mappe Humidex per il RUN {nome_run} ({dt_run_utc})")
        genera_mappe_humidex(dt_run_utc, nome_run)
    else:
        print("Nessun nuovo run completo trovato. Uscita.")

if __name__ == "__main__":
    main()