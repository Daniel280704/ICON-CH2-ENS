import os
import sys
import time
import json
import requests
import urllib3
import pytz
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
FILE_LAST_HOUR = "ultima_ora_icon_ch2_hail_24h.txt"
RUN_DURATION = 120
START_DELAY = 1

def scarica_variabile_con_retry(request, max_retries=3, delay=5):
    """Tenta lo scaricamento fino a max_retries prima di sollevare eccezione."""
    for tentativo in range(max_retries):
        try:
            return ogd_api.get_from_ogd(request)
        except Exception as e:
            if tentativo == max_retries - 1:
                raise e
            time.sleep(delay)

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
            print(f"✅ Run ICON-CH2 {nome_run} già elaborato per la grandine (Ultimo blocco: {ultima_ora_valida_str}).")
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
    thread_id = os.getenv("TELEGRAM_THREAD_ID_22")
    
    if not token or not chat_id: return
    
    if len(file_paths) == 1:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        payload = {"chat_id": chat_id, "caption": caption}
        
        if thread_id:
            payload["message_thread_id"] = thread_id
            
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
    if thread_id:
        payload["message_thread_id"] = thread_id

    try:
        requests.post(url, data=payload, files=files)
        print(f"📸 Album Telegram inviato con successo ({len(file_paths)} mappe).")
    except Exception as e:
        print(f"Errore invio album Telegram: {e}")
    finally:
        for f in files.values():
            f.close()

def genera_album_giornalieri(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    
    intervals_by_day = {}
    last_h = 0
    
    for h in range(1, 121):
        dt_target = dt_run_local + timedelta(hours=h)
        if dt_target.hour % 3 == 0 or h == 120:
            if last_h < h:
                dt_start_interval = dt_run_local + timedelta(hours=last_h)
                day_str = dt_start_interval.strftime('%d/%m/%Y')
                
                if day_str not in intervals_by_day:
                    intervals_by_day[day_str] = []
                intervals_by_day[day_str].append((last_h, h))
            last_h = h

    lead_times_needed = list(range(1, 121))
    lead_times_str = [f"P{l // 24}DT{l % 24}H" for l in lead_times_needed]

    req = ogd_api.Request(
        collection="ogd-forecasting-icon-ch2",
        variable="DHAIL_MX", # Variabile ufficiale corretta per la grandine
        ref_time=dt_run_utc,
        perturbed=True,
        lead_time=lead_times_str,
    )
    
    try:
        print(f"  ⬇️  Scarico dati DHAIL_MX per le 120 ore...")
        hail_data = scarica_variabile_con_retry(req)
        print(f"  ✅ Dati scaricati con successo.")
    except IndexError:
        print("❌ Errore: Variabile non trovata nel catalogo o ore mancanti per DHAIL_MX.")
        return
    except Exception as e:
        print(f"  ❌ Errore nel download dopo 3 tentativi: {e}")
        return

    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    nx, ny = 300, 300
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

    my_levels = [0.5, 1, 2, 3, 4, 5, 7, 10, 15]
    my_colors = ["#a0e6ff", "#00a0ff", "#00ff00", "#ffff00", "#ffaa00", "#ff0000", "#ff00ff", "#ffffff"]
    domain = domains.Domain.from_bbox(bbox=bounds.BoundingBox(xmin, xmax, ymin, ymax, ccrs.Geodetic()), name="Piemonte")

    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    prov_feature = None
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    lats = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92]
    lons = [7.68,  7.55,  8.20,  8.61,  8.42,  8.61,  8.05,  8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    for day_str, intervals in intervals_by_day.items():
        percorsi_foto = []
        
        for h_start, h_end in intervals:
            hours_slice = [np.timedelta64(h, 'h') for h in range(h_start + 1, h_end + 1)]
            
            hail_interval = hail_data.sel(lead_time=hours_slice).max(dim="lead_time").mean(dim="eps")

            hail_geo = regrid.iconremap(hail_interval, destination)
            hail_geo = hail_geo.where(hail_geo >= 0.5)

            chart = earthkit.plots.Map(domain=domain)
            chart.grid_cells(hail_geo, x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))

            chart.ax.add_feature(regions_feature)
            if prov_feature:
                chart.ax.add_feature(prov_feature)
            else:
                chart.borders()

            chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())

            for lon, lat, sigla in zip(lons, lats, sigle):
                chart.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                chart.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

            start_local = dt_run_local + timedelta(hours=h_start)
            end_local = dt_run_local + timedelta(hours=h_end)
            
            orario_str = f"{start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M')}"
            title = f"ICON-CH2 EPS - Grandine (cm)\nMEDIA DEI MASSIMI | {day_str} Fascia {orario_str}\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')}"
            
            chart.title(title)
            chart.legend(label="Grandine (cm)")

            filename = f"hail_max_{h_start}_{h_end}.png"
            chart.save(filename)
            percorsi_foto.append(filename)
            
            plt.close(chart.fig)
        
        caption_album = f"ICON-CH2 EPS: Rischio Grandine (Media dei Massimi)\nFasce triorarie del {day_str}\nRun {nome_run}"
        invia_album_telegram(percorsi_foto, caption_album)
        
        for f in percorsi_foto:
            if os.path.exists(f): os.remove(f)

    del hail_data

def main():
    print("Cerco l'ultimo run completo ICON-CH2 via Open-Meteo per rischio grandine...")
    data = fetch_dati_con_retry()
    if not data: sys.exit(0)
        
    hourly = data.get("hourly", {})
    utc_offset = data.get("utc_offset_seconds", 0)
    is_new, nome_run, dt_run_utc = estrai_limiti_run(hourly, "temperature_2m", utc_offset)
    
    if is_new:
        print(f"🚀 Lancio generazione Album Grandine giornalieri/spezzoni per il RUN {nome_run} ({dt_run_utc})")
        genera_album_giornalieri(dt_run_utc, nome_run)
    else:
        print("Nessun nuovo run completo trovato. Uscita.")

if __name__ == "__main__":
    main()
