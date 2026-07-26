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
from meteodatalab.operators.destagger import destagger
from meteodatalab.operators.vertical_interpolation import interpolate_k2p
from rasterio.crs import CRS

warnings.filterwarnings('ignore')
urllib3.disable_warnings()
config.set("cache-policy", "temporary")

LATITUDE = 45.07
LONGITUDE = 7.54
FILE_LAST_HOUR = "ultima_ora_icon_ch2_updraft700.txt"
RUN_DURATION = 120
START_DELAY = 1

def scarica_variabile_con_retry(request, max_retries=3, delay=5):
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
            print(f"✅ Run ICON-CH2 {nome_run} Updraft già elaborato (Ultimo blocco: {ultima_ora_valida_str}).")
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
    thread_id = os.getenv("TELEGRAM_THREAD_ID_31")
    
    if not token or not chat_id: return
    
    if len(file_paths) == 1:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        payload = {"chat_id": chat_id, "caption": caption}
        if thread_id: payload["message_thread_id"] = thread_id
        try:
            with open(file_paths[0], "rb") as photo:
                requests.post(url, data=payload, files={"photo": photo})
        except Exception as e: print(f"Errore invio: {e}")
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
        print(f"📸 Album Updraft inviato ({len(file_paths)} mappe).")
    except Exception as e: print(f"Errore invio: {e}")
    finally:
        for f in files.values(): f.close()

def raggruppa_in_blocchi(dt_run_local: datetime) -> dict:
    blocchi = {}
    for h in range(3, 121, 3):
        dt_target = dt_run_local + timedelta(hours=h)
        if dt_target.hour == 0:
            date_str = (dt_target.date() - timedelta(days=1)).strftime("%d/%m/%Y")
        else:
            date_str = dt_target.date().strftime("%d/%m/%Y")
            
        key = f"Data: {date_str}"
        if key not in blocchi:
            blocchi[key] = []
        blocchi[key].append(h)
    return blocchi

def genera_album_orari(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    blocchi = raggruppa_in_blocchi(dt_run_local)

    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    nx, ny = 300, 300
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

    my_levels = [0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    my_colors = [
        "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", 
        "#f46d43", "#d73027", "#a50026", "#800026"
    ]
    
    domain = domains.Domain.from_bbox(bbox=bounds.BoundingBox(xmin, xmax, ymin, ymax, ccrs.Geodetic()), name="Piemonte")

    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    prov_feature = None
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    lats = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92]
    lons = [7.68,  7.55,  8.20,  8.61,  8.42,  8.61,  8.05,  8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    for block_name, ore_list in blocchi.items():
        print(f"\nGenerazione album Updraft per {block_name}")
        percorsi_foto = []

        for h in ore_list:
            # Creiamo la lista per le 3 ore (es. se h=6 -> ore 4, 5, 6)
            lead_times_3h = [f"P{step // 24}DT{step % 24}H" for step in range(h - 2, h + 1)]

            req_p = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="P", ref_time=dt_run_utc, perturbed=True, lead_time=lead_times_3h)
            req_w = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="W", ref_time=dt_run_utc, perturbed=True, lead_time=lead_times_3h)
            
            try:
                print(f"  ⬇️  Scarico dati P, W per la finestra ore [{h-2} - {h}]...")
                
                # Attenzione: Rimuoviamo il .mean() immediato qui per preservare l'EPS
                data_p = scarica_variabile_con_retry(req_p)
                data_w = scarica_variabile_con_retry(req_w)
                print(f"  ✅ Dati scaricati")
                
                # Allineamento verticale
                data_w_hfl = destagger(data_w, "z")
                
                # Interpolazione a 700 hPa per tutti i membri EPS e per tutte e 3 le ore
                w_700 = interpolate_k2p(field=data_w_hfl, mode="linear_in_lnp", p_field=data_p, p_tc_values=[700], p_tc_units="hPa")
                
                # Trova dinamicamente il nome della dimensione temporale
                time_dims = [d for d in w_700.dims if d in ["time", "valid_time", "step", "lead_time"]]
                time_dim = time_dims[0] if time_dims else "time"
                
                # ---> LA MODIFICA CHIAVE <---
                # 1. Trova il valore massimo su ogni punto griglia nelle 3 ore (dimensione temporale)
                # 2. Calcola la media dei massimi estratti per ogni scenario (dimensione eps)
                w_700_max_mean = w_700.max(dim=time_dim).mean(dim="eps")

            except Exception as e:
                print(f"  ❌ Salto la finestra dell'ora {h} dopo 3 tentativi. Errore: {e}")
                continue

            # Mappiamo il DataArray risultante, eliminando le dimensioni superflue come p_tc
            w_700_geo = regrid.iconremap(w_700_max_mean.squeeze(drop=True), destination)

            chart = earthkit.plots.Map(domain=domain)
            chart.grid_cells(w_700_geo, x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))

            chart.ax.add_feature(regions_feature)
            if prov_feature: chart.ax.add_feature(prov_feature)
            else: chart.borders()

            chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())

            for lon, lat, sigla in zip(lons, lats, sigle):
                chart.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                chart.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

            # Titolo aggiornato
            target_local = dt_run_local + timedelta(hours=h)
            start_local = dt_run_local + timedelta(hours=h-3)
            str_valida = f"Valido da: {start_local.strftime('%H:00')} a {target_local.strftime('%H:00 del %d/%m')}"
            title = f"ICON-CH2 EPS - Updraft (W) a 700 hPa (m/s)\nMEDIA DEI MASSIMI SULLE 3 ORE | Run: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')}\n{str_valida}"
            
            chart.title(title)
            chart.legend(label="Updraft (m/s)")

            filename = f"updraft700_{h}.png"
            chart.save(filename)
            percorsi_foto.append(filename)
            plt.close(chart.fig)
            
            # Pulizia per gestire la RAM
            del data_p, data_w, data_w_hfl, w_700, w_700_max_mean, w_700_geo
            gc.collect()
        
        if percorsi_foto:
            caption_album = f"Updraft (W) 700 hPa Media Massimi (3h)\n{block_name}\nRun {nome_run}"
            invia_album_telegram(percorsi_foto, caption_album)
            
            for f in percorsi_foto:
                if os.path.exists(f): os.remove(f)
            time.sleep(15)

def main():
    print("Cerco l'ultimo run completo ICON-CH2 per Updraft (Step 3h, Media Scenari)...")
    data = fetch_dati_con_retry()
    if not data: sys.exit(0)
        
    hourly = data.get("hourly", {})
    utc_offset = data.get("utc_offset_seconds", 0)
    is_new, nome_run, dt_run_utc = estrai_limiti_run(hourly, "temperature_2m", utc_offset)
    
    if is_new:
        print(f"🚀 Lancio generazione Album Updraft per il RUN {nome_run}")
        genera_album_orari(dt_run_utc, nome_run)
    else:
        print("Nessun nuovo run completo trovato. Uscita.")

if __name__ == "__main__":
    main()
