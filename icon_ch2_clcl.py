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

import earthkit.data
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
FILE_LAST_HOUR = "ultima_ora_icon_ch2_clcl.txt"
RUN_DURATION = 120
START_DELAY = 1

def scarica_variabile_con_retry(request, max_retries=6):
    """Tenta lo scaricamento fino a max_retries, svuotando la cache in caso di fallimento."""
    for tentativo in range(max_retries):
        try:
            if tentativo > 0:
                print(f"    🔄 Retry {tentativo + 1}/{max_retries} in corso...")
            
            # Tenta il download vero e proprio
            return ogd_api.get_from_ogd(request)
            
        except Exception as e:
            if tentativo == max_retries - 1:
                print(f"    💥 Fallimento definitivo dopo {max_retries} tentativi.")
                raise e
            
            # Pausa progressiva (10s, 20s, 30s...) per far sbloccare il server
            delay = 10 * (tentativo + 1)
            print(f"    ⚠️ Errore o blocco di rete: {e}")
            print(f"    🧹 Svuoto la cache corrotta e attendo {delay}s...")
            
            # FORZATURA: Svuota la cache di earthkit per eliminare i file a metà
            try:
                earthkit.data.cache.purge()
            except Exception:
                pass
                
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
    
    try: start_idx = times.index(start_time_str)
    except ValueError: return False, "", None
        
    expected_points = RUN_DURATION - START_DELAY + 1
    actual_points = end_idx - start_idx + 1
    
    if actual_points < expected_points: return False, "", None
        
    if os.path.exists(FILE_LAST_HOUR):
        with open(FILE_LAST_HOUR, "r") as f:
            ultima_ora_salvata = f.read().strip()
        if ultima_ora_valida_str <= ultima_ora_salvata: return False, "", None

    with open(FILE_LAST_HOUR, "w") as f: f.write(ultima_ora_valida_str)
    return True, nome_run, dt_run_utc_naive.replace(tzinfo=timezone.utc)

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
        except: time.sleep(15)
    return {}

def invia_album_telegram(file_paths: list, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    thread_id = os.getenv("TELEGRAM_THREAD_ID_15")
    
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
    media = []
    files = {}
    for idx, path in enumerate(file_paths):
        media.append({"type": "photo", "media": f"attach://photo_{idx}", "caption": caption if idx == 0 else ""})
        files[f"photo_{idx}"] = open(path, "rb")

    payload = {"chat_id": chat_id, "media": json.dumps(media)}
    if thread_id: payload["message_thread_id"] = thread_id

    try: requests.post(url, data=payload, files=files)
    except: pass
    finally:
        for f in files.values(): f.close()

def raggruppa_in_blocchi(dt_run_local: datetime) -> dict:
    blocchi = {}
    for h in range(1, 121):
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
        if key not in blocchi: blocchi[key] = []
        blocchi[key].append(h)
    return blocchi

def genera_album_orari(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    blocchi = raggruppa_in_blocchi(dt_run_local)

    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), 300, 300, xmin, xmax, ymin, ymax)

    my_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    my_colors = ["#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373"]
    domain = domains.Domain.from_bbox(bbox=bounds.BoundingBox(xmin, xmax, ymin, ymax, ccrs.Geodetic()), name="Piemonte")

    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    prov_feature = None
    if os.path.exists("shapefiles/ProvCM01012026_WGS84.shp"):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader("shapefiles/ProvCM01012026_WGS84.shp").geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    for block_name, ore_list in blocchi.items():
        req = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="CLCL", ref_time=dt_run_utc, perturbed=True, lead_time=[f"P{l // 24}DT{l % 24}H" for l in ore_list])
        try:
            print(f"  ⬇️  Scarico dati CLCL per {len(ore_list)} ore...")
            var_raw = scarica_variabile_con_retry(req)
            var_mean = var_raw.mean(dim="eps")
            print(f"  ✅ Dati scaricati: {len(ore_list)} ore")
        except Exception as e:
            print(f"  ❌ Salto il blocco {block_name} dopo 3 tentativi. Errore: {e}")
            continue

        percorsi_foto = []
        for h in ore_list:
            var_geo = regrid.iconremap(var_mean.sel(lead_time=np.timedelta64(h, 'h')), destination)
            chart = earthkit.plots.Map(domain=domain)
            chart.grid_cells(var_geo, x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))
            chart.ax.add_feature(regions_feature)
            if prov_feature: chart.ax.add_feature(prov_feature)
            else: chart.borders()

            target_local = dt_run_local + timedelta(hours=h)
            chart.title(f"ICON-CH2 EPS - Cloud Cover Low (%)\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')} | Ore {target_local.strftime('%H:%M del %d/%m')}")
            chart.legend(label="CLCL (%)")

            filename = f"oraria_{h}.png"
            chart.save(filename)
            percorsi_foto.append(filename)
            plt.close(chart.fig)
        
        invia_album_telegram(percorsi_foto, f"ICON-CH2 EPS: Dettaglio Nubi Basse (CLCL)\n{block_name}\nRun {nome_run}")
        for f in percorsi_foto:
            if os.path.exists(f): os.remove(f)
        time.sleep(15)

def main():
    data = fetch_dati_con_retry()
    if not data: sys.exit(0)
    is_new, nome_run, dt_run_utc = estrai_limiti_run(data.get("hourly", {}), "temperature_2m", data.get("utc_offset_seconds", 0))
    if is_new: genera_album_orari(dt_run_utc, nome_run)

if __name__ == "__main__": main()