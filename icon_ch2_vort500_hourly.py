import os
import sys
import time
import json
import requests
import urllib3
import pytz
import gc
import numpy as np
import scipy.ndimage
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
from meteodatalab.operators.vertical_interpolation import interpolate_k2p
from rasterio.crs import CRS

warnings.filterwarnings('ignore')
urllib3.disable_warnings()
config.set("cache-policy", "temporary")

LATITUDE = 45.07
LONGITUDE = 7.54
FILE_LAST_HOUR = "ultima_ora_icon_ch2_vort500.txt"
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
            print(f"✅ Run ICON-CH2 {nome_run} VORT500 già elaborato (Ultimo blocco: {ultima_ora_valida_str}).")
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
    thread_id = os.getenv("TELEGRAM_THREAD_ID_5662")
    
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
        print(f"📸 Album VORT500 inviato ({len(file_paths)} mappe).")
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

    # --- SCALA ALLARGATA PER LA VORTICITÀ ---
    # Passi piccoli al centro (-10, -5, 0, 5, 10) e passi ampi agli estremi (20, 40, 60) per assorbire i picchi
    my_levels = [-60, -40, -20, -10, -5, 0, 5, 10, 20, 40, 60]
    
    # 10 colori abbinati ai 10 intervalli (i due centrali restano bianchi per neutralità attorno allo zero)
    my_colors = [
        "#000066", # da -60 a -40 (Anticiclonico molto forte)
        "#0033cc", # da -40 a -20
        "#4488ff", # da -20 a -10
        "#aaddff", # da -10 a -5
        "#ffffff", # da -5 a 0   (Neutro)
        "#ffffff", # da 0 a 5    (Neutro)
        "#ffcc00", # da 5 a 10
        "#ff6600", # da 10 a 20
        "#cc0000", # da 20 a 40
        "#800080"  # da 40 a 60  (Ciclonico molto forte)
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

    # Pre-calcolo delle distanze dx e dy (in metri)
    R_earth = 6371000.0
    lat_1d = np.linspace(ymin, ymax, ny)
    lon_1d = np.linspace(xmin, xmax, nx)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
    
    dy_meters = R_earth * np.deg2rad((ymax - ymin) / (ny - 1))
    dx_meters = R_earth * np.cos(np.deg2rad(lat2d)) * np.deg2rad((xmax - xmin) / (nx - 1))

    for block_name, ore_list in blocchi.items():
        print(f"\nGenerazione album VORT500 per {block_name}")
        percorsi_foto = []

        for h in ore_list:
            print(f"Scaricamento e calcolo vorticità ensemble per l'ora +{h}...")
            lead_time_str = [f"P{h // 24}DT{h % 24}H"]

            req_p = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="P", ref_time=dt_run_utc, perturbed=True, lead_time=lead_time_str)
            req_u = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="U", ref_time=dt_run_utc, perturbed=True, lead_time=lead_time_str)
            req_v = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="V", ref_time=dt_run_utc, perturbed=True, lead_time=lead_time_str)
            
            try:
                data_p = ogd_api.get_from_ogd(req_p).mean(dim="eps")
                data_u = ogd_api.get_from_ogd(req_u).mean(dim="eps")
                data_v = ogd_api.get_from_ogd(req_v).mean(dim="eps")
                
                u_500 = interpolate_k2p(field=data_u, mode="linear_in_p", p_field=data_p, p_tc_values=[500], p_tc_units="hPa")
                v_500 = interpolate_k2p(field=data_v, mode="linear_in_p", p_field=data_p, p_tc_values=[500], p_tc_units="hPa")

            except Exception as e:
                print(f"Salto l'ora {h} causa errore: {e}")
                continue

            u_500_geo = regrid.iconremap(u_500.squeeze(drop=True), destination)
            v_500_geo = regrid.iconremap(v_500.squeeze(drop=True), destination)

            # CALCOLO VORTICITÀ
            du_dy_idx, du_dx_idx = np.gradient(u_500_geo.values)
            dv_dy_idx, dv_dx_idx = np.gradient(v_500_geo.values)

            du_dy = du_dy_idx / dy_meters
            dv_dx = dv_dx_idx / dx_meters

            # Vorticità relativa grezza
            vort_500_np_raw = (dv_dx - du_dy) * 1e5

            # FILTRO GAUSSIANO per ridurre il rumore del LAM
            vort_500_np_smoothed = scipy.ndimage.gaussian_filter(vort_500_np_raw, sigma=3)

            # Re-incapsuliamo i risultati nel DataArray xarray
            vort_500_xr = u_500_geo.copy(data=vort_500_np_smoothed)

            chart = earthkit.plots.Map(domain=domain)
            chart.grid_cells(vort_500_xr, x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))

            chart.ax.add_feature(regions_feature)
            if prov_feature: chart.ax.add_feature(prov_feature)
            else: chart.borders()

            chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())

            for lon, lat, sigla in zip(lons, lats, sigle):
                chart.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                chart.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

            target_local = dt_run_local + timedelta(hours=h)
            str_valida = f"Valido per le: {target_local.strftime('%H:%M del %d/%m')}"
            title = f"ICON-CH2 EPS - Vorticità Relativa 500 hPa (10^-5 s^-1)\nMEDIA SCENARI (Smoothed) | Run: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')}\n{str_valida}"
            
            chart.title(title)
            chart.legend(label="Vorticity (10^-5 s^-1)")

            filename = f"vort500_{h}.png"
            chart.save(filename)
            percorsi_foto.append(filename)
            plt.close(chart.fig)
            
            del data_p, data_u, data_v, u_500, v_500, u_500_geo, v_500_geo, vort_500_np_raw, vort_500_np_smoothed, vort_500_xr
            gc.collect()
        
        if percorsi_foto:
            caption_album = f"Vorticità 500 hPa Media EPS\n{block_name}\nRun {nome_run}"
            invia_album_telegram(percorsi_foto, caption_album)
            
            for f in percorsi_foto:
                if os.path.exists(f): os.remove(f)
            time.sleep(15)

def main():
    print("Cerco l'ultimo run completo ICON-CH2 per VORT500 (Step 3h, Media Scenari)...")
    data = fetch_dati_con_retry()
    if not data: sys.exit(0)
        
    hourly = data.get("hourly", {})
    utc_offset = data.get("utc_offset_seconds", 0)
    is_new, nome_run, dt_run_utc = estrai_limiti_run(hourly, "temperature_2m", utc_offset)
    
    if is_new:
        print(f"🚀 Lancio generazione Album VORT500 per il RUN {nome_run}")
        genera_album_orari(dt_run_utc, nome_run)
    else:
        print("Nessun nuovo run completo trovato. Uscita.")

if __name__ == "__main__":
    main()
