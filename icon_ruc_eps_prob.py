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
from matplotlib.colors import BoundaryNorm, ListedColormap
from datetime import datetime, timedelta, timezone
import warnings

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import xarray as xr

import earthkit.data
from earthkit.data import config

warnings.filterwarnings('ignore')
urllib3.disable_warnings()
config.set("cache-policy", "temporary")

LATITUDE = 45.07
LONGITUDE = 7.54

FILE_LAST_HOUR = "ultima_ora_icon_ruc_eps_prob.txt" 
RUN_DURATION = 27 # ICON-D2-RUC si ferma a 27h
START_DELAY = 0

def get_latest_ruc_run():
    """
    Calcola l'ultimo run ICON-D2-RUC EPS basandosi sull'ora UTC attuale.
    Interroga direttamente il DWD per run orari (00, 01, 02... 23).
    """
    now = datetime.now(timezone.utc)
    # I file RUC escono molto velocemente, usiamo 1.5 ore di margine di sicurezza
    dt_safe = now - timedelta(hours=1, minutes=30)
    dt_run = dt_safe.replace(minute=0, second=0, microsecond=0)

    for attempt in range(4): # Proviamo fino a 4 ore indietro
        date_hour = dt_run.strftime('%Y%m%d%H')
        run_hour_str = f"{dt_run.hour:02d}"

        # URL di test per TOT_PREC allo step 001
        url_test = f"https://opendata.dwd.de/weather/nwp/icon-d2-ruc-eps/grib/{run_hour_str}/tot_prec/icon-d2-ruc-eps_germany_icosahedral_single-level_{date_hour}_001_2d_tot_prec.grib2.bz2"
        print(f"  🔍 Controllo server DWD per run RUC {run_hour_str}Z...")

        try:
            r = requests.head(url_test, timeout=10)
            if r.status_code == 200:
                print(f"  🟢 File trovato! Il run RUC {run_hour_str}Z è online.")
                nome_run = dt_run.strftime("%H") + "Z"

                if os.path.exists(FILE_LAST_HOUR):
                    with open(FILE_LAST_HOUR, "r") as f:
                        ultimo_salvato = f.read().strip()
                    if date_hour <= ultimo_salvato:
                        print(f"  ✅ Run ICON-D2-RUC EPS {nome_run} già elaborato in precedenza (Ultimo: {ultimo_salvato}).")
                        return False, "", None

                with open(FILE_LAST_HOUR, "w") as f:
                    f.write(date_hour)

                return True, nome_run, dt_run
            else:
                print(f"  ⚠️ Run RUC {run_hour_str}Z non ancora disponibile (HTTP {r.status_code})")
        except Exception as e:
            print(f"  ❌ Errore testando {run_hour_str}Z: {e}")

        # Scaliamo di 1 ora indietro (il RUC è orario)
        dt_run -= timedelta(hours=1)

    return False, "", None

def scarica_step_precipitazione(dt_run_utc, h_step, max_retries=3):
    """
    Scarica l'accumulo TOT_PREC sulla griglia nativa icosaedrale del RUC EPS.
    """
    run_hour_syn = dt_run_utc.hour          
    run_hour = f"{run_hour_syn:02d}"
    date_hour = dt_run_utc.strftime('%Y%m%d%H')
    step_str = f"{h_step:03d}"
    
    # URL Diretto per TOT_PREC del RUC
    url_tot = f"https://opendata.dwd.de/weather/nwp/icon-d2-ruc-eps/grib/{run_hour}/tot_prec/icon-d2-ruc-eps_germany_icosahedral_single-level_{date_hour}_{step_str}_2d_tot_prec.grib2.bz2"

    def _download_one(url: str):
        for tentativo in range(max_retries):
            try:
                r = requests.get(url, stream=True, timeout=30)
                r.raise_for_status()
                fd, temp_path = tempfile.mkstemp(suffix=".grib2")
                with os.fdopen(fd, 'wb') as f_out:
                    decompressor = bz2.BZ2Decompressor()
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk: f_out.write(decompressor.decompress(chunk))
                return temp_path
            except Exception as e:
                if tentativo == max_retries - 1: raise e
                time.sleep(5 * (tentativo + 1))

    p_tot = _download_one(url_tot)
    ds_tot = earthkit.data.from_source("file", p_tot).to_xarray()

    if 'member' in ds_tot.dims: ds_tot = ds_tot.rename({'member': 'eps'})
    elif 'number' in ds_tot.dims: ds_tot = ds_tot.rename({'number': 'eps'})

    # Estrazione dinamica della variabile
    tot_var = list(ds_tot.data_vars)[0]
    tot_prec = ds_tot[tot_var].compute()

    ds_tot.close()
    try: os.remove(p_tot) 
    except: pass

    return tot_prec

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
    for h in range(1, RUN_DURATION + 1): # Modificato per 27h
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
        percorsi_foto = []

        prev_step_idx = -1
        prev_tot = None

        for h in ore_list:
            try:
                print(f"  ⬇️  Elaborazione accumulo orario H={h}...")
                curr_tot = scarica_step_precipitazione(dt_run_utc, h)

                if h == 1:
                    prec_oraria = curr_tot
                else:
                    if prev_step_idx == h - 1 and prev_tot is not None:
                        prec_h_minus_1 = prev_tot
                    else:
                        prec_h_minus_1 = scarica_step_precipitazione(dt_run_utc, h - 1)
                    prec_oraria = curr_tot - prec_h_minus_1

                prev_tot = curr_tot
                prev_step_idx = h

                # --- Calcolo Probabilità e Disegno Diretto ---
                prob_xr = (prec_oraria >= 0.5).astype(float).mean(dim="eps") * 100
                
                lat_vals = prob_xr['latitude'].values
                lon_vals = prob_xr['longitude'].values
                prob_vals = prob_xr.values

                fig = plt.figure(figsize=(10, 8))
                ax = plt.axes(projection=ccrs.Mercator())
                ax.set_extent(domain, crs=ccrs.PlateCarree())

                ax.add_feature(regions_feature)
                if prov_feature: ax.add_feature(prov_feature)
                else: 
                    ax.coastlines(resolution='10m')
                    ax.add_feature(cfeature.BORDERS)

                cmap = ListedColormap(my_colors)
                norm = BoundaryNorm(my_levels, cmap.N)
                mask = prob_vals >= 10
                
                if np.any(mask):
                    sc = ax.scatter(lon_vals[mask], lat_vals[mask], 
                                    c=prob_vals[mask], cmap=cmap, norm=norm,
                                    s=4, marker='s', transform=ccrs.PlateCarree(),
                                    edgecolors='none')
                    
                    cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', shrink=0.7, pad=0.05)
                    cbar.set_label("Probabilità (%)", fontweight='bold')

                ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())
                for lo, la, sig in zip(lons, lats, sigle):
                    ax.plot(lo, la, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                    ax.text(lo + 0.05, la + 0.05, sig, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

                start_local = dt_run_local + timedelta(hours=h-1)
                end_local = dt_run_local + timedelta(hours=h)
                str_valida = f"{start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M del %d/%m')}"

                title = f"ICON-D2-RUC EPS - Probabilità Pioggia >= 0.5 mm/h (%)\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')} | {str_valida}"
                plt.title(title, fontweight='bold')

                filename = f"oraria_{h}.png"
                plt.savefig(filename, dpi=200, bbox_inches='tight')
                plt.close(fig)
                percorsi_foto.append(filename)

            except Exception as e:
                print(f"  ❌ Errore elaborando l'ora {h}: {e}")
                continue

        if percorsi_foto:
            caption_album = f"ICON-D2-RUC EPS: Probabilità Pioggia oraria >= 0.5 mm\n{block_name}\nRun {nome_run}"
            invia_album_telegram(percorsi_foto, caption_album)
            
            for f in percorsi_foto:
                if os.path.exists(f): os.remove(f)
                
        time.sleep(10)

def main():
    print("Cerco l'ultimo run completo ICON-D2-RUC EPS direttamente sul server DWD...")
    is_new, nome_run, dt_run_utc = get_latest_ruc_run()

    if is_new:
        print(f"🚀 Lancio generazione Probabilità Orarie ICON-RUC EPS per il RUN {nome_run} ({dt_run_utc.strftime('%Y-%m-%d %H:%M')})")
        genera_album_orari(dt_run_utc, nome_run)
    else:
        print("Nessun nuovo run trovato o run in fase di caricamento. Uscita.")

if __name__ == "__main__":
    main()