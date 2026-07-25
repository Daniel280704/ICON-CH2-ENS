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
from meteodatalab.operators.vertical_interpolation import interpolate_k2p
from rasterio.crs import CRS

warnings.filterwarnings('ignore')
urllib3.disable_warnings()
config.set("cache-policy", "temporary")

LATITUDE = 45.07
LONGITUDE = 7.54
FILE_LAST_HOUR = "ultima_ora_icon_ch2_w_700.txt"
RUN_DURATION = 120
START_DELAY = 1

def invia_album_telegram(file_paths: list, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    thread_id = os.getenv("TELEGRAM_THREAD_ID_31")
    
    if not token or not chat_id: return
    
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
        print(f"📸 Album W 700hPa inviato con successo.")
    except Exception as e: print(f"Errore invio: {e}")
    finally:
        for f in files.values(): f.close()

def genera_mappe_w_700(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    
    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    nx, ny = 300, 300
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

    # Scala colori per updraft (m/s)
    my_levels = [0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0]
    my_colors = ["#e6f2ff", "#99ccff", "#3399ff", "#004cff", "#66e666", "#33cc33", "#009900", "#ff0000"]
    domain = domains.Domain.from_bbox(bbox=bounds.BoundingBox(xmin, xmax, ymin, ymax, ccrs.Geodetic()), name="Piemonte")

    # Ciclo su blocchi di 3 ore
    for start_h in range(1, 120, 3):
        end_h = min(start_h + 2, 120)
        lead_times = [f"P{h//24}DT{h%24}H" for h in range(start_h, end_h + 1)]
        
        print(f"Elaborazione blocco: +{start_h}h a +{end_h}h")
        
        # Request W e P
        req_w = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="W", ref_time=dt_run_utc, perturbed=True, lead_time=lead_times)
        req_p = ogd_api.Request(collection="ogd-forecasting-icon-ch2", variable="P", ref_time=dt_run_utc, perturbed=False, lead_time=lead_times)
        
        try:
            w_data = ogd_api.get_from_ogd(req_w)
            p_data = ogd_api.get_from_ogd(req_p)
            
            # Interpolazione a 700 hPa
            w_700 = interpolate_k2p(field=w_data, p_field=p_data, p_tc_values=[700], p_tc_units='hPa', mode='linear_in_lnp')
            
            # Calcolo max (nel tempo) e media (sull'ensemble)
            w_max = w_700.max(dim="lead_time")
            w_final = w_max.mean(dim="eps")
            
        except Exception as e:
            print(f"Errore elaborazione blocco {start_h}-{end_h}: {e}")
            continue

        w_geo = regrid.iconremap(w_final.squeeze(), destination)
        
        # Plotting
        chart = earthkit.plots.Map(domain=domain)
        chart.grid_cells(w_geo, x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))
        chart.ax.add_feature(cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=0.5))
        chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=4, transform=ccrs.PlateCarree())
        
        title = f"ICON-CH2 EPS - Updraft (W) a 700 hPa\nMax nel blocco +{start_h}h/+{end_h}h | Run: {dt_run_utc.strftime('%d/%m %H:%MZ')}"
        chart.title(title)
        chart.legend(label="Velocità Verticale (m/s)")

        filename = f"w700_{start_h}.png"
        chart.save(filename)
        
        invia_album_telegram([filename], f"Updraft 700hPa (Media Massimi 3h)\nBlocco +{start_h}h/+{end_h}h")
        
        if os.path.exists(filename): os.remove(filename)
        del w_data, p_data, w_700, w_final, w_geo
        gc.collect()
        time.sleep(10)

def main():
    # Logica fetch simile ai precedenti...
    pass # Inserire qui la logica di estrai_limiti_run e fetch_dati_con_retry già usata negli script precedenti

if __name__ == "__main__":
    main()