import os
import sys
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

LOCK_FILE = "lock_kenda_ch1_ieri.txt"

def estrai_finestra_temporale():
    rome_tz = pytz.timezone("Europe/Rome")
    now_local = datetime.now(rome_tz)
    
    # Determina l'orario di fine finestra: le 21:00 più recenti passate
    if now_local.hour >= 21:
        end_local = now_local.replace(hour=21, minute=0, second=0, microsecond=0)
    else:
        end_local = (now_local - timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)
        
    # L'inizio è esattamente 24 ore prima (le 21:00 del giorno prima)
    start_local = end_local - timedelta(hours=24)
    
    # Target string per il file di lock (es. 2026-07-26_2100)
    target_date_str = end_local.strftime("%Y-%m-%d_2100")
    
    # Controllo sistema di Lock
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            saved_date = f.read().strip()
        if saved_date == target_date_str:
            print(f"✅ Precipitazioni per la finestra {target_date_str} già elaborate e inviate. Esco.")
            return False, None, None, None, []
            
    # Per ottenere l'accumulo, servono i run orari in formato UTC
    start_utc = start_local.astimezone(timezone.utc)
    ref_times = [start_utc + timedelta(hours=i) for i in range(24)]
    
    return True, target_date_str, start_local, end_local, ref_times

def invia_telegram(file_path: str, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    thread_id = os.getenv("TELEGRAM_THREAD_ID_19080")
    
    if not token or not chat_id: return
    
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    payload = {"chat_id": chat_id, "caption": caption}
    
    if thread_id:
        payload["message_thread_id"] = thread_id
        
    try:
        with open(file_path, "rb") as photo:
            requests.post(url, data=payload, files={"photo": photo})
        print(f"📸 Mappa Telegram inviata con successo.")
    except Exception as e:
        print(f"Errore invio foto: {e}")

def main():
    is_new, target_date_str, start_local, end_local, ref_times = estrai_finestra_temporale()
    
    if not is_new:
        sys.exit(0) # Blocca lo script silenziosamente, la Action farà skip
        
    print(f"🚀 Inizio calcolo precipitazioni per il {target_date_str} (dalle 21:00 alle 21:00)...")
    
    prec_24h_tot = None
    for ref_time in ref_times:
        req = ogd_api.Request(
            collection="ogd-analysis-kenda-ch1",
            variable="TOT_PREC",
            ref_time=ref_time,
            perturbed=False,
            lead_time="P0DT1H",
        )
        try:
            print(f"  ⬇️  Scarico ore: {ref_time.strftime('%H:%M')} UTC...")
            da = ogd_api.get_from_ogd(req)
            if getattr(da, "size", 0) == 0:
                print(f"  ⚠️ Dati ancora mancanti per le {ref_time.strftime('%H:%M')} UTC. Interrompo e riproverò al prossimo cron.")
                sys.exit(0) # Esce senza scrivere il lock, ritenterà dopo un'ora

            if prec_24h_tot is None:
                prec_24h_tot = da.copy(deep=True)
            else:
                prec_24h_tot.values += da.values
        except Exception as e:
            print(f"  ❌ Errore download {ref_time.strftime('%H:%M')} UTC: {e}. Riproverò al prossimo cron.")
            sys.exit(0)

    # --- REGIONALIZZAZIONE E PLOT ---
    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    nx, ny = 300, 300
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

    # SCALA COLORI PERSONALIZZATA METEOLOGIX
    my_levels = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 130, 150, 250, 350, 500]
    my_colors = [
        "#00d4ff", "#0080ff", "#0000ff", "#ccffcc", "#99ff99", "#66ff66", "#33ff33", "#00ff00", 
        "#00e600", "#00cc00", "#009900", "#ffff00", "#ffcc00", "#ff9900", "#ff6600", "#ff3300", 
        "#ff0000", "#cc0000", "#990000", "#ffccff", "#ff99ff", "#cc66ff", "#9933ff", "#6600cc", 
        "#4d0099", "#800080", "#ffffff", "#cccccc", "#808080", "#333333"
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

    prec_geo = regrid.iconremap(prec_24h_tot, destination)

    chart = earthkit.plots.Map(domain=domain)
    chart.grid_cells(prec_geo, x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))

    chart.ax.add_feature(regions_feature)
    if prov_feature:
        chart.ax.add_feature(prov_feature)
    else:
        chart.borders()

    chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())

    for lon, lat, sigla in zip(lons, lats, sigle):
        chart.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
        chart.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())
    
    str_valida = f"Accumulo 24h: dalle {start_local.strftime('%H:%M')} del {start_local.strftime('%d/%m')} alle {end_local.strftime('%H:%M')} del {end_local.strftime('%d/%m')}"
    title = f"KENDA-CH1 Analisi - Precipitazioni (mm)\n{str_valida}"
    
    chart.title(title)
    chart.legend(label="Accumulo (mm)")

    filename = "kenda_ch1_accumulo_24h_esatto.png"
    chart.save(filename)
    plt.close(chart.fig)
    
    caption_album = f"KENDA-CH1 Analisi: Precipitazioni 24h\n{str_valida}"
    invia_telegram(filename, caption_album)
    
    if os.path.exists(filename): os.remove(filename)

    # Scrive il Lock File SOLO se tutto è andato a buon fine
    with open(LOCK_FILE, "w") as f:
        f.write(target_date_str)
    print("✅ Mappa processata correttamente e Lock file aggiornato.")

if __name__ == "__main__":
    main()
