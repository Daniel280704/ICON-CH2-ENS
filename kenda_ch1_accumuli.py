import os
import sys
import glob
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

LOCK_FILE = "lock_kenda_ch1_accumuli.txt"

def estrai_date_riferimento():
    rome_tz = pytz.timezone("Europe/Rome")
    now_local = datetime.now(rome_tz)
    
    # Mezzanotte di oggi
    today_mid = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Giorno 1 (Ieri), Giorno 2 (L'altro ieri), Giorno 3 (3 giorni fa)
    day1_mid = today_mid - timedelta(days=1)
    day2_mid = today_mid - timedelta(days=2)
    day3_mid = today_mid - timedelta(days=3)
    
    day1_str = day1_mid.strftime("%Y-%m-%d")
    day2_str = day2_mid.strftime("%Y-%m-%d")
    day3_str = day3_mid.strftime("%Y-%m-%d")
    
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            if f.read().strip() == day1_str:
                print(f"✅ Accumuli per il {day1_str} già elaborati. Esco.")
                return False, None, None, None, None

    # Vettore temporale per scaricare le 24 ore di "Ieri" (Day 1)
    start_utc = day1_mid.astimezone(timezone.utc)
    ref_times_day1 = [start_utc + timedelta(hours=i) for i in range(24)]
    
    return True, day1_str, day2_str, day3_str, ref_times_day1

def invia_telegram(file_path: str, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    thread_id = os.getenv("TELEGRAM_THREAD_ID_19080")
    
    if not token or not chat_id: return
    
    url = f"https://api.telegram.org/bot{token}/sendPhoto"
    payload = {"chat_id": chat_id, "caption": caption}
    if thread_id: payload["message_thread_id"] = thread_id
        
    try:
        with open(file_path, "rb") as photo:
            requests.post(url, data=payload, files={"photo": photo})
        print(f"📸 {file_path} inviato con successo.")
    except Exception as e:
        print(f"Errore invio foto: {e}")

def genera_e_invia_mappa(da_prec, titolo, filename, levels, colors):
    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    nx, ny = 300, 300
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

    domain = domains.Domain.from_bbox(bbox=bounds.BoundingBox(xmin, xmax, ymin, ymax, ccrs.Geodetic()), name="Piemonte")

    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    prov_feature = None
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    lats = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92]
    lons = [7.68,  7.55,  8.20,  8.61,  8.42,  8.61,  8.05,  8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    prec_geo = regrid.iconremap(da_prec, destination)

    chart = earthkit.plots.Map(domain=domain)
    chart.grid_cells(prec_geo, x="lon", y="lat", style=Style(colors=colors, levels=levels))

    chart.ax.add_feature(regions_feature)
    if prov_feature: chart.ax.add_feature(prov_feature)
    else: chart.borders()

    chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())

    for lon, lat, sigla in zip(lons, lats, sigle):
        chart.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
        chart.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())
    
    chart.title(f"KENDA-CH1 Analisi - Precipitazioni (mm)\n{titolo}")
    chart.legend(label="Accumulo (mm)")

    chart.save(filename)
    plt.close(chart.fig)
    
    caption_album = f"KENDA-CH1 Analisi: Precipitazioni\n{titolo}"
    invia_telegram(filename, caption_album)
    if os.path.exists(filename): os.remove(filename)

def main():
    is_new, day1_str, day2_str, day3_str, ref_times_day1 = estrai_date_riferimento()
    if not is_new:
        sys.exit(0)
        
    print(f"🚀 Download 24h per la giornata di ieri ({day1_str})...")
    
    # 1. SCARICA IERI E SALVA MATRICE
    prec_day1_da = None
    for ref_time in ref_times_day1:
        req = ogd_api.Request(
            collection="ogd-analysis-kenda-ch1",
            variable="TOT_PREC",
            ref_time=ref_time,
            perturbed=False,
            lead_time="P0DT1H",
        )
        try:
            da = ogd_api.get_from_ogd(req)
            if getattr(da, "size", 0) == 0:
                print(f"  ⚠️ Dato mancante per le {ref_time.strftime('%H:%M')} UTC. Riproverò al prossimo cron.")
                sys.exit(0)

            if prec_day1_da is None:
                prec_day1_da = da.copy(deep=True)
            else:
                prec_day1_da.values += da.values
        except Exception as e:
            print(f"  ❌ Errore download {ref_time.strftime('%H:%M')} UTC: {e}. Riproverò al prossimo cron.")
            sys.exit(0)

    # Salviamo la matrice di ieri usando numpy
    np.save(f"kenda_matrix_{day1_str}.npy", prec_day1_da.values)

    # 2. IMPOSTAZIONE SCALE COLORI
    levels_24h = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 130, 150, 250, 350, 500]
    colors_24h = [
        "#00d4ff", "#0080ff", "#0000ff", "#ccffcc", "#99ff99", "#66ff66", "#33ff33", "#00ff00", 
        "#00e600", "#00cc00", "#009900", "#ffff00", "#ffcc00", "#ff9900", "#ff6600", "#ff3300", 
        "#ff0000", "#cc0000", "#990000", "#ffccff", "#ff99ff", "#cc66ff", "#9933ff", "#6600cc", 
        "#4d0099", "#800080", "#ffffff", "#cccccc", "#808080", "#333333"
    ]
    
    levels_48h = levels_24h[:-1] + [350, 400, 500]
    colors_48h = colors_24h + ["#1a1a1a"]
    
    levels_72h = levels_24h[:-1] + [350, 400, 500, 600, 700]
    colors_72h = colors_24h + ["#1a1a1a", "#000000", "#3b170b"]

    # 3. GENERA MAPPA 24H
    titolo_24h = f"Accumulo 24h: {day1_str}"
    genera_e_invia_mappa(prec_day1_da, titolo_24h, "kenda_24h.png", levels_24h, colors_24h)

    # 4. CALCOLA E GENERA 48H (se esiste l'altro ieri)
    file_day2 = f"kenda_matrix_{day2_str}.npy"
    if os.path.exists(file_day2):
        print(f"🔄 Trovato storico 48h ({day2_str}), genero mappa...")
        matrice_day2 = np.load(file_day2)
        prec_48h_da = prec_day1_da.copy(deep=True)
        prec_48h_da.values += matrice_day2
        
        titolo_48h = f"Accumulo 48h: dal {day2_str} al {day1_str}"
        genera_e_invia_mappa(prec_48h_da, titolo_48h, "kenda_48h.png", levels_48h, colors_48h)

        # 5. CALCOLA E GENERA 72H (se esiste 3 giorni fa)
        file_day3 = f"kenda_matrix_{day3_str}.npy"
        if os.path.exists(file_day3):
            print(f"🔄 Trovato storico 72h ({day3_str}), genero mappa...")
            matrice_day3 = np.load(file_day3)
            prec_72h_da = prec_48h_da.copy(deep=True)
            prec_72h_da.values += matrice_day3
            
            titolo_72h = f"Accumulo 72h: dal {day3_str} al {day1_str}"
            genera_e_invia_mappa(prec_72h_da, titolo_72h, "kenda_72h.png", levels_72h, colors_72h)

    # Scrive il Lock File
    with open(LOCK_FILE, "w") as f:
        f.write(day1_str)
        
    # Pulizia vecchie matrici (tiene solo le ultime 3 per sicurezza)
    for f in glob.glob("kenda_matrix_*.npy"):
        if f not in [f"kenda_matrix_{day1_str}.npy", file_day2, file_day3]:
            os.remove(f)
            print(f"🧹 Pulizia vecchio file storico: {f}")

    print("✅ Elaborazione completa conclusa.")

if __name__ == "__main__":
    main()