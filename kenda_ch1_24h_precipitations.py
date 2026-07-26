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

FILE_LAST_HOUR = "ultima_ora_kenda_ch1_stac_24h.txt"
HOURS_TO_RETRIEVE = 24

def estrai_limiti() -> tuple[bool, datetime, list]:
    now_utc = datetime.now(timezone.utc)
    
    # Aumentato il margine a 3 ore per garantire la completa pubblicazione del dataset
    latest_ref_time = now_utc.replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    
    ultima_ora_valida_str = latest_ref_time.strftime("%Y-%m-%dT%H:%M")

    if os.path.exists(FILE_LAST_HOUR):
        with open(FILE_LAST_HOUR, "r") as f:
            ultima_ora_salvata = f.read().strip()
        if ultima_ora_valida_str <= ultima_ora_salvata:
            print(f"✅ Analisi 24h fino a {ultima_ora_valida_str} già elaborata.")
            return False, None, []

    with open(FILE_LAST_HOUR, "w") as f:
        f.write(ultima_ora_valida_str)

    # Scarichiamo le 24 ore distinte (ref_time) con lead_time=1h per comporre l'accumulo
    ref_times_to_fetch = [latest_ref_time - timedelta(hours=i) for i in range(HOURS_TO_RETRIEVE)]
    ref_times_to_fetch.reverse() # Dal più vecchio al più recente

    return True, latest_ref_time, ref_times_to_fetch

def invia_telegram(file_path: str, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    thread_id = os.getenv("TELEGRAM_THREAD_ID_3")
    
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

def genera_mappa_24h(latest_ref_time: datetime, ref_times: list):
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
            print(f"  ⬇️  Scarico dati TOT_PREC per ref_time: {ref_time.strftime('%Y-%m-%d %H:%M')}...")
            da = ogd_api.get_from_ogd(req)
            
            # Salvaguardia contro array vuoti o non inizializzati correttamente
            if getattr(da, "size", 0) == 0:
                print(f"  ⚠️ Dati mancanti o vuoti per {ref_time.strftime('%Y-%m-%d %H:%M')}. Salto l'ora.")
                continue

            if prec_24h_tot is None:
                prec_24h_tot = da.copy()
            else:
                prec_24h_tot = prec_24h_tot + da
                
        except Exception as e:
            print(f"  ❌ Errore nel download per {ref_time.strftime('%Y-%m-%d %H:%M')}: {e}")
            continue
            
    if prec_24h_tot is None:
        print("Impossibile generare la mappa, nessun dato orario scaricato con successo.")
        return

    # Limiti geografici centrati su Piemonte 
    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    nx, ny = 300, 300
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

    my_levels = [0.5, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]
    my_colors = ["#e6f2ff", "#99ccff", "#3399ff", "#004cff", "#66e666", "#33cc33", "#009900", "#99cc00", "#ffe600", "#e6b300", "#ff9900", "#ff6600", "#ff3300", "#ff3333", "#b30000", "#cc33ff", "#8000cc", "#4d0080"]
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

    # Marker per Rivoli
    chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())

    for lon, lat, sigla in zip(lons, lats, sigle):
        chart.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
        chart.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

    rome_tz = pytz.timezone("Europe/Rome")
    end_local = (latest_ref_time + timedelta(hours=1)).astimezone(rome_tz)
    start_local = end_local - timedelta(hours=24)
    
    str_valida = f"Dalle {start_local.strftime('%H:%M')} del {start_local.strftime('%d/%m')} alle {end_local.strftime('%H:%M')} del {end_local.strftime('%d/%m')}"
    title = f"KENDA-CH1 Analisi - Precipitazione Accumulata 24h (mm)\nAggiornamento: {latest_ref_time.strftime('%d/%m/%Y %H:%M UTC')} | {str_valida}"
    
    chart.title(title)
    chart.legend(label="Accumulo (mm)")

    filename = "kenda_ch1_accumulo_24h.png"
    chart.save(filename)
    plt.close(chart.fig)
    
    caption_album = f"KENDA-CH1 Analisi: Precipitazioni 24h\n{str_valida}"
    invia_telegram(filename, caption_album)
    
    if os.path.exists(filename): os.remove(filename)

def main():
    print("Inizio calcolo precipitazioni 24h da KENDA-CH1...")
    is_new, latest_ref_time, ref_times = estrai_limiti()
    
    if is_new:
        print(f"🚀 Lancio generazione mappa 24h fino all'aggiornamento delle {latest_ref_time.strftime('%H:%M')} UTC")
        genera_mappa_24h(latest_ref_time, ref_times)
    else:
        print("Nessuna nuova ora completata per KENDA-CH1. Uscita.")

if __name__ == "__main__":
    main()
