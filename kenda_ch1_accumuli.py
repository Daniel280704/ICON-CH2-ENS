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

LOCK_FILE = "lock_kenda_ch1_daily.txt"
ARCHIVE_DIR = "kenda_archive"

def aggiorna_archivio_silente():
    """Scarica le ore mancanti nelle ultime 24h e le salva come matrici numpy"""
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    # Latenza server di 3 ore
    latest_ref_time = now_utc.replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    
    print(f"🔄 [FASE 1] Aggiornamento silente archivio orario (fino a {latest_ref_time.strftime('%H:%M')} UTC)...")
    
    # Controlliamo la disponibilità delle ultime 24h per recuperare eventuali buchi
    for i in range(24):
        ref_time = latest_ref_time - timedelta(hours=i)
        file_name = f"kenda_prec_{ref_time.strftime('%Y%m%d_%H%M')}.npy"
        file_path = os.path.join(ARCHIVE_DIR, file_name)
        
        if not os.path.exists(file_path):
            req = ogd_api.Request(
                collection="ogd-analysis-kenda-ch1",
                variable="TOT_PREC",
                ref_time=ref_time,
                perturbed=False,
                lead_time="P0DT1H",
            )
            try:
                da = ogd_api.get_from_ogd(req)
                if getattr(da, "size", 0) > 0:
                    np.save(file_path, da.values)
                    print(f"  📥 Trovato e salvato: {file_name}")
            except Exception:
                pass # Fallisce silenziosamente, riproverà l'ora successiva

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

def verifica_e_invia_mappe():
    print("🔎 [FASE 2] Verifica completezza dati giornalieri...")
    rome_tz = pytz.timezone("Europe/Rome")
    now_local = datetime.now(rome_tz)
    
    # Date di riferimento a Mezzanotte locale
    today_mid_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    day1_mid_local = today_mid_local - timedelta(days=1)
    day2_mid_local = today_mid_local - timedelta(days=2)
    day3_mid_local = today_mid_local - timedelta(days=3)
    
    lock_str = day1_mid_local.strftime("%Y%m%d")
    target_date_str = day1_mid_local.strftime("%d/%m/%Y")
    
    # Controllo Lock: Abbiamo già inviato l'album per "ieri"?
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            if f.read().strip() == lock_str:
                print(f"✅ Mappe per il {target_date_str} già inviate. Esco in silenzio.")
                return

    # Controlliamo se abbiamo tutti i 24 file esatti di Ieri
    ref_times_day1 = [day1_mid_local.astimezone(timezone.utc) + timedelta(hours=i) for i in range(24)]
    
    for rt in ref_times_day1:
        file_path = os.path.join(ARCHIVE_DIR, f"kenda_prec_{rt.strftime('%Y%m%d_%H%M')}.npy")
        if not os.path.exists(file_path):
            print(f"⏳ Manca ancora l'ora {rt.strftime('%H:%M')} UTC. Nessuna mappa inviata, attendo i prossimi run orari.")
            return

    print(f"🎉 Trovate tutte le 24 ore esatte del {target_date_str}! Procedo con le mappe...")

    # Recuperiamo un template xarray grezzo da MeteoSvizzera (serve per la riproiezione geometrica)
    template_da = None
    for rt in ref_times_day1:
        try:
            req = ogd_api.Request(collection="ogd-analysis-kenda-ch1", variable="TOT_PREC", ref_time=rt, perturbed=False, lead_time="P0DT1H")
            template_da = ogd_api.get_from_ogd(req)
            if getattr(template_da, "size", 0) > 0:
                break
        except:
            pass
            
    if template_da is None:
        print("❌ Impossibile recuperare un template xarray di base. Riprovo al prossimo cron.")
        return

    # Funzione interna per sommare giorni interi
    def crea_accumulo_multiplo(start_local_mid, days):
        accumulo_da = template_da.copy(deep=True)
        accumulo_da.values = np.zeros_like(accumulo_da.values)
        
        ref_times = [start_local_mid.astimezone(timezone.utc) + timedelta(hours=i) for i in range(24 * days)]
        ore_trovate = 0
        
        for rt in ref_times:
            path = os.path.join(ARCHIVE_DIR, f"kenda_prec_{rt.strftime('%Y%m%d_%H%M')}.npy")
            if os.path.exists(path):
                accumulo_da.values += np.load(path)
                ore_trovate += 1
                
        return accumulo_da, ore_trovate

    # COLORI METEOLOGIX
    levels_24h = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 130, 150, 250, 350, 400]
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

    # 1. MAPPA 24H (Ieri)
    da_24, ore_24 = crea_accumulo_multiplo(day1_mid_local, 1)
    if ore_24 == 24:
        titolo_24h = f"Accumulo esatto 24h: {target_date_str} (da mezzanotte a mezzanotte)"
        genera_e_invia_mappa(da_24, titolo_24h, "kenda_24h.png", levels_24h, colors_24h)

    # 2. MAPPA 48H (Se i dati dell'altro ieri sono in archivio)
    da_48, ore_48 = crea_accumulo_multiplo(day2_mid_local, 2)
    if ore_48 == 48:
        str_48 = f"dal {day2_mid_local.strftime('%d/%m/%Y')} al {target_date_str} inclusi"
        titolo_48h = f"Accumulo esatto 48h: {str_48}"
        genera_e_invia_mappa(da_48, titolo_48h, "kenda_48h.png", levels_48h, colors_48h)

    # 3. MAPPA 72H (Se i dati di 3 giorni fa sono in archivio)
    da_72, ore_72 = crea_accumulo_multiplo(day3_mid_local, 3)
    if ore_72 == 72:
        str_72 = f"dal {day3_mid_local.strftime('%d/%m/%Y')} al {target_date_str} inclusi"
        titolo_72h = f"Accumulo esatto 72h: {str_72}"
        genera_e_invia_mappa(da_72, titolo_72h, "kenda_72h.png", levels_72h, colors_72h)

    # Scrive il Lock File
    with open(LOCK_FILE, "w") as f:
        f.write(lock_str)

def pulisci_archivio():
    """Mantiene solo gli ultimi 100 file (circa 4 giorni) per non pesare su GitHub"""
    all_files = glob.glob(os.path.join(ARCHIVE_DIR, "kenda_prec_*.npy"))
    if len(all_files) > 100:
        all_files.sort()
        for f in all_files[:-100]:
            os.remove(f)
            print(f"🧹 Pulizia vecchio file: {f}")

def main():
    aggiorna_archivio_silente()
    verifica_e_invia_mappe()
    pulisci_archivio()

if __name__ == "__main__":
    main()