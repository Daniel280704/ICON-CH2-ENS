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
ARCHIVE_DIR = "kenda_archive"

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
    
    caption_album = f"KENDA-CH1 Analisi\n{titolo}"
    invia_telegram(filename, caption_album)
    if os.path.exists(filename): os.remove(filename)

def main():
    rome_tz = pytz.timezone("Europe/Rome")
    today_str = datetime.now(rome_tz).strftime("%Y-%m-%d")
    
    # 1. CONTROLLO LOCK (Gira solo una volta al giorno)
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE, "r") as f:
            if f.read().strip() == today_str:
                print("✅ Mappe già inviate per oggi. Esco.")
                sys.exit(0)

    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    
    # 2. DEFINISCI L'ULTIMA ORA DISPONIBILE (ora UTC - 3 ore di latenza server)
    now_utc = datetime.now(timezone.utc)
    latest_ref_time = now_utc.replace(minute=0, second=0, microsecond=0) - timedelta(hours=3)
    
    # 3. LEGGI ARCHIVIO LOCALE
    existing_files = glob.glob(os.path.join(ARCHIVE_DIR, "kenda_prec_*.npy"))
    existing_times = []
    for f in existing_files:
        basename = os.path.basename(f)
        time_str = basename.replace("kenda_prec_", "").replace(".npy", "")
        dt = datetime.strptime(time_str, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc)
        existing_times.append(dt)
        
    # 4. CALCOLA ORE MANCANTI
    times_to_download = []
    if not existing_times:
        # Primo avvio in assoluto: prendi ultime 24 ore
        times_to_download = [latest_ref_time - timedelta(hours=i) for i in range(24)]
    else:
        last_saved_time = max(existing_times)
        curr = last_saved_time + timedelta(hours=1)
        while curr <= latest_ref_time:
            times_to_download.append(curr)
            curr += timedelta(hours=1)
            
        if len(times_to_download) > 24:
            print("⚠️ Gap maggiore di 24 ore! Recupero solo le ultime 24 disponibili su MeteoSvizzera.")
            times_to_download = [latest_ref_time - timedelta(hours=i) for i in range(24)]
            
    times_to_download.sort()

    # 5. ESTRAI TEMPLATE XARRAY PER IL REGRID (Serve una base dati intatta)
    template_da = None
    try:
        req = ogd_api.Request(collection="ogd-analysis-kenda-ch1", variable="TOT_PREC", ref_time=latest_ref_time, perturbed=False, lead_time="P0DT1H")
        template_da = ogd_api.get_from_ogd(req)
    except Exception as e:
        print(f"❌ Impossibile ottenere il template xarray: {e}. Riprovo al prossimo cron.")
        sys.exit(0)

    # 6. DOWNLOAD E SALVATAGGIO ORE SINGOLE
    for ref_time in times_to_download:
        req = ogd_api.Request(
            collection="ogd-analysis-kenda-ch1", variable="TOT_PREC", ref_time=ref_time, perturbed=False, lead_time="P0DT1H"
        )
        try:
            print(f"  ⬇️  Scarico ora mancante: {ref_time.strftime('%Y-%m-%d %H:%M')} UTC...")
            da = ogd_api.get_from_ogd(req)
            if getattr(da, "size", 0) == 0: continue
            
            # Salva la singola ora nell'archivio come matrice numpy pura
            file_name = f"kenda_prec_{ref_time.strftime('%Y%m%d_%H%M')}.npy"
            np.save(os.path.join(ARCHIVE_DIR, file_name), da.values)
            if ref_time not in existing_times:
                existing_times.append(ref_time)
        except Exception as e:
            print(f"  ❌ Errore download {ref_time.strftime('%H:%M')} UTC: {e}")

    if not existing_times:
        sys.exit(0)

    # 7. FUNZIONE DINAMICA DI ACCUMULO (Prende le ultime X ore disponibili)
    sorted_times = sorted(existing_times, reverse=True)
    
    def crea_accumulo(ore_richieste):
        target_times = sorted_times[:ore_richieste]
        if len(target_times) == 0: return None, None, None, 0
        
        # Azzera il template e inizia a sommare i file
        accumulo_da = template_da.copy(deep=True)
        accumulo_da.values = np.zeros_like(accumulo_da.values)
        
        ore_effettive = 0
        for t in target_times:
            file_path = os.path.join(ARCHIVE_DIR, f"kenda_prec_{t.strftime('%Y%m%d_%H%M')}.npy")
            if os.path.exists(file_path):
                accumulo_da.values += np.load(file_path)
                ore_effettive += 1
                
        # Calcola la stringa temporale effettiva
        start_t_local = min(target_times).astimezone(rome_tz)
        end_t_local = (max(target_times) + timedelta(hours=1)).astimezone(rome_tz)
        return accumulo_da, start_t_local, end_t_local, ore_effettive

    # SCALE COLORI
    levels_24h = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 130, 150, 250, 350, 400]
    colors_24h = [
        "#00d4ff", "#0080ff", "#0000ff", "#ccffcc", "#99ff99", "#66ff66", "#33ff33", "#00ff00", 
        "#00e600", "#00cc00", "#009900", "#ffff00", "#ffcc00", "#ff9900", "#ff6600", "#ff3300", 
        "#ff0000", "#cc0000", "#990000", "#ffccff", "#ff99ff", "#cc66ff", "#9933ff", "#6600cc", 
        "#4d0099", "#800080", "#ffffff", "#cccccc", "#808080", "#333333"
    ]
    levels_48h = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 130, 150, 250, 350, 400, 500]
    colors_48h = colors_24h + ["#1a1a1a"]
    levels_72h = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 130, 150, 250, 350, 400, 500, 700]
    colors_72h = colors_48h + ["#000000"]

    # --- PLOT MAPPE ---
    da_24, start_24, end_24, ore_24 = crea_accumulo(24)
    if da_24 is not None:
        titolo = f"Ultime 24 Ore esatte disponibili\nDalle {start_24.strftime('%H:%M del %d/%m')} alle {end_24.strftime('%H:%M del %d/%m')}\n(Ore processate: {ore_24}/24)"
        genera_e_invia_mappa(da_24, titolo, "kenda_24h.png", levels_24h, colors_24h)

    da_48, start_48, end_48, ore_48 = crea_accumulo(48)
    if da_48 is not None and ore_48 > 24:
        titolo = f"Ultime 48 Ore esatte disponibili\nDalle {start_48.strftime('%H:%M del %d/%m')} alle {end_48.strftime('%H:%M del %d/%m')}\n(Ore processate: {ore_48}/48)"
        genera_e_invia_mappa(da_48, titolo, "kenda_48h.png", levels_48h, colors_48h)

    da_72, start_72, end_72, ore_72 = crea_accumulo(72)
    if da_72 is not None and ore_72 > 48:
        titolo = f"Ultime 72 Ore esatte disponibili\nDalle {start_72.strftime('%H:%M del %d/%m')} alle {end_72.strftime('%H:%M del %d/%m')}\n(Ore processate: {ore_72}/72)"
        genera_e_invia_mappa(da_72, titolo, "kenda_72h.png", levels_72h, colors_72h)

    # 8. SCRIVI LOCK
    with open(LOCK_FILE, "w") as f:
        f.write(today_str)
        
    # 9. PULIZIA REPOSITORY (tiene solo le ultime 80 ore per non superare limiti Github)
    all_files = glob.glob(os.path.join(ARCHIVE_DIR, "kenda_prec_*.npy"))
    if len(all_files) > 80:
        all_files.sort() # I più vecchi all'inizio
        for f in all_files[:-80]:
            os.remove(f)
            print(f"🧹 Pulizia vecchio file: {f}")

    print("✅ Elaborazione completa conclusa.")

if __name__ == "__main__":
    main()