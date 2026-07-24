import os
import sys
import time
import json
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
from shapely.ops import unary_union

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
FILE_LAST_HOUR = "ultima_ora_icon_ch2_cape.txt"

def estrai_limiti_run(hourly_data: dict, ref_param: str) -> tuple[bool, str, datetime]:
    times = hourly_data.get("time", [])
    mean_vals = hourly_data.get(ref_param, [])
    if not times or not mean_vals: return False, "", None
    
    end_idx = -1
    for i in range(len(mean_vals) - 1, -1, -1):
        if mean_vals[i] is not None:
            end_idx = i
            break
            
    if end_idx == -1: return False, "", None
    
    rome_tz = pytz.timezone("Europe/Rome")
    ultima_ora_valida_str = times[end_idx]
    
    dt_end_local = rome_tz.localize(datetime.fromisoformat(ultima_ora_valida_str))
    dt_end_utc = dt_end_local.astimezone(timezone.utc)
    
    # Troviamo l'innesco sapendo che ICON-CH2 dura 120 ore
    dt_run_utc = dt_end_utc - timedelta(hours=120)
    
    # Cerchiamo l'indice di partenza del forecast (+1 ora di delay)
    dt_start_local = (dt_run_utc + timedelta(hours=1)).astimezone(rome_tz)
    start_time_str = dt_start_local.strftime("%Y-%m-%dT%H:%M")
    
    try:
        start_idx = times.index(start_time_str)
    except ValueError:
        return False, "", None
        
    expected_points = 120
    actual_points = end_idx - start_idx + 1
    nome_run = dt_run_utc.strftime("%H") + "Z"
    
    if actual_points < expected_points:
        print(f"⏳ Run {nome_run} in caricamento... ({actual_points}/{expected_points} ore)")
        return False, "", None
        
    if os.path.exists(FILE_LAST_HOUR):
        with open(FILE_LAST_HOUR, "r") as f:
            ultima_ora_salvata = f.read().strip()
        if ultima_ora_valida_str <= ultima_ora_salvata:
            print(f"✅ Run ICON-CH2 CAPE {nome_run} già elaborato.")
            return False, "", None

    with open(FILE_LAST_HOUR, "w") as f:
        f.write(ultima_ora_valida_str)

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
    for attempt in range(3):
        try:
            r = requests.get(URL, params=params, timeout=30)
            r.raise_for_status()
            print(f"✅ Dati scaricati correttamente")
            return r.json()
        except Exception as e:
            print(f"⚠️ Tentativo {attempt + 1}/3 fallito: {e}")
            if attempt < 2:
                time.sleep(15)
    print("❌ Impossibile scaricare i dati dopo 3 tentativi")
    return {}

def invia_album_telegram(file_paths: list, caption: str):
    token = os.getenv("TELEGRAM_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("❌ TELEGRAM_TOKEN o TELEGRAM_CHAT_ID non trovati")
        return
    
    if not file_paths:
        print("⚠️ Nessuna foto da inviare")
        return
    
    if len(file_paths) == 1:
        url = f"https://api.telegram.org/bot{token}/sendPhoto"
        try:
            with open(file_paths[0], "rb") as photo:
                requests.post(url, data={"chat_id": chat_id, "caption": caption}, files={"photo": photo})
            print(f"📸 Foto singola inviata con successo")
        except Exception as e:
            print(f"❌ Errore invio singola foto: {e}")
        return

    url = f"https://api.telegram.org/bot{token}/sendMediaGroup"
    media = []
    files = {}
    
    for idx, path in enumerate(file_paths):
        if not os.path.exists(path):
            print(f"⚠️ File non trovato: {path}")
            continue
        media.append({
            "type": "photo",
            "media": f"attach://photo_{idx}",
            "caption": caption if idx == 0 else ""
        })
        files[f"photo_{idx}"] = open(path, "rb")

    if not files:
        print("❌ Nessun file valido da inviare")
        return

    try:
        r = requests.post(url, data={"chat_id": chat_id, "media": json.dumps(media)}, files=files)
        r.raise_for_status()
        print(f"📸 Album Telegram inviato con successo ({len(files)} mappe).")
    except Exception as e:
        print(f"❌ Errore invio album Telegram: {e}")
    finally:
        for f in files.values():
            f.close()

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
        if key not in blocchi:
            blocchi[key] = []
        blocchi[key].append(h)
        
    return blocchi

def genera_album_cape(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    
    blocchi = raggruppa_in_blocchi(dt_run_local)

    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    nx, ny = 300, 300
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

    # Scala colori CAPE dettagliata: 200-5000 J/kg
    # Da blu (instabilità debole) a rosso scuro (instabilità estrema)
    my_levels = [200, 300, 400, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    my_colors = ["#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", 
                 "#d73027", "#c7192b", "#b30000", "#8b0000", "#660000", "#440000", "#220000"]
    domain = domains.Domain.from_bbox(bbox=bounds.BoundingBox(xmin, xmax, ymin, ymax, ccrs.Geodetic()), name="Piemonte")

    # Gestione livelli confini Geografici
    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    prov_feature = None
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    # Dati per capoluoghi
    lats = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92]
    lons = [7.68,  7.55,  8.20,  8.61,  8.42,  8.61,  8.05,  8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    for block_name, ore_list in blocchi.items():
        print(f"\n📊 Generazione album CAPE: {block_name}")
        
        lead_times_str = [f"P{l // 24}DT{l % 24}H" for l in ore_list]

        req = ogd_api.Request(
            collection="ogd-forecasting-icon-ch2",
            variable="CAPE_MU",
            ref_time=dt_run_utc,
            perturbed=True,
            lead_time=lead_times_str,
        )
        
        try:
            print(f"  ⬇️  Scarico dati CAPE_MU per {len(ore_list)} ore...")
            cape_raw = ogd_api.get_from_ogd(req)
            cape_mean = cape_raw.mean(dim="eps")
            print(f"  ✅ Dati scaricati: {len(ore_list)} ore")
        except Exception as e:
            print(f"  ❌ Salto il blocco {block_name} causa errore: {e}")
            continue

        percorsi_foto = []
        
        for h in ore_list:
            cape_step = cape_mean.sel(lead_time=np.timedelta64(h, 'h'))
            cape_geo = regrid.iconremap(cape_step, destination)

            chart = earthkit.plots.Map(domain=domain)
            chart.grid_cells(cape_geo, x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))

            # Aggiunta Confini (Regione spessa, Provincia fine)
            chart.ax.add_feature(regions_feature)
            if prov_feature:
                chart.ax.add_feature(prov_feature)
            else:
                chart.borders()

            # Aggiunta Pallino Rivoli
            chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree(), zorder=12)

            # Aggiunta Capoluoghi con Sigle
            for lon, lat, sigla in zip(lons, lats, sigle):
                chart.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree(), zorder=12)
                chart.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree(), zorder=12)

            start_local = dt_run_local + timedelta(hours=h-1)
            end_local = dt_run_local + timedelta(hours=h)
            str_valida = f"{start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M del %d/%m')}"

            chart.title(f"ICON-CH2 EPS - CAPE MU (J/kg)\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')} | {str_valida}")
            chart.legend(label="CAPE MU (J/kg)")
            
            f_name = f"cape_{h}.png"
            chart.save(f_name)
            
            if os.path.exists(f_name):
                percorsi_foto.append(f_name)
                print(f"    ✅ Mappa ora {h}: {f_name}")
            else:
                print(f"    ❌ Errore salvataggio: {f_name}")
            
            plt.close(chart.fig)
        
        if percorsi_foto:
            caption_album = f"⚡ ICON-CH2 EPS: CAPE MU (Instabilità)\n🗓 {block_name}\n⚙️ Run {nome_run}"
            invia_album_telegram(percorsi_foto, caption_album)
            
            for f in percorsi_foto:
                if os.path.exists(f): 
                    os.remove(f)
                    print(f"    🗑️  Rimosso: {f}")
        else:
            print(f"  ⚠️  Nessuna mappa generata per {block_name}")
            
        del cape_raw, cape_mean
        print(f"  💤 Attendo 15 secondi prima del prossimo blocco...")
        time.sleep(15)

def main():
    print("🔄 Cerco l'ultimo run completo ICON-CH2 via Open-Meteo...")
    data = fetch_dati_con_retry()
    if not data: 
        print("❌ Nessun dato scaricato. Uscita.")
        sys.exit(1)
        
    hourly = data.get("hourly", {})
    print(f"📊 Analizzando dati orari ({len(hourly.get('time', []))} timestep)")
    is_new, nome_run, dt_run_utc = estrai_limiti_run(hourly, "temperature_2m")
    
    if is_new:
        print(f"🚀 Lancio generazione Album CAPE per il RUN {nome_run} ({dt_run_utc})")
        genera_album_cape(dt_run_utc, nome_run)
        print("✅ Elaborazione completata!")
    else:
        print("⏭️  Nessun nuovo run completo trovato. Uscita.")

if __name__ == "__main__":
    main()
