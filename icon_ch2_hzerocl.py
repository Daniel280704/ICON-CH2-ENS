import os, sys, time, json, requests, urllib3, pytz
import numpy as np, matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import warnings
import cartopy.crs as ccrs, cartopy.feature as cfeature, cartopy.io.shapereader as shpreader
import earthkit.data
import earthkit.plots; from earthkit.plots.geo import bounds, domains; from earthkit.plots.styles import Style; from earthkit.data import config
from meteodatalab import ogd_api; from meteodatalab.operators import regrid; from rasterio.crs import CRS

warnings.filterwarnings('ignore'); urllib3.disable_warnings(); config.set("cache-policy", "temporary")

LATITUDE, LONGITUDE, FILE_LAST_HOUR = 45.07, 7.54, "ultima_ora_icon_ch2_hzerocl.txt"

def scarica_variabile_con_retry(request, max_retries=6):
    """Tenta lo scaricamento fino a max_retries, svuotando la cache in caso di fallimento."""
    for tentativo in range(max_retries):
        try:
            if tentativo > 0:
                print(f"    🔄 Retry {tentativo + 1}/{max_retries} in corso...")

            return ogd_api.get_from_ogd(request)

        except Exception as e:
            if tentativo == max_retries - 1:
                print(f"    💥 Fallimento definitivo dopo {max_retries} tentativi.")
                raise e

            delay = 10 * (tentativo + 1)
            print(f"    ⚠️ Errore o blocco di rete: {e}")
            print(f"    🧹 Svuoto la cache corrotta e attendo {delay}s...")

            try: earthkit.data.cache.purge()
            except Exception: pass

            time.sleep(delay)

def estrai_limiti_run(hourly_data, ref_param, utc_offset_sec):
    times, mean_vals = hourly_data.get("time", []), hourly_data.get(ref_param, [])
    if not times or not mean_vals: return False, "", None
    end_idx = next((i for i in range(len(mean_vals)-1, -1, -1) if mean_vals[i] is not None), -1)
    if end_idx == -1: return False, "", None
    valida_str = times[end_idx]
    dt_end_utc = datetime.fromisoformat(valida_str) - timedelta(seconds=utc_offset_sec)
    dt_run_utc = dt_end_utc - timedelta(hours=120)
    if end_idx - times.index((dt_run_utc + timedelta(hours=1) + timedelta(seconds=utc_offset_sec)).strftime("%Y-%m-%dT%H:%M")) + 1 < 120: return False, "", None
    if os.path.exists(FILE_LAST_HOUR) and valida_str <= open(FILE_LAST_HOUR).read().strip(): return False, "", None
    open(FILE_LAST_HOUR, "w").write(valida_str)
    return True, dt_run_utc.strftime("%H") + "Z", dt_run_utc.replace(tzinfo=timezone.utc)

def invia_album(file_paths, caption):
    t, c, th = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), os.getenv("TELEGRAM_THREAD_ID_14")
    if not t or not c: return
    m, f = [{"type": "photo", "media": f"attach://p_{i}", "caption": caption if i==0 else ""} for i in range(len(file_paths))], {f"p_{i}": open(p, "rb") for i, p in enumerate(file_paths)}
    try: requests.post(f"https://api.telegram.org/bot{t}/sendMediaGroup", data={"chat_id": c, "media": json.dumps(m), "message_thread_id": th} if th else {"chat_id": c, "media": json.dumps(m)}, files=f)
    except: pass
    finally: [v.close() for v in f.values()]

def genera(dt_utc, n_run):
    dt_loc = dt_utc.astimezone(pytz.timezone("Europe/Rome"))
    dest = regrid.RegularGrid(CRS.from_string("epsg:4326"), 300, 300, 6.0, 10.5, 43.5, 46.8)
    dom = domains.Domain.from_bbox(bbox=bounds.BoundingBox(6.0, 10.5, 43.5, 46.8, ccrs.Geodetic()), name="Piemonte")
    f_reg = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    f_prov = cfeature.ShapelyFeature(shpreader.Reader("shapefiles/ProvCM01012026_WGS84.shp").geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':') if os.path.exists("shapefiles/ProvCM01012026_WGS84.shp") else None

    # Nuovi livelli e colori aggiornati con step di 200m
    my_levels = list(range(0, 6001, 200)) # Genera [0, 200, 400, ..., 6000] (31 livelli)
    my_colors = [ # 30 colori sfumati per coprire i 30 intervalli creati dai 31 livelli
        "#313695", "#374a9f", "#3d5ea9", "#4372b3", "#4c84be", 
        "#5a97c9", "#68aad4", "#7ebce0", "#91cce9", "#a2daf1", 
        "#b4e5f7", "#c6effb", "#d7f7fa", "#e8faf5", "#f5fad8", 
        "#fdedb3", "#fde08f", "#fdd26d", "#fdc34e", "#fcb137", 
        "#f8972b", "#f47b22", "#ef5c1c", "#e53b1b", "#d62025", 
        "#c20b26", "#a80025", "#8e0023", "#70001f", "#520018"
    ]

    intervals_by_day = {}
    last_h = 0

    for h in range(1, 121):
        dt_target = dt_loc + timedelta(hours=h)
        if dt_target.hour % 6 == 0 or h == 120:
            if last_h < h:
                dt_start_interval = dt_loc + timedelta(hours=last_h)
                day_str = dt_start_interval.strftime('%d/%m/%Y')

                if day_str not in intervals_by_day: intervals_by_day[day_str] = []
                intervals_by_day[day_str].append((last_h, h))
            last_h = h

    req = ogd_api.Request("ogd-forecasting-icon-ch2", "HZEROCL", dt_utc, True, [f"P{l//24}DT{l%24}H" for l in range(1, 121)])
    try:
        print(f"  ⬇️  Scarico dati HZEROCL per le 120 ore...")
        vm_data = scarica_variabile_con_retry(req)
        print(f"  ✅ Dati scaricati con successo.")
    except Exception as e:
        print(f"  ❌ Errore nel download: {e}")
        return

    for day_str, intervals in intervals_by_day.items():
        fp = []
        for h_start, h_end in intervals:
            hours_slice = [np.timedelta64(h, 'h') for h in range(h_start + 1, h_end + 1)]
            hzero_interval = vm_data.sel(lead_time=hours_slice).max(dim="lead_time").mean(dim="eps")

            ch = earthkit.plots.Map(domain=dom)
            ch.grid_cells(regrid.iconremap(hzero_interval, dest), x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))
            ch.ax.add_feature(f_reg); ch.ax.add_feature(f_prov) if f_prov else ch.borders()

            start_local = dt_loc + timedelta(hours=h_start)
            end_local = dt_loc + timedelta(hours=h_end)
            orario_str = f"{start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M')}"

            ch.title(f"ICON-CH2 EPS - Zero Termico (m MSL) MAX (Media dei Massimi)\n{day_str} | Fascia {orario_str}\nRun: {dt_utc.strftime('%d/%m %H:%M UTC')}")
            ch.legend(label="HZEROCL (m)")

            fname = f"h_max_{h_start}_{h_end}.png"
            ch.save(fname); fp.append(fname); plt.close(ch.fig)

        invia_album(fp, f"ICON-CH2 EPS: Zero Termico MAX\nFasce da 6 ore del {day_str}\nRun {n_run}")
        for f in fp:
            if os.path.exists(f): os.remove(f)

    del vm_data

if __name__ == "__main__":
    d = next((requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={"latitude":LATITUDE,"longitude":LONGITUDE,"hourly":"temperature_2m","models":"meteoswiss_icon_ch2_ensemble_mean","timezone":"Europe/Rome","past_days":1,"forecast_days":6}).json() for _ in range(3) if time.sleep(1) is None), {})
    if d: 
        is_n, nr, dt = estrai_limiti_run(d.get("hourly",{}), "temperature_2m", d.get("utc_offset_seconds",0))
        if is_n: genera(dt, nr)