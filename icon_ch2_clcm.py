import os, sys, time, json, requests, urllib3, pytz
import numpy as np, matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import warnings
import cartopy.crs as ccrs, cartopy.feature as cfeature, cartopy.io.shapereader as shpreader
import earthkit.plots; from earthkit.plots.geo import bounds, domains; from earthkit.plots.styles import Style; from earthkit.data import config
from meteodatalab import ogd_api; from meteodatalab.operators import regrid; from rasterio.crs import CRS

warnings.filterwarnings('ignore'); urllib3.disable_warnings(); config.set("cache-policy", "temporary")

LATITUDE, LONGITUDE, FILE_LAST_HOUR = 45.07, 7.54, "ultima_ora_icon_ch2_clcm.txt"

def scarica_variabile_con_retry(request, max_retries=3, delay=5):
    """Tenta lo scaricamento fino a max_retries prima di sollevare eccezione."""
    for tentativo in range(max_retries):
        try:
            return ogd_api.get_from_ogd(request)
        except Exception as e:
            if tentativo == max_retries - 1:
                raise e
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
    t, c, th = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), os.getenv("TELEGRAM_THREAD_ID_17")
    if not t or not c: return
    m, f = [{"type": "photo", "media": f"attach://p_{i}", "caption": caption if i==0 else ""} for i in range(len(file_paths))], {f"p_{i}": open(p, "rb") for i, p in enumerate(file_paths)}
    try: requests.post(f"https://api.telegram.org/bot{t}/sendMediaGroup", data={"chat_id": c, "media": json.dumps(m), "message_thread_id": th} if th else {"chat_id": c, "media": json.dumps(m)}, files=f)
    except: pass
    finally: [v.close() for v in f.values()]

def raggruppa(dt_run):
    b = {}
    for h in range(1, 121):
        dt = dt_run + timedelta(hours=h); hr = dt.hour
        nm = "18-24" if hr == 0 else "00-06" if 1<=hr<=6 else "06-12" if 7<=hr<=12 else "12-18" if 13<=hr<=18 else "18-24"
        ds = (dt.date() - timedelta(days=1)).strftime("%Y-%m-%d") if hr == 0 else dt.date().strftime("%Y-%m-%d")
        b.setdefault(f"{ds} (Fascia {nm})", []).append(h)
    return b

def genera(dt_utc, n_run):
    dt_loc = dt_utc.astimezone(pytz.timezone("Europe/Rome"))
    dest = regrid.RegularGrid(CRS.from_string("epsg:4326"), 300, 300, 6.0, 10.5, 43.5, 46.8)
    dom = domains.Domain.from_bbox(bbox=bounds.BoundingBox(6.0, 10.5, 43.5, 46.8, ccrs.Geodetic()), name="Piemonte")
    f_reg = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    f_prov = cfeature.ShapelyFeature(shpreader.Reader("shapefiles/ProvCM01012026_WGS84.shp").geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':') if os.path.exists("shapefiles/ProvCM01012026_WGS84.shp") else None

    for bn, ore in raggruppa(dt_loc).items():
        req = ogd_api.Request("ogd-forecasting-icon-ch2", "CLCM", dt_utc, True, [f"P{l//24}DT{l%24}H" for l in ore])
        try:
            print(f"  ⬇️  Scarico dati CLCM per {len(ore)} ore...")
            vm_raw = scarica_variabile_con_retry(req)
            vm = vm_raw.mean(dim="eps")
            print(f"  ✅ Dati scaricati: {len(ore)} ore")
        except Exception as e:
            print(f"  ❌ Salto il blocco {bn} dopo 3 tentativi. Errore: {e}")
            continue
            
        fp = []
        for h in ore:
            ch = earthkit.plots.Map(domain=dom)
            ch.grid_cells(regrid.iconremap(vm.sel(lead_time=np.timedelta64(h, 'h')), dest), x="lon", y="lat", style=Style(colors=["#4292c6", "#6baed6", "#9ecae1", "#c6dbef", "#deebf7", "#f0f0f0", "#d9d9d9", "#bdbdbd", "#969696", "#737373"], levels=[0,10,20,30,40,50,60,70,80,90,100]))
            ch.ax.add_feature(f_reg); ch.ax.add_feature(f_prov) if f_prov else ch.borders()
            ch.title(f"ICON-CH2 EPS - Cloud Cover Middle (%)\nRun: {dt_utc.strftime('%d/%m/%Y %H:%M UTC')} | Ore {(dt_loc + timedelta(hours=h)).strftime('%H:%M del %d/%m')}"); ch.legend(label="CLCM (%)")
            ch.save(f"h_{h}.png"); fp.append(f"h_{h}.png"); plt.close(ch.fig)
        invia_album(fp, f"ICON-CH2 EPS: Dettaglio Nubi Medie (CLCM)\n{bn}\nRun {n_run}")
        [os.remove(f) for f in fp]; time.sleep(15)

if __name__ == "__main__":
    d = next((requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={"latitude":LATITUDE,"longitude":LONGITUDE,"hourly":"temperature_2m","models":"meteoswiss_icon_ch2_ensemble_mean","timezone":"Europe/Rome","past_days":1,"forecast_days":6}).json() for _ in range(3) if time.sleep(1) is None), {})
    if d: 
        is_n, nr, dt = estrai_limiti_run(d.get("hourly",{}), "temperature_2m", d.get("utc_offset_seconds",0))
        if is_n: genera(dt, nr)
