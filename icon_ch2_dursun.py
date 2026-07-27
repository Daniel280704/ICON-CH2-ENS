import os, sys, time, json, requests, urllib3, pytz
import numpy as np, matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import warnings
import cartopy.crs as ccrs, cartopy.feature as cfeature, cartopy.io.shapereader as shpreader
import earthkit.data
import earthkit.plots; from earthkit.plots.geo import bounds, domains; from earthkit.plots.styles import Style; from earthkit.data import config
from meteodatalab import ogd_api; from meteodatalab.operators import regrid; from rasterio.crs import CRS

warnings.filterwarnings('ignore'); urllib3.disable_warnings(); config.set("cache-policy", "temporary")

LATITUDE, LONGITUDE, FILE_LAST_HOUR = 45.07, 7.54, "ultima_ora_icon_ch2_dursun.txt"

def scarica_variabile_con_retry(request, max_retries=6):
    """Tenta lo scaricamento fino a max_retries, svuotando la cache in caso di fallimento."""
    for tentativo in range(max_retries):
        try:
            if tentativo > 0:
                print(f"    🔄 Retry {tentativo + 1}/{max_retries} in corso...")

            # Tenta il download vero e proprio
            return ogd_api.get_from_ogd(request)

        except Exception as e:
            if tentativo == max_retries - 1:
                print(f"    💥 Fallimento definitivo dopo {max_retries} tentativi.")
                raise e

            # Pausa progressiva (10s, 20s, 30s...) per far sbloccare il server
            delay = 10 * (tentativo + 1)
            print(f"    ⚠️ Errore o blocco di rete: {e}")
            print(f"    🧹 Svuoto la cache corrotta e attendo {delay}s...")

            # FORZATURA: Svuota la cache di earthkit per eliminare i file a metà
            try:
                earthkit.data.cache.purge()
            except Exception:
                pass

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
    t, c, th = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), os.getenv("TELEGRAM_THREAD_ID_23")
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

    # Scala in minuti (diviso 60), dal grigio chiaro al sole brillante (giallo scuro/arancio)
    my_levels = [0, 5, 10, 20, 30, 40, 50, 60]
    my_colors = ["#e0e0e0", "#fff5cc", "#ffeb99", "#ffe066", "#ffd633", "#ffcc00", "#ff9900"]

    for bn, ore in raggruppa(dt_loc).items():
        # Calcolando la differenza, dobbiamo richiedere anche h-1
        need = list(ore); need.insert(0, ore[0] - 1) if ore[0] > 1 else None
        req = ogd_api.Request("ogd-forecasting-icon-ch2", "DURSUN", dt_utc, True, [f"P{l//24}DT{l%24}H" for l in need])
        try:
            print(f"  ⬇️  Scarico dati DURSUN per {len(need)} ore...")
            vm_raw = scarica_variabile_con_retry(req)
            vm = vm_raw.mean(dim="eps")
            print(f"  ✅ Dati scaricati: {len(need)} ore")
        except Exception as e:
            print(f"  ❌ Salto il blocco {bn} dopo 3 tentativi. Errore: {e}")
            continue

        fp = []
        for h in ore:
            # Calcolo accumulo orario e trasformazione secondi -> minuti
            if h == 1: diff_sec = vm.sel(lead_time=np.timedelta64(h, 'h'))
            else: diff_sec = vm.sel(lead_time=np.timedelta64(h, 'h')) - vm.sel(lead_time=np.timedelta64(h-1, 'h'))
            
            diff_min = diff_sec / 60.0
            
            # MODIFICA: Utilizziamo np.clip solo sull'array '.values' per limitare forzatamente i valori tra 0 e 59.99.
            # Questo impedisce la perdita dei metadati (metadata) necessari a meteodatalab per il regridding.
            diff_min.values = np.clip(diff_min.values, 0, 59.99)

            ch = earthkit.plots.Map(domain=dom)
            ch.grid_cells(regrid.iconremap(diff_min, dest), x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))
            ch.ax.add_feature(f_reg); ch.ax.add_feature(f_prov) if f_prov else ch.borders()

            start_l, end_l = dt_loc + timedelta(hours=h-1), dt_loc + timedelta(hours=h)
            ch.title(f"ICON-CH2 EPS - Minuti di Sole orari (min/h)\nRun: {dt_utc.strftime('%d/%m/%Y %H:%M UTC')} | {start_l.strftime('%H:%M')} - {end_l.strftime('%H:%M del %d/%m')}")
            ch.legend(label="DURSUN (min)")

            ch.save(f"h_{h}.png"); fp.append(f"h_{h}.png"); plt.close(ch.fig)

        invia_album(fp, f"ICON-CH2 EPS: Dettaglio Irraggiamento Solare\n{bn}\nRun {n_run}")
        [os.remove(f) for f in fp]; time.sleep(15)

if __name__ == "__main__":
    d = next((requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={"latitude":LATITUDE,"longitude":LONGITUDE,"hourly":"temperature_2m","models":"meteoswiss_icon_ch2_ensemble_mean","timezone":"Europe/Rome","past_days":1,"forecast_days":6}).json() for _ in range(3) if time.sleep(1) is None), {})
    if d: 
        is_n, nr, dt = estrai_limiti_run(d.get("hourly",{}), "temperature_2m", d.get("utc_offset_seconds",0))
        if is_n: genera(dt, nr)