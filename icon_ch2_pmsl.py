import os, sys, time, json, requests, urllib3, pytz, gc
import numpy as np, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from datetime import datetime, timedelta, timezone
import warnings

import cartopy.crs as ccrs, cartopy.feature as cfeature, cartopy.io.shapereader as shpreader
import earthkit.plots; from earthkit.plots.geo import bounds, domains; from earthkit.plots.styles import Style; from earthkit.data import config
from meteodatalab import ogd_api; from meteodatalab.operators import regrid; from rasterio.crs import CRS

warnings.filterwarnings('ignore'); urllib3.disable_warnings(); config.set("cache-policy", "temporary")

LATITUDE, LONGITUDE, FILE_LAST_HOUR = 45.07, 7.54, "ultima_ora_icon_ch2_pmsl.txt"

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
    t, c, th = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), os.getenv("TELEGRAM_THREAD_ID_30")
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

    # Scala colori da pressioni molto basse a molto alte
    my_levels = list(range(970, 1042, 2))
    lats, lons = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92], [7.68, 7.55, 8.20, 8.61, 8.42, 8.61, 8.05, 8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    for bn, ore in raggruppa(dt_loc).items():
        try: vm = ogd_api.get_from_ogd(ogd_api.Request("ogd-forecasting-icon-ch2", "PMSL", dt_utc, True, [f"P{l//24}DT{l%24}H" for l in ore])).mean(dim="eps")
        except: continue
        fp = []
        for h in ore:
            # Conversione da Pascal a hPa
            val_hpa = vm.sel(lead_time=np.timedelta64(h, 'h')) / 100.0
            var_geo = regrid.iconremap(val_hpa, dest)
            
            ch = earthkit.plots.Map(domain=dom)
            ch.grid_cells(var_geo, x="lon", y="lat", style=Style(colors="turbo", levels=my_levels))
            ch.ax.add_feature(f_reg); ch.ax.add_feature(f_prov) if f_prov else ch.borders()
            
            # --- ISOBARE OGNI 2 hPa ---
            try:
                coords_x = var_geo.coords['lon'].values if 'lon' in var_geo.coords else var_geo.coords['x'].values
                coords_y = var_geo.coords['lat'].values if 'lat' in var_geo.coords else var_geo.coords['y'].values
                val_arr = var_geo.values
                
                if coords_x.ndim == 1:
                    lon_grid, lat_grid = np.meshgrid(coords_x, coords_y)
                else:
                    lon_grid, lat_grid = coords_x, coords_y
                    
                contour_levels = np.arange(950, 1060, 2)
                cs = ch.ax.contour(lon_grid, lat_grid, val_arr, levels=contour_levels, colors='black', linewidths=0.8, alpha=0.8, transform=ccrs.PlateCarree())
                texts = ch.ax.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')
                for txt in texts:
                    txt.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])
            except Exception as e:
                print(f"Errore isobare: {e}")

            for lon, lat, sigla in zip(lons, lats, sigle):
                ch.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree(), zorder=5)
                ch.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree(), path_effects=[pe.withStroke(linewidth=1.5, foreground="white")], zorder=6)

            ch.title(f"ICON-CH2 EPS - MSLP (hPa)\nRun: {dt_utc.strftime('%d/%m/%Y %H:%M UTC')} | Ore {(dt_loc + timedelta(hours=h)).strftime('%H:%M del %d/%m')}")
            ch.legend(label="Pressione s.l.m. (hPa)")
            ch.save(f"h_{h}.png"); fp.append(f"h_{h}.png"); plt.close(ch.fig)
            
        invia_album(fp, f"ICON-CH2 EPS: Dettaglio Pressione MSLP\n{bn}\nRun {n_run}")
        [os.remove(f) for f in fp]; time.sleep(15)

if __name__ == "__main__":
    d = next((requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={"latitude":LATITUDE,"longitude":LONGITUDE,"hourly":"temperature_2m","models":"meteoswiss_icon_ch2_ensemble_mean","timezone":"Europe/Rome","past_days":1,"forecast_days":6}).json() for _ in range(3) if time.sleep(1) is None), {})
    if d: 
        is_n, nr, dt = estrai_limiti_run(d.get("hourly",{}), "temperature_2m", d.get("utc_offset_seconds",0))
        if is_n: genera(dt, nr)