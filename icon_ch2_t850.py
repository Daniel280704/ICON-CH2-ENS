import os, sys, time, json, requests, urllib3, pytz, gc
import numpy as np, matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from datetime import datetime, timedelta, timezone
import warnings

import cartopy.crs as ccrs, cartopy.feature as cfeature, cartopy.io.shapereader as shpreader
import earthkit.plots; from earthkit.plots.geo import bounds, domains; from earthkit.plots.styles import Style; from earthkit.data import config
from meteodatalab import ogd_api; from meteodatalab.operators import regrid; from rasterio.crs import CRS
from meteodatalab.operators.vertical_interpolation import interpolate_k2p

warnings.filterwarnings('ignore'); urllib3.disable_warnings(); config.set("cache-policy", "temporary")

LATITUDE, LONGITUDE, FILE_LAST_HOUR = 45.07, 7.54, "ultima_ora_icon_ch2_t850.txt"
TARGET_HPA = 850

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
    t, c, th = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"), os.getenv("TELEGRAM_THREAD_ID_52")
    if not t or not c: return
    m, f = [{"type": "photo", "media": f"attach://p_{i}", "caption": caption if i==0 else ""} for i in range(len(file_paths))], {f"p_{i}": open(p, "rb") for i, p in enumerate(file_paths)}
    try: requests.post(f"https://api.telegram.org/bot{t}/sendMediaGroup", data={"chat_id": c, "media": json.dumps(m), "message_thread_id": th} if th else {"chat_id": c, "media": json.dumps(m)}, files=f)
    except: pass
    finally: [v.close() for v in f.values()]

def raggruppa(dt_run):
    b = {}
    # Logica 3-oraria: raggruppa per giornata
    for h in range(3, 121, 3):
        dt = dt_run + timedelta(hours=h)
        ds = (dt.date() - timedelta(days=1)).strftime("%d/%m/%Y") if dt.hour == 0 else dt.date().strftime("%d/%m/%Y")
        b.setdefault(f"Data: {ds}", []).append(h)
    return b

def genera(dt_utc, n_run):
    dt_loc = dt_utc.astimezone(pytz.timezone("Europe/Rome"))
    dest = regrid.RegularGrid(CRS.from_string("epsg:4326"), 300, 300, 6.0, 10.5, 43.5, 46.8)
    dom = domains.Domain.from_bbox(bbox=bounds.BoundingBox(6.0, 10.5, 43.5, 46.8, ccrs.Geodetic()), name="Piemonte")
    f_reg = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    f_prov = cfeature.ShapelyFeature(shpreader.Reader("shapefiles/ProvCM01012026_WGS84.shp").geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':') if os.path.exists("shapefiles/ProvCM01012026_WGS84.shp") else None

    my_levels = list(range(-25, 36))
    lats, lons = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92], [7.68, 7.55, 8.20, 8.61, 8.42, 8.61, 8.05, 8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    for bn, ore in raggruppa(dt_loc).items():
        fp = []
        for h in ore:
            try:
                lt_str = [f"P{h//24}DT{h%24}H"]
                req_t = ogd_api.Request("ogd-forecasting-icon-ch2", "T", dt_utc, True, lt_str)
                req_p = ogd_api.Request("ogd-forecasting-icon-ch2", "P", dt_utc, True, lt_str)
                t_mean = ogd_api.get_from_ogd(req_t).mean(dim="eps")
                p_mean = ogd_api.get_from_ogd(req_p).mean(dim="eps")
                
                t_int = interpolate_k2p(field=t_mean, mode="linear_in_lnp", p_field=p_mean, p_tc_values=[TARGET_HPA], p_tc_units='hPa')
                del t_mean, p_mean; gc.collect()

                val_celsius = t_int.squeeze(drop=True) - 273.15
                var_geo = regrid.iconremap(val_celsius, dest)
                
                ch = earthkit.plots.Map(domain=dom)
                ch.grid_cells(var_geo, x="lon", y="lat", style=Style(colors="turbo", levels=my_levels))
                ch.ax.add_feature(f_reg); ch.ax.add_feature(f_prov) if f_prov else ch.borders()
                
                try:
                    lon_arr, lat_arr = var_geo.lon.values if hasattr(var_geo, 'lon') else var_geo.x.values, var_geo.lat.values if hasattr(var_geo, 'lat') else var_geo.y.values
                    val_arr, step = var_geo.values, 25
                    for i in range(10, val_arr.shape[0], step):
                        for j in range(10, val_arr.shape[1], step):
                            v = val_arr[i, j]
                            if not np.isnan(v):
                                x_c = lon_arr[j] if val_arr.shape == (len(lat_arr), len(lon_arr)) else lon_arr[i]
                                y_c = lat_arr[i] if val_arr.shape == (len(lat_arr), len(lon_arr)) else lat_arr[j]
                                ch.ax.text(x_c, y_c, f"{v:.0f}", color='black', fontsize=6, ha='center', va='center', transform=ccrs.PlateCarree(), path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])
                except Exception: pass

                for lon, lat, sigla in zip(lons, lats, sigle):
                    ch.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                    ch.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree(), path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

                ch.title(f"ICON-CH2 EPS - T {TARGET_HPA}hPa (°C)\nRun: {dt_utc.strftime('%d/%m/%Y %H:%M UTC')} | Ore {(dt_loc + timedelta(hours=h)).strftime('%H:%M del %d/%m')}")
                ch.legend(label=f"Temperatura {TARGET_HPA}hPa (°C)")
                ch.save(f"h_{h}.png"); fp.append(f"h_{h}.png"); plt.close(ch.fig)
                
                del t_int, val_celsius, var_geo, ch; gc.collect()

            except Exception: continue
            
        invia_album(fp, f"ICON-CH2 EPS: Dettaglio Temp {TARGET_HPA}hPa\n{bn}\nRun {n_run}")
        [os.remove(f) for f in fp if os.path.exists(f)]; time.sleep(15)

if __name__ == "__main__":
    d = next((requests.get("https://ensemble-api.open-meteo.com/v1/ensemble", params={"latitude":LATITUDE,"longitude":LONGITUDE,"hourly":"temperature_2m","models":"meteoswiss_icon_ch2_ensemble_mean","timezone":"Europe/Rome","past_days":1,"forecast_days":6}).json() for _ in range(3) if time.sleep(1) is None), {})
    if d: 
        is_n, nr, dt = estrai_limiti_run(d.get("hourly",{}), "temperature_2m", d.get("utc_offset_seconds",0))
        if is_n: genera(dt, nr)