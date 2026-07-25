def genera_album_orari(dt_run_utc: datetime, nome_run: str):
    rome_tz = pytz.timezone("Europe/Rome")
    dt_run_local = dt_run_utc.astimezone(rome_tz)
    
    blocchi = raggruppa_in_blocchi(dt_run_local)

    xmin, xmax, ymin, ymax = 6.0, 10.5, 43.5, 46.8
    nx, ny = 300, 300
    destination = regrid.RegularGrid(CRS.from_string("epsg:4326"), nx, ny, xmin, xmax, ymin, ymax)

    # Nuovi livelli e colori uniformati allo script della grandine
    my_levels = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    my_colors = [
        "#a0e6ff", # 10-20%
        "#00a0ff", # 20-30%
        "#00ff00", # 30-40%
        "#ffff00", # 40-50%
        "#ffaa00", # 50-60%
        "#ff0000", # 60-70%
        "#cc0000", # 70-80%
        "#ff00ff", # 80-90%
        "#800080"  # 90-100%
    ]
    domain = domains.Domain.from_bbox(bbox=bounds.BoundingBox(xmin, xmax, ymin, ymax, ccrs.Geodetic()), name="Piemonte")

    regions_feature = cfeature.NaturalEarthFeature('cultural', 'admin_1_states_provinces', '10m', edgecolor='black', facecolor='none', linewidth=1.5)
    prov_feature = None
    shp_path = "shapefiles/ProvCM01012026_WGS84.shp"
    if os.path.exists(shp_path):
        prov_feature = cfeature.ShapelyFeature(shpreader.Reader(shp_path).geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none', linewidth=0.5, linestyle=':')

    lats = [45.07, 44.38, 44.90, 44.91, 45.32, 45.45, 45.56, 45.92]
    lons = [7.68,  7.55,  8.20,  8.61,  8.42,  8.61,  8.05,  8.55]
    sigle = ["TO", "CN", "AT", "AL", "VC", "NO", "BI", "VB"]

    for block_name, ore_list in blocchi.items():
        print(f"\nGenerazione album probabilità: {block_name}")
        
        lead_times_needed = list(ore_list)
        if ore_list[0] > 1:
            lead_times_needed.insert(0, ore_list[0] - 1)
            
        lead_times_str = [f"P{l // 24}DT{l % 24}H" for l in lead_times_needed]

        req = ogd_api.Request(
            collection="ogd-forecasting-icon-ch2",
            variable="TOT_PREC",
            ref_time=dt_run_utc,
            perturbed=True,
            lead_time=lead_times_str,
        )
        
        try:
            tot_prec = ogd_api.get_from_ogd(req)
        except Exception as e:
            print(f"Salto il blocco {block_name} causa errore download: {e}")
            continue

        percorsi_foto = []
        
        for h in ore_list:
            if h == 1:
                prec_diff = tot_prec.sel(lead_time=np.timedelta64(h, 'h'))
            else:
                prec_diff = tot_prec.sel(lead_time=np.timedelta64(h, 'h')) - tot_prec.sel(lead_time=np.timedelta64(h-1, 'h'))

            # Aggiunto .astype(float) per replicare esattamente la logica e rendere robusto il calcolo della media
            prob_xr = (prec_diff >= 0.5).astype(float).mean(dim="eps") * 100
            
            # Regrid sulla variabile di probabilità
            prec_geo = regrid.iconremap(prob_xr, destination)
            
            # Filtro visivo: mostra solo probabilità >= 10% per uniformità con l'altro script
            prec_geo = prec_geo.where(prec_geo >= 10)

            chart = earthkit.plots.Map(domain=domain)
            chart.grid_cells(prec_geo, x="lon", y="lat", style=Style(colors=my_colors, levels=my_levels))

            chart.ax.add_feature(regions_feature)
            if prov_feature:
                chart.ax.add_feature(prov_feature)
            else:
                chart.borders()

            chart.ax.plot(7.51, 45.07, marker='o', color='brown', markersize=6, transform=ccrs.PlateCarree())

            for lon, lat, sigla in zip(lons, lats, sigle):
                chart.ax.plot(lon, lat, marker='o', color='black', markersize=3, transform=ccrs.PlateCarree())
                chart.ax.text(lon + 0.05, lat + 0.05, sigla, color='black', fontsize=9, fontweight='bold', transform=ccrs.PlateCarree())

            start_local = dt_run_local + timedelta(hours=h-1)
            end_local = dt_run_local + timedelta(hours=h)
            str_valida = f"{start_local.strftime('%H:%M')} - {end_local.strftime('%H:%M del %d/%m')}"

            title = f"ICON-CH2 EPS - Probabilità Pioggia >= 0.5 mm/h (%)\nRun: {dt_run_utc.strftime('%d/%m/%Y %H:%M UTC')} | {str_valida}"
            chart.title(title)
            chart.legend(label="Probabilità (%)")

            filename = f"oraria_{h}.png"
            chart.save(filename)
            percorsi_foto.append(filename)
            
            plt.close(chart.fig)
        
        caption_album = f"ICON-CH2 EPS: Probabilità Pioggia oraria >= 0.5 mm\n{block_name}\nRun {nome_run}"
        invia_album_telegram(percorsi_foto, caption_album)
        
        for f in percorsi_foto:
            if os.path.exists(f): os.remove(f)
        del tot_prec
        time.sleep(15)