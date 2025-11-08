# CM618
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import warnings, os
warnings.filterwarnings("ignore")

# ================================
#  FILE NAMES – same directory
# ================================
ERA5_FILE = "era5_t2m_JJ_1981-2025.nc"
NAT_FILE  = "CESM2_histnat_merged_1950-2014.nc"
ALL_FILE  = "tas_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"

OUTDIR = "./analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# ================================
#  LOAD DATA
# ================================
ds_era = xr.open_dataset(ERA5_FILE)
ds_nat = xr.open_dataset(NAT_FILE)
ds_all = xr.open_dataset(ALL_FILE)

# --- fix coordinates for ERA5 ---
coord_map = {}
if "longitude" in ds_era.coords:
    coord_map["longitude"] = "lon"
if "latitude" in ds_era.coords:
    coord_map["latitude"] = "lat"
for candidate in ["time", "date", "valid_time"]:
    if candidate in ds_era.dims or candidate in ds_era.coords:
        if candidate != "time":
            coord_map[candidate] = "time"
        break
if coord_map:
    ds_era = ds_era.rename(coord_map)

# --- fix variable names ---
def find_var(ds, pref):
    return pref if pref in ds.data_vars else list(ds.data_vars)[0]

era_var = find_var(ds_era, 't2m')
nat_var = find_var(ds_nat, 'tas')
all_var = find_var(ds_all, 'tas')

# --- Kelvin → °C ---
for ds, var in [(ds_era, era_var), (ds_nat, nat_var), (ds_all, all_var)]:
    if float(ds[var].mean()) > 100:
        ds[var] = ds[var] - 273.15

# --- harmonise lon ---
def normalize_lon(ds):
    if "lon" in ds.coords and ds.lon.max() > 180:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby('lon')
    return ds

ds_era, ds_nat, ds_all = map(normalize_lon, [ds_era, ds_nat, ds_all])

# ================================
#  REGION + JJ means
# ================================
lat_slice, lon_slice = slice(35,60), slice(-10,40)

def jj_mean(da):
    jj = da.sel(time=da['time'].dt.month.isin([6,7]))
    return jj.groupby('time.year').mean('time')

def area_mean(da):
    w = np.cos(np.deg2rad(da.lat))
    return (da * w).mean(('lat','lon')) / w.mean()

era_jj = jj_mean(ds_era[era_var])
nat_jj = jj_mean(ds_nat[nat_var].sel(lat=lat_slice, lon=lon_slice))
all_jj = jj_mean(ds_all[all_var].sel(lat=lat_slice, lon=lon_slice))

era_reg = area_mean(era_jj)
nat_reg = area_mean(nat_jj)
all_reg = area_mean(all_jj)

# ================================
#  SYNC YEARS (common 1981–2010)
# ================================
common_start, common_end = 1981, 2010
era_reg = era_reg.sel(year=slice(common_start, common_end))
nat_reg = nat_reg.sel(year=slice(common_start, common_end))
all_reg = all_reg.sel(year=slice(common_start, common_end))

# ================================
#  BASELINE + THRESHOLD
# ================================
baseline_nat = nat_reg.sel(year=slice(common_start, common_end))
thr = np.percentile(baseline_nat, 95)
print(f"95th-percentile NAT threshold = {thr:.2f} °C")

# ================================
#  PROBABILITIES + PR
# ================================
p_nat = float((nat_reg > thr).sum() / nat_reg.size)
p_all = float((all_reg > thr).sum() / all_reg.size)
PR = p_all / p_nat if p_nat>0 else np.nan
print(f"P(event|ALL)={p_all:.3f}  P(event|NAT)={p_nat:.3f}  →  PR={PR:.1f}×")

# ================================
#  TIME-SERIES PLOT (same years)
# ================================
plt.figure(figsize=(9,4))
plt.plot(nat_reg.year, nat_reg, color='tab:blue', label='CESM2 NAT')
plt.plot(all_reg.year, all_reg, color='tab:orange', label='CESM2 ALL')
plt.plot(era_reg.year, era_reg, color='tab:red', label='ERA5')
plt.axhline(thr, color='k', ls='--', label='95th % NAT thr')
plt.ylabel('JJ mean °C (Europe)')
plt.xlabel('Year')
plt.title(f'European JJ Mean Temperature ({common_start}-{common_end})')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'timeseries_all_nat_era5_synced.png'), dpi=200)
plt.show()

# ================================
#  HISTOGRAMS (NAT vs ALL)
# ================================
plt.figure(figsize=(7,4))
plt.hist(nat_reg, bins=15, alpha=0.6, label='NAT')
plt.hist(all_reg, bins=15, alpha=0.6, label='ALL')
plt.axvline(thr, color='k', ls='--', label='95th % NAT thr')
plt.xlabel('JJ mean °C')
plt.ylabel('Frequency (years)')
plt.legend()
plt.title(f'CESM2 ALL vs NAT ({common_start}-{common_end}) (PR ≈ {PR:.1f}×)')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'hist_all_nat_synced.png'), dpi=200)
plt.show()

# ================================
#  ERA5 vs NAT distribution
# ================================
plt.figure(figsize=(7,4))
plt.hist(nat_reg, bins=15, alpha=0.6, label='NAT')
plt.hist(era_reg, bins=15, alpha=0.6, label='ERA5')
plt.axvline(thr, color='k', ls='--', label='95th % NAT thr')
plt.legend(); plt.xlabel('JJ mean °C')
plt.title(f'ERA5 vs CESM2 NAT ({common_start}-{common_end})')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR,'hist_era5_nat_synced.png'), dpi=200)
plt.show()

# ================================
#  MAP OF ERA5 ANOMALY (optional)
# ================================
try:
    mean_nat = ds_nat[nat_var].sel(time=slice('1981','2010')).mean('time')
    mean_era = ds_era[era_var].sel(time=slice('1981','2010')).mean('time')
    era_last = ds_era[era_var].isel(time=-1)
    anom = era_last - mean_era
    plt.figure(figsize=(7,4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    anom.sel(lat=lat_slice, lon=lon_slice).plot(
        ax=ax, transform=ccrs.PlateCarree(), cmap='RdYlBu_r',
        cbar_kwargs={'label':'°C'})
    ax.coastlines(); ax.set_extent([-10,40,35,60])
    plt.title('ERA5 latest JJ anomaly vs 1981–2010')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR,'map_era5_anomaly.png'), dpi=200)
    plt.show()
except Exception as e:
    print("Skipped map:", e)

# ================================
#  BASIC STATS
# ================================
print("\n--- SUMMARY ---")
print(f"NAT mean ({common_start}-{common_end}): {float(nat_reg.mean()):.2f} °C")
print(f"ALL mean ({common_start}-{common_end}): {float(all_reg.mean()):.2f} °C")
print(f"ERA5 mean ({common_start}-{common_end}): {float(era_reg.mean()):.2f} °C")
print(f"Probability Ratio (ALL/NAT): {PR:.1f}×")
print(f"Figures saved in: {OUTDIR}")
