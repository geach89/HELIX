import numpy as np
import netCDF4 as nc
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.coordinates import get_sun
import datetime as dt

path = ''#Designate the path 
    #in which the required data files have been saved.

def rolling_window(a, window):
    #This function calculates a moving average calculated over the specified
    #window-width.
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad - int(pad/2), np.zeros(len(a.shape), dtype=np.int32)
                   + int(pad/2)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def filter_data(file, location = 'OP'):
    #This function reads in the data and combines the 12 readout channels. It
    #acquires the solar position for each time interval, then performs several
    #filters. It returns the filtered data, corresponding timestamps and solar
    #positions, and the index of local solar midnight.
    
    df = nc.Dataset(file)
    data = np.zeros((df.variables['photons_ch0'].shape[0],360))
    for i in range(12):
        ch_data = df.variables['photons_ch{0}'.format(i)][:,:360]
        data += ch_data
    shots = df.variables['shots_ch0'][:]
    
    
    times = np.zeros(df.variables['photons_ch0'].shape[0])

    for i in range(df.variables['photons_ch0'].shape[0]):
        times[i] = (df.variables['time'][:][i]/1000
                    + df.variables['time_offset'][:])
        
    sun_alt = np.zeros(len(times))
    
    loc = EarthLocation(lat=48.087*u.deg, lon=11.280*u.deg, height=590*u.m)
    
    for i in range(len(times)):
        timestamp = Time(times[i], format='unix')
        frame = AltAz(obstime = timestamp, location = loc)
        sunaltaz = get_sun(timestamp).transform_to(frame)
        sun_alt[i] = sunaltaz.alt.value

    bg = np.mean(data[:,50:125], axis=1)/shots
        
    N = max(np.where(sun_alt < -7)[0][0], 1)
    M = min(np.where(sun_alt < -7)[0][-1], len(bg) - 2)
    ds = bg[N:M]
    times_section = times[N:M]
    data_section = data[N:M,:]
    shots_section = shots[N:M]
    sun_alt_section = sun_alt[N:M]
    
    ds_median = np.median(rolling_window(ds, 60), axis=-1)
    mask = np.where(ds < 2*ds_median)[0]
    
    he = (np.mean(data_section[mask,150:], axis=1)/shots_section[mask]
          - ds[mask])
    he_median = np.median(rolling_window(he, 60), axis=-1)
    std = np.std(rolling_window(he, 60), axis=-1)
    mask_temp = np.where(np.abs(he - he_median) < 3*std)[0]
    mask = mask[mask_temp]
    
    metric = (np.mean(data_section[mask,15:20], axis=1)/shots_section[mask]
              - np.mean(data_section[mask,50:125], axis=1)/shots_section[mask])
    threshhold = 0.4*np.max(metric)
    mask_temp = np.where(metric > threshhold)[0]
    mask = mask[mask_temp]
    
    split = np.where(sun_alt_section[mask] == 
                     np.min(sun_alt_section[mask]))[0][0]
    
    return data_section[mask,:], times_section[mask], sun_alt_section[mask], split

def get_He_profiles(data):
    #This function takes filtered data and calculates raw photon return
    #profiles and metastable helium density profiles, each with associated
    #uncertainties, over 100 km vertical bins.
    bg = np.sum(data[:,50:125], axis=1)
    rayleigh = np.sum(np.mean(data[:,18:25], axis=1) - np.mean(data[:,50:125],
                      axis=1))
    
    bin_size = 100 #km
    bg_bin = 75 #km
    
    n = int(bin_size / 2.)
    N = int(360  / n)
    
    binned_data = np.zeros((data.shape[0], N), dtype=np.float64)
    binned_bg_sub = np.zeros((data.shape[0], N), dtype=np.float64)
    binned_bg_sub_error = np.zeros((data.shape[0], N), dtype=np.float64)
    z_squared = np.zeros(N)
    
    for i in range(N):
        binned_data[:,i] = np.sum(data[:,n*i:n*(i + 1)], axis=1)
        z_squared[i] = np.mean(np.arange(1,700,2)[n*i:n*(i + 1)]**2)
         
    for i in range(data.shape[0]):
        binned_bg_sub[i,:] = binned_data[i,:] - n/bg_bin*bg[i]
        binned_bg_sub_error[i,:] = np.sqrt(binned_data[i,:] + (n/75)**2*bg[i])
        
    integ_bin_bg_sub = np.sum(binned_bg_sub, axis = 0)
    integ_bin_bg_sub_error = np.sqrt(np.sum(binned_bg_sub_error**2,
                                                    axis = 0))    
    
    n_R_over_z_R_squared = np.mean(n_dens[72:101]/alt[72:101]**2) # m^-5 
    sig_R = 3.0 * 10**-32 # m^2
    delta_z_R = 2000 # m
    sig_He = 2.5 * 10**-16 # m^2
    
    conversion_factor = (rayleigh * sig_He * bin_size / n_R_over_z_R_squared /
                         z_squared / sig_R / delta_z_R * 10**3) # counts cm^3
    density = integ_bin_bg_sub / conversion_factor # cm^-3
    density_error = np.sqrt((integ_bin_bg_sub_error / conversion_factor)**2
                            + (0.3*density)**2) # cm^-3
    
    return integ_bin_bg_sub, integ_bin_bg_sub_error, density, density_error

#Loads the MSIS2.0 profile for calibration of the helium return via the
#Rayleigh return in the lower atmosphere.
profile = np.loadtxt(path + 'msis20output.txt',
                     skiprows=1).T
alt = profile[0]
n_dens = profile[1]

#Loads the lidar return data. Choose the time period for which helium profiles
#will be returned.
time_periods = ['Jan-Feb', 'Feb-Mar']
time_period = time_periods[0]

if time_period == 'Jan-Feb':
    data_files = ['20220113-1456.nc',
        '20220114-1424.nc',
        '20220205-1541.nc',
        '20220208-1356.nc',
        '20220209-1612.nc',
        '20220213-1610.nc']

if time_period == 'Feb-Mar':
    data_files=['20220223-1630.nc',
        '20220227-1657.nc',
        '20220228-1512.nc',
        '20220301-1708.nc',
        '20220303-1738.nc']
    
    
#Filters the data, gets solar elevation data and separates into morning/evening
data_evenings = []
data_mornings = []
times_evenings = []
times_mornings = []
sun_alts_evenings = []
sun_alts_mornings = []


for i in range(len(data_files)):
    file = path + data_files[i]
    data, times, sun_alts, split = filter_data(file)
    data_evenings.append(data[:split,:])
    data_mornings.append(data[split:,:])
    times_evenings.append(times[:split])
    times_mornings.append(times[split:])
    sun_alts_evenings.append(sun_alts[:split])
    sun_alts_mornings.append(sun_alts[split:])

#Sorts the data based on solar elevation angles.
bin_edges = [-7, -20, -30, -40]
sza_edges = 90 - np.array(bin_edges)

data_sza97to110_evening = np.zeros((0,360))
data_sza110to120_evening = np.zeros((0,360)) 
data_sza120to130_evening = np.zeros((0,360)) 
data_sza130plus_evening = np.zeros((0,360)) 

data_sza97to110_morning = np.zeros((0,360)) 
data_sza110to120_morning = np.zeros((0,360)) 
data_sza120to130_morning = np.zeros((0,360)) 
data_sza130plus_morning = np.zeros((0,360)) 


for i in range(len(data_files)):
    data_sza97to110_evening = np.vstack((data_sza97to110_evening,
                    data_evenings[i][(sun_alts_evenings[i] > bin_edges[1])
                    & (sun_alts_evenings[i] < bin_edges[0])]))
    data_sza110to120_evening = np.vstack((data_sza110to120_evening,
                    data_evenings[i][(sun_alts_evenings[i] > bin_edges[2])
                    & (sun_alts_evenings[i] < bin_edges[1])]))
    data_sza120to130_evening = np.vstack((data_sza120to130_evening,
                    data_evenings[i][(sun_alts_evenings[i] > bin_edges[3])
                    & (sun_alts_evenings[i] < bin_edges[2])]))
    data_sza130plus_evening = np.vstack((data_sza130plus_evening,
                    data_evenings[i][(sun_alts_evenings[i] < bin_edges[3])]))
    
    data_sza97to110_morning = np.vstack((data_sza97to110_morning,
                    data_mornings[i][(sun_alts_mornings[i] > bin_edges[1])
                    & (sun_alts_mornings[i] < bin_edges[0])]))
    data_sza110to120_morning = np.vstack((data_sza110to120_morning,
                    data_mornings[i][(sun_alts_mornings[i] > bin_edges[2])
                    & (sun_alts_mornings[i] < bin_edges[1])]))
    data_sza120to130_morning = np.vstack((data_sza120to130_morning,
                    data_mornings[i][(sun_alts_mornings[i] > bin_edges[3])
                    & (sun_alts_mornings[i] < bin_edges[2])]))
    data_sza130plus_morning = np.vstack((data_sza130plus_morning,
                    data_mornings[i][(sun_alts_mornings[i] < bin_edges[3])]))

#Calculates lidar return profiles in counts with background subtracted and
#helium number density profiles in cm^-3, each with associated uncertainties
#and computed over 100 km vertical bins. The data is organized as follows:
#profile_XXX[0] is the lidar return profile in counts
#profile_XXX[1] is the corresponding uncertainty
#profile_XXX[2] is the helium number density profile in cm^-3
#profile_XXX[3] is the corresponding uncertainty

profile_sza97to110_evening = get_He_profiles(data_sza97to110_evening) 
profile_sza110to120_evening = get_He_profiles(data_sza110to120_evening)
profile_sza120to130_evening = get_He_profiles(data_sza120to130_evening)
profile_sza130plus_evening = get_He_profiles(data_sza130plus_evening)

profile_sza97to110_morning = get_He_profiles(data_sza97to110_morning)
profile_sza110to120_morning = get_He_profiles(data_sza110to120_morning)
profile_sza120to130_morning = get_He_profiles(data_sza120to130_morning)
profile_sza130plus_morning = get_He_profiles(data_sza130plus_morning)



alts = np.arange(7)*100 + 50
figure_data = np.vstack((alts,
                         profile_sza97to110_evening[2],
                         profile_sza97to110_evening[3],
                         profile_sza110to120_evening[2],
                         profile_sza110to120_evening[3],
                         profile_sza120to130_evening[2],
                         profile_sza120to130_evening[3],
                         profile_sza130plus_evening[2],
                         profile_sza130plus_evening[3],
                         profile_sza97to110_morning[2],
                         profile_sza97to110_morning[3],
                         profile_sza110to120_morning[2],
                         profile_sza110to120_morning[3],
                         profile_sza120to130_morning[2],
                         profile_sza120to130_morning[3],
                         profile_sza130plus_morning[2],
                         profile_sza130plus_morning[3]))[:,1:].T


if time_period == 'Jan-Feb':
    fn = path + 'paper_figure_data/active_measurement_Jan-Feb.txt'
if time_period == 'Feb-Mar':
    fn = path + 'paper_figure_data/active_measurement_Feb-Mar.txt'
    
np.savetxt(fn, figure_data, delimiter=',')

print(dt.datetime.now())