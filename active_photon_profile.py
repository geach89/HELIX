import numpy as np
import netCDF4 as nc
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.coordinates import get_sun


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

def get_count_profile(data):
    #This function takes filtered data and calculates raw photon return
    #profiles and metastable helium density profiles, each with associated
    #uncertainties, over 100 km vertical bins.
    bin_size = 100 #km
    total_bg = np.sum(data[:,50:125])*bin_size/150.
    
    n = int(bin_size / 2.)
    N = int(360  / n)
    
    binned_data = np.zeros((data.shape[0], N), dtype=np.float64)
    binned_data_error = np.zeros((data.shape[0], N), dtype=np.float64)

    for i in range(N):
        binned_data[:,i] = np.sum(data[:,n*i:n*(i + 1)], axis=1)
    binned_data_error = np.sqrt(binned_data)  
                 
    integ_bin = np.sum(binned_data, axis = 0)
    integ_bin_error = np.sqrt(np.sum(binned_data_error**2, axis = 0))    
    
    return integ_bin, integ_bin_error, np.ones(N)*total_bg

#Loads the lidar return data.
data_files = ['20220113-1456.nc',
         '20220114-1424.nc',
         '20220205-1541.nc',
        '20220208-1356.nc',
        '20220209-1612.nc',
        '20220213-1610.nc']

#Filters the data, gets solar elevation data and separates into morning/evening

data_mornings = []
times_mornings = []
sun_alts_mornings = []


for i in range(len(data_files)):
    file = path + data_files[i]
    data, times, sun_alts, split = filter_data(file)
    data_mornings.append(data[split:,:])
    times_mornings.append(times[split:])
    sun_alts_mornings.append(sun_alts[split:])

#Sorts the data based on solar elevation angles.
bin_edges = [-7, -20, -30, -40]
sza_edges = 90 - np.array(bin_edges)

data_sza97to110_morning = np.zeros((0,360)) 

for i in range(len(data_files)):  
    data_sza97to110_morning = np.vstack((data_sza97to110_morning,
                    data_mornings[i][(sun_alts_mornings[i] > bin_edges[1])
                    & (sun_alts_mornings[i] < bin_edges[0])]))

    
#Calculates the lidar return profile for the bottom 100 km at 2 km resolution
#and 10 second integration, then calculates the associated uncertainties.
#Writes the data to disk as a txt file with the following columns: altitude in
#km, lidar return profile in counts and the associated uncertainties in counts.
alts1 = np.arange(50)*2 + 1
profile1 = data_sza97to110_morning[0,:]
sigma_profile1 = np.sqrt(profile1)
bg_profile1 = np.mean(profile1[50:125])
figure_data1 = np.vstack((alts1, profile1[:50], sigma_profile1[:50],
                          np.ones(50)*bg_profile1)).T
fn1 = path + 'paper_figure_data/photon_profile1.txt'
np.savetxt(fn1, figure_data1, delimiter=',')

#Calculates the lidar return profile for 100 to 700 km and calculates the
#associated uncertainties. Writes the data to disk as a txt file with the
#following columns: altitude in km, lidar return profile in counts, the
#associated uncertainties in counts.
alts2 = np.arange(7)*100 + 50
profile_sza97to110_morning = get_count_profile(data_sza97to110_morning)
figure_data2 = np.vstack((alts2, profile_sza97to110_morning[0],
                         profile_sza97to110_morning[1],
                         profile_sza97to110_morning[2]))[:,1:].T
fn2 = path + 'paper_figure_data/photon_profile2.txt'
np.savetxt(fn2, figure_data2, delimiter=',')