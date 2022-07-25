import numpy as np
import netCDF4 as nc
import scipy.optimize as opt
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import EarthLocation, AltAz
from astropy.coordinates import get_sun

path = ''#Designate the path 
    #in which the required data files have been saved.

def scale(params, x, y, y_err):
    return np.sum(((x*params - y)/y_err)**2)
    
def rolling_window(a, window):
    pad = np.ones(len(a.shape), dtype=np.int32)
    pad[-1] = window-1
    pad = list(zip(pad - int(pad/2), np.zeros(len(a.shape),
                             dtype=np.int32) + int(pad/2)))
    a = np.pad(a, pad,mode='reflect')
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def filter_data(file, location, Rayleigh = False):
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
    if location == 'conj':
        conj_sun_alt = np.zeros(len(times))
        
    loc = EarthLocation(lat=48.087*u.deg, lon=11.28*u.deg, height=590*u.m)
    conj_loc = EarthLocation(lat=-35.6*u.deg, lon=21.3*u.deg, height=590*u.m)
    
    for i in range(len(times)):
        timestamp = Time(times[i], format='unix')
        frame = AltAz(obstime = timestamp, location = loc)
        sunaltaz = get_sun(timestamp).transform_to(frame)
        sun_alt[i] = sunaltaz.alt.value
        if location == 'conj': 
            frame = AltAz(obstime = timestamp, location = conj_loc)
            sunaltaz = get_sun(timestamp).transform_to(frame)
            conj_sun_alt[i] = sunaltaz.alt.value
    
    if Rayleigh:
        bg = np.mean(data[:,:], axis=1)/shots
    else:
        bg = np.mean(data[:,50:125], axis=1)/shots
    
    if location == 'OP':
        sun_alt_limit = -4
    if location == 'conj':
        sun_alt_limit = -10
        
    N = max(np.where(sun_alt < sun_alt_limit)[0][0], 1)
    M = min(np.where(sun_alt < sun_alt_limit)[0][-1], len(bg) - 2)
    bg_section = bg[N:M]
    data_section = data[N:M,:]
    shots_section = shots[N:M]
    sun_alt_section = sun_alt[N:M]
    if location == 'conj':
        conj_sun_alt_section = conj_sun_alt[N:M]
        
        
    bg_median = np.median(rolling_window(bg_section, 60), axis=-1)
    mask = np.where(bg_section < 2*bg_median)[0]
    
    he = (np.mean(data_section[mask,150:], axis=1)/shots_section[mask]
          - bg_section[mask])
    he_median = np.median(rolling_window(he, 60), axis=-1)
    he_std = np.std(rolling_window(he, 60), axis=-1)
    mask_temp = np.where(np.abs(he - he_median) < 3*he_std)[0]
    mask = mask[mask_temp]
    
    if not Rayleigh:
        metric = (np.mean(data_section[mask,15:20], axis=1)/shots_section[mask]
                  - np.mean(data_section[mask,50:125], axis=1)/shots_section[mask])
        threshhold = 0.4*np.max(metric)
        mask_temp = np.where(metric > threshhold)[0]
        mask = mask[mask_temp]
    
    split = np.where(sun_alt_section[mask] == np.min(sun_alt_section[mask]))[0][0]
    
    #if Rayleigh:
    #    baseline = np.mean(bg_section[mask][np.where(sun_alt_section[mask] < -30)[0]])
    #else:
    #    baseline = np.mean(bg_section[mask][np.where(sun_alt_section[mask] < -35)[0]])
    
    baseline = np.mean(bg_section[mask][np.where(sun_alt_section[mask] < -35)[0]])
    
    if location == 'conj':
        sun_alt_return = conj_sun_alt_section[mask]
    else:
        sun_alt_return = sun_alt_section[mask]
        
    return bg_section[mask], sun_alt_return, split, baseline

data_files1 = ['20220113-1456.nc',
         '20220114-1424.nc',
         '20220205-1541.nc',
        '20220208-1356.nc',
        '20220209-1612.nc',
        '20220213-1610.nc']
               
data_files2=['20220223-1630.nc',
        '20220227-1657.nc',
        '20220228-1512.nc',
        '20220301-1708.nc',
        '20220303-1738.nc']

data_files_Rayleigh=['20220310-1358.nc']

# Choose location at which the SZA will be computed: 'OP' for Oberpfaffenhofen
# and 'conj' for the geomagnetic conjugate point.
locations = ['OP', 'conj']
location = locations[1]


bg_even1 = np.array([])
sza_even1 = np.array([])
bg_morn1 = np.array([])
sza_morn1 = np.array([])
B = []
S = []

for i in range(len(data_files1)):
    file = path + data_files1[i]
    bg, sun_alt, split, base = filter_data(file, location=location,
                                           Rayleigh=False)
    if len(bg) > 0:
        if len(bg) > split:
            bg_morn1 = np.hstack((bg_morn1, 75000*(bg[split:] - base)))
            sza_morn1 = np.hstack((sza_morn1, sun_alt[split:]))
            bg_even1 = np.hstack((bg_even1, 75000*(bg[:split] - base)))
            sza_even1 = np.hstack((sza_even1, sun_alt[:split]))
        else:
            bg_even1 = np.hstack((bg_even1, 75000*(bg - base)))
            sza_even1 = np.hstack((sza_even1, sun_alt))
            
bg_even2 = np.array([])
sza_even2 = np.array([])
bg_morn2 = np.array([])
sza_morn2 = np.array([])

for i in range(len(data_files2)):
    file = path + data_files2[i]
    bg, sun_alt, split, base = filter_data(file, location=location,
                                           Rayleigh=False)
    if len(bg) > 0:
        if len(bg) > split:
            bg_morn2 = np.hstack((bg_morn2, 75000*(bg[split:] - base)))
            sza_morn2 = np.hstack((sza_morn2, sun_alt[split:]))
            bg_even2 = np.hstack((bg_even2, 75000*(bg[:split] - base)))
            sza_even2 = np.hstack((sza_even2, sun_alt[:split]))
        else:
            bg_even2 = np.hstack((bg_even2, 75000*(bg - base)))
            sza_even2 = np.hstack((sza_even2, sun_alt))

if location == 'OP':            
    file = path + data_files_Rayleigh[0]
    bg, sun_alt, split, base = filter_data(file, location=location,
                                       Rayleigh = 'True')

    bg_morn_Rayleigh = 75000*(bg[split:] - base)
    sza_morn_Rayleigh = sun_alt[split:]
    bg_even_Rayleigh = 75000*(bg[:split] - base)
    sza_even_Rayleigh = sun_alt[:split]

bg_total = np.hstack((bg_even1, bg_morn1, bg_even2, bg_morn2))
sza_total = np.hstack((sza_even1, sza_morn1, sza_even2, sza_morn2))

SZA = np.arange(-55, 25, 1)
B_even1 = np.zeros(len(SZA) - 1)
S_even1 = np.zeros(len(SZA) - 1)
B_morn1 = np.zeros(len(SZA) - 1)
S_morn1 = np.zeros(len(SZA) - 1)

B_even2 = np.zeros(len(SZA) - 1)
S_even2 = np.zeros(len(SZA) - 1)
B_morn2 = np.zeros(len(SZA) - 1)
S_morn2 = np.zeros(len(SZA) - 1)

B_total = np.zeros(len(SZA) - 1)
S_total = np.zeros(len(SZA) - 1)

if location == 'OP':  
    B_Rayleigh = np.zeros(len(SZA) - 1)
    S_Rayleigh = np.zeros(len(SZA) - 1)

for i in range(len(B_even1)):
    B_even1[i] = np.mean(bg_even1[(sza_even1 >= SZA[i]) & (sza_even1 < SZA[i+1])])
    S_even1[i] =  np.std(bg_even1[(sza_even1 >= SZA[i]) & (sza_even1 < SZA[i+1])])
    B_morn1[i] = np.mean(bg_morn1[(sza_morn1 >= SZA[i]) & (sza_morn1 < SZA[i+1])])
    S_morn1[i] =  np.std(bg_morn1[(sza_morn1 >= SZA[i]) & (sza_morn1 < SZA[i+1])])
    
    B_even2[i] = np.mean(bg_even2[(sza_even2 >= SZA[i]) & (sza_even2 < SZA[i+1])])
    S_even2[i] = np.std(bg_even2[(sza_even2 >= SZA[i]) & (sza_even2 < SZA[i+1])])
    B_morn2[i] = np.mean(bg_morn2[(sza_morn2 >= SZA[i]) & (sza_morn2 < SZA[i+1])])
    S_morn2[i] = np.std(bg_morn2[(sza_morn2 >= SZA[i]) & (sza_morn2 < SZA[i+1])])
    
    B_total[i] = np.mean(bg_total[(sza_total >= SZA[i]) & (sza_total < SZA[i+1])])
    S_total[i] = np.std(bg_total[(sza_total >= SZA[i]) & (sza_total < SZA[i+1])])
    
    if location == 'OP':  
        B_Rayleigh[i] = 0.5*(np.mean(bg_even_Rayleigh[(sza_even_Rayleigh >= SZA[i]) & (sza_even_Rayleigh < SZA[i+1])])
                         + np.mean(bg_morn_Rayleigh[(sza_morn_Rayleigh >= SZA[i]) & (sza_morn_Rayleigh < SZA[i+1])]))
        S_Rayleigh[i] = 0.5*(np.std(bg_even_Rayleigh[(sza_even_Rayleigh >= SZA[i]) & (sza_even_Rayleigh < SZA[i+1])])
                         + np.std(bg_morn_Rayleigh[(sza_morn_Rayleigh >= SZA[i]) & (sza_morn_Rayleigh < SZA[i+1])]))
    
if location == 'OP':  
    x0 = 1
    res = opt.minimize(scale, x0, method='Nelder-Mead',
                   args=(B_Rayleigh[-2:], B_total[-2:], S_total[-2:]))

if location == 'OP':  
    figure_data = np.vstack((89.5 - SZA[:-1], B_even1, S_even1, B_morn1, S_morn1,
                         B_even2, S_even2, B_morn2, S_morn2, B_Rayleigh*res.x,
                         S_Rayleigh*res.x)).T
    fn = path + 'paper_figure_data/passive_measurement.txt'
    
if location == 'conj':                          
     figure_data = np.vstack((89.5 - SZA[:-1], B_even1, S_even1, B_morn1, S_morn1,
                         B_even2, S_even2, B_morn2, S_morn2)).T
     fn = path + 'paper_figure_data/passive_measurement_conj.txt'

np.savetxt(fn, figure_data, delimiter=',')