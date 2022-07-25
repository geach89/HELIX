# HELIX Analysis Code

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Results](#results)
- [License](./LICENSE)
- [Issues](https://github.com/ebridge2/lol/issues)
- [Citation](#citation)

# Overview

This repo contains python code written to perform the data analysis associated with the HELIX instrument to derive metastable helium density profiles and assess the magnitude of the passive metastable helium airglow contribution.

# Repo Contents

- active.py
- passive.py
- active_photon_profile.py
- Figure_data.xlsx

# System Requirements

## Hardware Requirements

Any modern computer should be capable of running the scripts contained here.

## Software Requirements

### OS Requirements

The code has only been tested on a Windows computer, but should be compatible with any Linux, Windows or Mac operating systems.

### Python 3

The code is written in Python and requires Python 3.7.4 to run (it may be compatible with other versions of Python but this has not been verified).

#### Package dependencies 

The code relies on the standard suite of Python packages, with the addition of netCDF4, which is available for download [here](https://pypi.org/project/netCDF4/#files)

# Installation instructions

Given a functioning Python environment and the required packages, no installation is required. The necesary data files must simply be downloaded, and the corresponding path must be designated in each program (on line 9 of each program). The liadr data is available [here](https://figshare.com/s/b68943b88521b1ce696d), and the MSIS2.0 profile that is used for calibration of the return signal is contained within this repo (msis20output.txt).

# Demo

Given the simplicity of the analysis being performed here, and the relatively small size of the dataset, no dedicated demo dataset has been created - instead, the code can be run on the full dataset, to produce the results presented in the associated manuscript. The data is available [here](https://figshare.com/s/b68943b88521b1ce696d) and the resulting data is provided for comparison in this repository (Figure_Data.xlsx).

Each program takes between 5 and 10 minutes to run.

# Results

The scripts should produce the data in 'Figure_data.xslx' when run on the full data set, as follows:
- active.py generates the data in the "Figure 3" sheet. The first 9 columns of data are produced when time_period='Jan-Feb' is selected, the other 8 columns are produced when time_period='Feb-Mar' is used.
- passive.py generates the data in the "Figure 4" sheets. The data in "Figure 4 (left)" is produced when location='OP' is selected, and the data in "Figure 4 (right)" is produced when location='conj' is selected.
- active_photon_profile.py generates the data in the Figure 6 sheets. No options need be selected.
