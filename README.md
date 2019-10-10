Kidsdata
========

[![image](https://img.shields.io/travis/abeelen/kidsdata.svg)](https://travis-ci.org/abeelen/kidsdata)

[![image](https://img.shields.io/pypi/v/kidsdata.svg)](https://pypi.python.org/pypi/kidsdata)

python package for KIDS / Kiss Data.

-   Free software: 3-clause BSD license
-   Documentation: (COMING SOON!) <https://abeelen.github.io/kidsdata>.

Features
--------

-   TODO


## Prerequesite

Clone the repository locally
```bash
git clone https://gitlab.lam.fr/KISS/kidsdata.git
```

You also need to retrieve the `kiss_pointing_model.py` and copy it in the `kidsdata` directory

```bash
cd kidsdata/kidsdata
svn export https://lpsc-secure.in2p3.fr/svn/NIKA/Processing/Labtools/JM/KISS/kiss_pointing_model.py
2to3 -w kiss_pointing_model.py
```

You can then install the package locally, allowing updates
```bash
cd ..
pip3 install -e .
```

Before using the package, you must setup two environment variables :
You must have the `libreadnika` compiled and setup a environement variable `NIKA_LIB_PATH` to point the directory containing the `libreadnikadata.so` file. And to use the database, give the location of KISS data


```bash
export NIKA_LIB_PATH=/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/
export KISS_DATA=/data/KISS/Raw/nika2c-data3/KISS
```

## Developer note

Before contributing, please check [this page](./developer_note.md).


## Basic Usage
There are a few basic function that can be used easily to check data :
```python
from datetime import datetime
from kidsdata import list_scan, list_extra, get_extra, beammap, skydip, check_pointing

list_scan() # Will printout the list of scans found in KISS_DATA
kd, figs = beammap(431) # Will project the data as a beam map
check_pointing(kd=kd) # Will plot the pointing of the scan and the source reusing the previously read KissRawData object

# To reduce skydips :
list_extra() # Will printout all the additionnal scans used for skydips in KISS_DATA
# select the scans

scans = get_extra(start=datetime(2019, 5, 1, 19, 14, 24),
                  end=datetime(2019, 5, 1, 19, 52, 53))
print(scans)
skydip(scans)
```
## Advanced usage


An full example can be found as a `jupyter notebook` in the `notebooks` repository.

```bash
cd notebooks
ipython3 notebook
```

Or in the `examples` directory. Alternatively, you can have a quick look beam map analysis with :

```python
import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS

from kidsdata.kiss_data import KissRawData
from kidsdata.db import list_scan, get_scan

plt.ion()

# Open the scan 431
kd = KissRawData(get_scan(431))

# Read All the valid data from array B
list_data = " ".join(kd.names.ComputedDataSc + kd.names.ComputedDataUc + ["I", "Q"])
kd.read_data(list_data=list_data, list_detector=kd.get_list_detector("B", flag=0), silent=True)

# Compute and plot the beam map
# change
fig, res = kd.plot_beammap(coord="pdiff")
beammap, wcs, popts = res

# Update the kidpar
for key in ["x0", "y0"]:
    popts[key] -= np.nanmedian(popts[key])
kd._extended_kidpar = popts

# Plot geometry
geometry, fwhm = kd.plot_kidpar()

# select good detector, ie within 60 arcmin of the center and fwhm 25 +- 10
kidpar = kd.kidpar.loc[kd.list_detector]
pos = np.array([kidpar["x0"], kidpar["y0"]]) * 60  # arcmin
fwhm = np.array(np.abs(kidpar["fwhm_x"]) + np.abs(kidpar["fwhm_y"])) / 2 * 60
ikid = np.where((np.sqrt(pos[0] ** 2 + pos[1] ** 2) < 60) & (np.abs(fwhm - 25) < 10))[0]

data, weight, hits = kd.continuum_map(coord="pdiff", ikid=ikid, cdelt=0.05)

plt.subplot(projection=WCS(data.header))
plt.imshow(data.data, origin='lower')

```
