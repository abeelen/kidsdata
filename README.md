# KissData

New python package for KIDS / Kiss Data.

Before using the package, you must setup two environment variables

```bash
export NIKA_LIB_PATH=/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/
export KISS_DATA=/data/KISS/Raw/nika2c-data3/KISS
```

An full example can be found as a `jupyter notebook` in the `notebooks` repository.

```bash
cd notebooks
ipython3 notebook
```

Alternatively, you can have a quick look beam map analysis with :

```python
import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS

from src.kiss_data import KissRawData
from src.db import list_scan, get_scan

plt.ion()

# Open the scan 431
kd = KissRawData(get_scan(431))

# Read All the valid data from array B
list_data = " ".join(kd.names.ComputedDataSc + kd.names.ComputedDataUc + ["I", "Q"])
kd.read_data(list_data=list_data, list_detector=kd.get_list_detector("B", flag=0), silent=True)

# Compute and plot the beam map
beammap, datas, wcs, popts = kd.plot_beammap(coord="pdiff")

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