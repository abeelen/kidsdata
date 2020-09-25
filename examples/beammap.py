import numpy as np
import matplotlib.pyplot as plt

from astropy.wcs import WCS

from kidsdata import KissData
from kidsdata.db import list_scan, get_scan

plt.ion()

# Open the scan 431
kd = KissData(get_scan(431))

# Read All the valid data from array B
list_data = kd.names.DataSc + kd.names.DataUc + ["I", "Q"]
kd.read_data(list_data=list_data, list_detector=kd.get_list_detector("B", flag=0, typedet=1), silent=True)

# Compute and plot the beam map
beammap, (datas, wcs, popts) = kd.plot_beammap(
    coord="pdiff", flatfield=None, cm_func="kidsdata.common_mode.pca_filtering", ncomp=2
)

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
plt.imshow(data.data, origin="lower")
