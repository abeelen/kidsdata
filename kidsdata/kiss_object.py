import os
import logging
from pathlib import Path
import astropy.units as u
from astropy.coordinates import get_body, solar_system_ephemeris, EarthLocation
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.coordinates.name_resolve import NameResolveError

NIKA_LIB_PATH = os.getenv("NIKA_LIB_PATH", "/data/KISS/NIKA_lib_AB_OB_gui/Readdata/C/")
KISS_CAT_FILENAME = (
    Path(NIKA_LIB_PATH).parent.parent
    / "Acquisition"
    / "instrument"
    / "kiss_telescope"
    / "library"
    / "KISS_Source_Position"
    / "CatalogKissJuan.sou"
)

if KISS_CAT_FILENAME.exists():
    KISS_CAT = Table.read(
        KISS_CAT_FILENAME,
        format="ascii",
        names=["name", "dummy1", "dummy2", "RA", "DEC"],
        exclude_names=["dummy1", "dummy2"],
    )
else:
    logging.warning("KISS catalog not found, please check {}".format(KISS_CAT_FILENAME))
    KISS_CAT = None


# Add used observatories
EarthLocation._get_site_registry()

# Alessandro Fasano Private Comm
EarthLocation._site_registry.add_site(
    ["Quijote", "KISS"], EarthLocation(lat=0.493931966 * u.rad, lon=-0.288155867 * u.rad, height=2395 * u.m)
)
# JMP code
EarthLocation._site_registry.add_site(
    ["Teide", "Tenerife"], EarthLocation(lat=28.7569444444 * u.deg, lon=-17.8925 * u.deg, height=2390 * u.m)
)
EarthLocation._site_registry.add_site(
    ["IRAM30m", "30m", "NIKA", "NIKA2"],
    EarthLocation(lat=37.066111111111105 * u.deg, lon=-3.392777777777778 * u.deg, height=2850 * u.m),
)


def get_coords(source, time=None):
    """Get coordinates of objects seen by KISS"""

    if source.lower() in solar_system_ephemeris.bodies:
        coords = get_body(source.lower(), time)
    elif KISS_CAT is not None and source in KISS_CAT["name"]:
        mask = source == KISS_CAT["name"]
        coords = SkyCoord(KISS_CAT[mask]["RA"], KISS_CAT[mask]["DEC"], frame="fk5", unit=(u.hourangle, u.deg))
    else:
        try:
            coords = SkyCoord.from_name(source.lower())
        except NameResolveError:
            raise KeyError("{} is not in astropy ephemeris, KISS_CAT or SkyCoord database".format(source))

    return coords
