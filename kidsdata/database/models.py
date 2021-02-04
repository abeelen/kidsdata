from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from . import Base

NAME_MAX = 255  # /* # chars in a file name */
PATH_MAX = 4096  # /* # chars in a path name including nul */


class Kids:
    id = Column(Integer, primary_key=True)
    filename = Column(String(PATH_MAX), unique=True, nullable=False)
    name = Column(String(NAME_MAX), nullable=False)

    size = Column(Float)
    date = Column(DateTime, nullable=False)
    ctime = Column(DateTime)
    mtime = Column(DateTime)

    comment = Column(Text, default="")

    # foreign key vers une row de param

    # La table param, assez long à charger car il faut lire le header
    # nomexp
    # acqfreq
    # div_kid
    # RawDataDetector
    # datasc etc. en string

    # voir pour éventuellement un dump/partage de la table param car long à charger

    # ~7000-15000 fichiers


class ScanBase(Kids):
    """Regular scans"""
    __tablename__ = "scan"
    scan = Column(Integer, nullable=False)
    source = Column(String(128), nullable=False)
    obsmode = Column(String(128), nullable=False)


class ExtraBase(Kids):
    """Extra files"""
    __tablename__ = "extra"


class InlabBase(Kids):
    """InLab test file"""
    __tablename__ = "inlab"
    scan = Column(Integer, nullable=False)


class KidparBase:
    __tablename__ = "kidpar"

    id = Column(Integer, primary_key=True)
    filename = Column(String(PATH_MAX), unique=True, nullable=False)
    name = Column(String(NAME_MAX), nullable=False)
    start = Column(DateTime)
    end = Column(DateTime)


class ParamBase:
    __tablename__ = "param"


class StatsBase:
    __tablename__ = "stats"


class Extra(ExtraBase, Base):
    pass


class Scan(ScanBase, Base):
    pass


class Inlab(InlabBase, Base):
    pass


class Kidpar(KidparBase, Base):
    pass
