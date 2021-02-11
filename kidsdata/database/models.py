from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declared_attr

from . import Base

NAME_MAX = 255  # /* # chars in a file name */
PATH_MAX = 4096  # /* # chars in a path name including nul */


class Kids:
    id = Column(Integer, primary_key=True)
    file_path = Column(String(PATH_MAX), unique=True, nullable=False)
    name = Column(String(NAME_MAX), nullable=False)
    # cache filename
    # instrument name

    size = Column(Float)
    date = Column(DateTime, nullable=False)
    ctime = Column(DateTime)
    mtime = Column(DateTime)

    comment = Column(Text, default="")

    @declared_attr
    def param_id(cls):
        return Column(Integer, ForeignKey('param.id'))

    # voir pour éventuellement un dump/partage de la table param car long à charger
    # ~7000-15000 fichiers


class ScanBase(Kids):
    """Regular scans"""
    __tablename__ = "scan"
    scan = Column(Integer, nullable=False)
    source = Column(String(128), nullable=False)
    obsmode = Column(String(128), nullable=False)
    # mbfits fichier du téléscope


class ExtraBase(Kids):
    """Extra files"""
    __tablename__ = "extra"


class TableBase(Kids):
    """Table test file"""
    __tablename__ = "table"
    scan = Column(Integer, nullable=False)


class KidparBase:
    __tablename__ = "kidpar"

    id = Column(Integer, primary_key=True)
    file_path = Column(String(PATH_MAX), unique=True, nullable=False)
    name = Column(String(NAME_MAX), nullable=False)
    start = Column(DateTime)
    end = Column(DateTime)
    # string uuid nullable pour l'id du run qui l'a produite


class ParamBase:
    __tablename__ = "param"

    id = Column(Integer, primary_key=True)
    parameters = Column(Text, nullable=False)
    param_hash = Column(String(64), nullable=False, unique=True)
    nomexp = Column(String(200))
    acqfreq = Column(Float)
    div_kid = Column(Integer)
    raw_data_detector = Column(Text)
    data_sc = Column(Text)
    data_sd = Column(Text)
    data_uc = Column(Text)
    data_ud = Column(Text)


class StatsBase:
    __tablename__ = "stats"

    id = Column(Integer, primary_key=True)


class Extra(ExtraBase, Base):
    pass


class Scan(ScanBase, Base):
    pass


class Table(TableBase, Base):
    pass


class Kidpar(KidparBase, Base):
    pass


class Stats(StatsBase, Base):
    pass


class Param(ParamBase, Base):
    pass


# chemin de base (dir name)