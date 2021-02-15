from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.ext.hybrid import hybrid_property

from . import Base

NAME_MAX = 255  # /* # chars in a file name */
PATH_MAX = 4096  # /* # chars in a path name including nul */


class Scan:
    id = Column(Integer, primary_key=True)
    file_path = Column(String(PATH_MAX), unique=True, nullable=False)
    name = Column(String(NAME_MAX), nullable=False)
    # cache filename
    # run enum, vient du 2e dossier parent du fichier

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


class AstroBase(Scan):
    """Regular scans"""
    __tablename__ = "astro"
    scan = Column(Integer, nullable=False)
    source = Column(String(128), nullable=False)
    obsmode = Column(String(128), nullable=False)
    # mbfits fichier du téléscope


class ManualBase(Scan):
    """Manual files
    renommer Manual
    """
    __tablename__ = "manual"


class TablebtBase(Scan):
    """Tablebt test file

    """
    __tablename__ = "tablebt"
    scan = Column(Integer, nullable=False)


class Product:
    id = Column(Integer, primary_key=True)
    start = Column(DateTime)
    end = Column(DateTime)
    # foreign key one to many vers astro ou manual ou tablebt
    valid = Column(Boolean, nullable=False, default=True)

    # @declared_attr
    @hybrid_property
    def scans_id(self):
        return self.astro_id or self.manual_id or self.tablebt_id

    # string uuid nullable pour l'id du run qui l'a produite


class Shifts(Product):
    position_shift = Column(Float)
    laser_shift = Column(Float)
    zpd = Column(Float)


class KidparBase(Product):
    __tablename__ = "kidpar"

    file_path = Column(String(PATH_MAX), unique=True, nullable=False)
    name = Column(String(NAME_MAX), nullable=False)


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


class Manual(ManualBase, Base):
    pass


class Astro(AstroBase, Base):
    pass


class Tablebt(TablebtBase, Base):
    pass


class Kidpar(KidparBase, Base):
    pass


class Stats(StatsBase, Base):
    pass


class Param(ParamBase, Base):
    pass

# chemin de base (dir name)
