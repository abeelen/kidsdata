from kidsdata.database.models.param import ParamBase

from kidsdata.database.models.product import KidparBase, ShiftsBase
from kidsdata.database.models.scan import AstroBase, ManualBase, TablebtBase, ScanBase
from kidsdata.database.models.stats import StatsBase
from kidsdata.database.models.product import ProductBase

from kidsdata.database import Base


# simple tables
class Stats(StatsBase, Base):
    pass


class Param(ParamBase, Base):
    pass


# scan table and its joined inheritance tables
class Scan(ScanBase, Base):
    pass


class Manual(ManualBase, ScanBase, Base):
    pass


class Astro(AstroBase, ScanBase, Base):
    pass


class Tablebt(TablebtBase, ScanBase, Base):
    pass


# product table and its joined inheritance table
class Product(ProductBase, Base):
    pass


class Kidpar(KidparBase, ProductBase, Base):
    pass


class Shifts(ShiftsBase, ProductBase, Base):
    pass


# +  chemin de base (dir name)
