from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship

from kidsdata.database.constants import PATH_MAX, NAME_MAX
from kidsdata.database.models.scan_x_product import scan_x_product


class ScanBase:
    __tablename__ = "scan"
    id = Column(Integer, primary_key=True)
    file_path = Column(String(PATH_MAX), unique=True, nullable=False)
    name = Column(String(NAME_MAX), nullable=False)
    # cache filename
    # run enum, vient du 2e dossier parent du fichier

    size = Column(Float)
    date = Column(DateTime, nullable=False)
    ctime = Column(DateTime)
    mtime = Column(DateTime)
    type = Column(String(40))

    comment = Column(Text, default="")

    @declared_attr
    def param_id(self):
        return Column(Integer, ForeignKey("param.id"))

    @declared_attr
    def products(self):
        return relationship("Product", secondary=scan_x_product, back_populates="scans")

    __mapper_args__ = {"polymorphic_identity": "scan", "polymorphic_on": type}
    # voir pour éventuellement un dump/partage de la table param car long à charger
    # ~7000-15000 fichiers


class AstroBase:
    """Regular scans"""

    __tablename__ = "astro"

    @declared_attr
    def id(self):
        return Column(Integer, ForeignKey("scan.id"), primary_key=True)

    scan = Column(Integer, nullable=False)
    source = Column(String(128), nullable=False)
    obsmode = Column(String(128), nullable=False)
    # mbfits fichier du téléscope

    __mapper_args__ = {"polymorphic_identity": "astro"}


class ManualBase:
    """Manual files
    renommer Manual
    """

    __tablename__ = "manual"

    @declared_attr
    def id(self):
        return Column(Integer, ForeignKey("scan.id"), primary_key=True)

    __mapper_args__ = {"polymorphic_identity": "manual"}


class TablebtBase:
    """Tablebt test file

    """

    __tablename__ = "tablebt"

    @declared_attr
    def id(self):
        return Column(Integer, ForeignKey("scan.id"), primary_key=True)

    scan = Column(Integer, nullable=False)

    __mapper_args__ = {"polymorphic_identity": "tablebt"}
