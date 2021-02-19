from sqlalchemy import Column, Integer, DateTime, Boolean, Float, String, ForeignKey
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship

from kidsdata.database.constants import PATH_MAX, NAME_MAX
from kidsdata.database.models.scan_x_product import scan_x_product


class ProductBase:
    __tablename__ = "product"
    id = Column(Integer, primary_key=True)
    start = Column(DateTime)
    end = Column(DateTime)
    valid = Column(Boolean, nullable=False, default=True)

    # string uuid nullable pour l'id du run qui l'a produite

    @declared_attr
    def scans(self):
        return relationship("Scan", secondary=scan_x_product, back_populates="products")

    type = Column(String(40))

    __mapper_args__ = {"polymorphic_identity": "product", "polymorphic_on": type}


class ShiftsBase:
    __tablename__ = "shifts"

    @declared_attr
    def id(self):
        return Column(Integer, ForeignKey("product.id"), primary_key=True)

    position_shift = Column(Float)
    laser_shift = Column(Float)
    zpd = Column(Float)

    __mapper_args__ = {"polymorphic_identity": "shifts"}


class KidparBase:
    __tablename__ = "kidpar"

    @declared_attr
    def id(self):
        return Column(Integer, ForeignKey("product.id"), primary_key=True)

    file_path = Column(String(PATH_MAX), unique=True, nullable=False)
    name = Column(String(NAME_MAX), nullable=False)

    __mapper_args__ = {"polymorphic_identity": "kidpar"}
