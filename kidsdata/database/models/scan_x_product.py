from sqlalchemy import Column, Integer, ForeignKey
from sqlalchemy.ext.declarative.base import declared_attr


class ScanXProductBase:
    __tablename__ = "scan_x_product"

    @declared_attr
    def scan_id(self):
        return Column(Integer, ForeignKey("scan.id"), primary_key=True)

    @declared_attr
    def product_id(self):
        return Column(Integer, ForeignKey("product.id"), primary_key=True)
