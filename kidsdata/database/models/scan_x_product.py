from sqlalchemy import Table, Column, Integer, ForeignKey

from kidsdata.database import Base

scan_x_product = Table(
    "scan_x_product",
    Base.metadata,
    Column("scan_id", Integer, ForeignKey("scan.id"), primary_key=True),
    Column("product_id", Integer, ForeignKey("product.id"), primary_key=True),
)
