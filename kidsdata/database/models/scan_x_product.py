from sqlalchemy import Column, Integer, ForeignKey

scan_x_product_base = (
    Column("scan_id", Integer, ForeignKey("scan.id"), primary_key=True),
    Column("product_id", Integer, ForeignKey("product.id"), primary_key=True),
)
