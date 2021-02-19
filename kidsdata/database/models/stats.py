from sqlalchemy import Column, Integer


class StatsBase:
    __tablename__ = "stats"

    id = Column(Integer, primary_key=True)
