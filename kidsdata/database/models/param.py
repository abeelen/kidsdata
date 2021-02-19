from sqlalchemy import Column, Integer, Text, String, Float


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
