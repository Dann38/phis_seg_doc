from sqlalchemy import MetaData, Table, Column, Integer, LargeBinary, DateTime, func, Float

metadata = MetaData()

image = Table(
    'image', metadata,
    Column('id', Integer(), primary_key=True),
    Column('original_image', LargeBinary(), nullable=True),
    Column('result_image', LargeBinary(),  nullable=True),
    Column('date_create', DateTime(), server_default=func.now()),
    Column('date_update', DateTime(), server_default=func.now(), onupdate=func.now()),
    Column("method", Integer()),
    Column("coef", Float(),)
)
