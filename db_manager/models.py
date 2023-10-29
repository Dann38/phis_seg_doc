from sqlalchemy import MetaData, Table, Column, Integer, LargeBinary

metadata = MetaData()

image = Table('image', metadata,
    Column('id', Integer(), primary_key=True),
    Column('original_image', LargeBinary(), nullable=True),
    Column('result_image', LargeBinary(),  nullable=True),
)

