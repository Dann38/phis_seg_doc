from sqlalchemy import create_engine
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from .models import metadata, image as image_table
from sqlalchemy import insert, select, update
from typing import Tuple

NAME = "postgres"
PASSWORD = "kopylov"
HOST = "localhost"
NAME_DB = "image"


class ManagerDB:
    def __init__(self):
        self.engine = None

    def first_start(self):
        self.create_db()
        self.open_db()
        self.delete_table()
        self.create_table()

    def create_db(self):
        try:
            # Устанавливаем соединение с postgres
            connection = psycopg2.connect(user=NAME, password=PASSWORD)
            connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

            # Создаем курсор для выполнения операций с базой данных
            cursor = connection.cursor()
            # Создаем базу данных
            cursor.execute(f'create database {NAME_DB}')
            # Закрываем соединение
            cursor.close()
            connection.close()
        except psycopg2.errors.DuplicateDatabase:
            print("Уже существует")

    def create_table(self):
        metadata.create_all(self.engine)

    def delete_table(self):
        metadata.drop_all(self.engine)

    def open_db(self):
        self.engine = create_engine(f"postgresql+psycopg2://{NAME}:{PASSWORD}@{HOST}/{NAME_DB}")

    def add_origin_image(self, origin_image) -> int:
        ins = insert(image_table).values(
            original_image=origin_image,
        )
        conn = self.engine.connect()
        r = conn.execute(ins)
        conn.commit()
        return r.inserted_primary_key[0]

    def add_result_image(self, id_image:int, result_image:bytes):
        upd = update(image_table).where(
            image_table.c.id == id_image
        ).values(
            result_image=result_image
        )
        conn = self.engine.connect()
        r = conn.execute(upd)
        conn.commit()

    def get_row_image_id(self, id_image: int) -> Tuple[bytes, bytes]:
        conn = self.engine.connect()
        s = select(image_table).where(
              image_table.c.id == id_image
        )
        r = conn.execute(s).first()
        return r[1], r[2]

    def get_result_image_id(self, id_image: int) -> bytes:
        conn = self.engine.connect()
        s = select(image_table).where(
            image_table.c.id == id_image
        )
        r = conn.execute(s)
        return r.first()[2]

    def get_id_10_last(self) -> list[int]:
        conn = self.engine.connect()
        s = select(image_table.c.id).order_by(image_table.c.date_update.desc()).limit(10)
        r = conn.execute(s)
        list_result = [id_[0] for id_ in r.all()]
        return list_result


if __name__ == '__main__':
    db_manager = ManagerDB()
    db_manager.open_db()
    # db_manager.first_start()
    # db_manager.delete_table()
    # db_manager.create_table()


