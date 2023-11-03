from db_manager import ManagerDB

if __name__ == '__main__':
    db_manager = ManagerDB()
    # db_manager.open_db()
    db_manager.first_start()
    db_manager.delete_table()
    db_manager.create_table()