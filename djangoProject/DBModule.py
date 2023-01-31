import sqlite3


class DBInterface:
    def __init__(self):
        con = sqlite3.connect('resources/database.db')
        cur = con.cursor()
        command_string = f"""CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                password TEXT NOT NULL,
                UNIQUE(username)
            );
            """

        cur.execute(command_string)
        con.commit()
        con.close()
    

    def login(self, username, password):

        con = sqlite3.connect('resources/database.db')
        cur = con.cursor()

        cur.execute(f'SELECT * FROM users WHERE users.username = "{username}" AND users.password = "{password}";')
        user = cur.fetchone()

        if user is not None:
            return True
        else:
            return False
    

    def add_user(self, username, password):

        con = sqlite3.connect('resources/database.db')
        cur = con.cursor()

        command_string = f'INSERT OR IGNORE INTO users(username, password) VALUES("{username}","{password}");'
        
        cur.execute(command_string)
        con.commit()
        con.close()

