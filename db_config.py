import mysql.connector

def get_db_connection():
    connection = mysql.connector.connect(
        host="localhost",
        user="land_user",
        password="LandProject@2026",
        database="land_acquisition_db",
        port=3306
    )

    return connection