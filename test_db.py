
from db_config import get_db_connection

try:
    connection = get_db_connection()

    if connection.is_connected():
        print("MySQL connected successfully!")

    connection.close()

except Exception as error:
    print("Database connection failed:", error)
