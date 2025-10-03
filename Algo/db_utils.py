import mysql.connector
import bcrypt

def get_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",           # change to your MySQL user
        password="",   # change to your MySQL password
        database="insurance_ai"
    )


def check_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Check admin credentials
def check_admin(email, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM admins WHERE email=%s", (email,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return check_password(password, result['hashed_password'])
    return False

# Check user credentials
def check_user(email, password):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM accounts WHERE email=%s", (email,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return check_password(password, result['hashed_password'])
    return False

def fetch_data(query: str):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)  # Use dictionary=True for dict results
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results  # This will be a list of dictionaries

def hash_password(password: str):
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')
    

def register(name, email, age, sex, password,role):
    if role == "Admin":
        if admin_exists(email):
            return False, "Email already exists!"
        hashed_password = hash_password(password)
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO admins (name, email, age, sex, hashed_password) VALUES (%s, %s, %s, %s, %s)",
            (name, email, age, sex, hashed_password)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True, "Admin registered successfully!"
    elif role == "User":
        if user_exists(email):
            return False, "Email already exists!"
        hashed_password = hash_password(password)
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            f"INSERT INTO accounts (name, email, age, sex, hashed_password) VALUES (%s, %s, %s, %s, %s)",
            (name, email, age, sex, hashed_password)
        )
        conn.commit()
        cursor.close()
        conn.close()
        return True, "User registered successfully!"


def user_exists(email: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM accounts WHERE email = %s", (email,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result is not None

def admin_exists(email: str) -> bool:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM admins WHERE email = %s", (email,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result is not None

