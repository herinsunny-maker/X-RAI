import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from backend.path_utils import get_db_path

DB_PATH = get_db_path()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
  
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
  
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        gender TEXT NOT NULL,
        phone TEXT,
        height REAL, -- in cm
        weight REAL, -- in kg
        created_by INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (created_by) REFERENCES users (id)
    )
    ''')
    
   
    try:
        cursor.execute("ALTER TABLE patients ADD COLUMN height REAL")
        cursor.execute("ALTER TABLE patients ADD COLUMN weight REAL")
    except sqlite3.OperationalError:
        pass 
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER NOT NULL,
        xray_path TEXT NOT NULL,
        heatmap_path TEXT,
        overlay_path TEXT,
        ai_grade INTEGER,
        ai_confidence REAL,
        final_grade INTEGER,
        staff_notes TEXT,
        doctor_notes TEXT,
        status TEXT DEFAULT 'pending', -- pending, reviewed
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (patient_id) REFERENCES patients (id)
    )
    ''')
    

    demo_users = [
        ("Demo Staff", "staff", "staff", "staff"),
        ("Demo Doctor", "doctor", "doctor", "doctor"),
        ("Admin User", "admin@demo.com", "admin123", "admin")
    ]
    
    for name, email, password, role in demo_users:
        try:
            cursor.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?, ?, ?, ?)",
                (name, email, generate_password_hash(password), role)
            )
        except sqlite3.IntegrityError:
            pass
            
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
