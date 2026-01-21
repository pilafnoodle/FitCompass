from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

app = Flask(__name__)
app.secret_key = "fitcompass_secret_key"

# -------------------------
# Database setup
# -------------------------
currentDirectory = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(currentDirectory, "UserLogins.db")

def get_db_connection():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

# Create tables
connection = get_db_connection()
cursor = connection.cursor()

# Drop old table if it exists (WARNING: deletes old user data!) Only do when adding columns to the table and want total reset
cursor.execute("DROP TABLE IF EXISTS UserLogins")

cursor.execute("""
CREATE TABLE IF NOT EXISTS UserLogins(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    goal TEXT,
    goal_other TEXT,
    workouts_per_week INTEGER,
    body_part TEXT
)
""")

connection.commit()
connection.close()

# -------------------------
# Login
# -------------------------
@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        raw_password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM UserLogins WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user["password"], raw_password):
            session['user_id'] = user["id"]
            session['username'] = username
            return redirect(url_for('home'))

        flash("Invalid username or password")
        return redirect(url_for('login'))

    return render_template('login.html')

# -------------------------
# Register + Intake Quiz
# -------------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])

        goal = request.form.get('goal')
        goal_other = request.form.get('goal_other') if goal == 'other' else None
        workouts_per_week = request.form.get('workouts_per_week')
        body_part = request.form.get('body_part')

        try:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Insert user
            cursor.execute(
                """
                INSERT INTO UserLogins (username, email, password, goal, goal_other, workouts_per_week, body_part)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (username, email, password, goal, goal_other, workouts_per_week, body_part)
            )
            user_id = cursor.lastrowid

            conn.commit()
            conn.close()

            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            conn.close()
            flash("Username or email already exists")
            return redirect(url_for('register'))

    return render_template('register.html')

# -------------------------
# Home
# -------------------------
@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))

    return render_template(
        'home.html',
        username=session['username'],
        points=120,
        goal_percent=62
    )

# -------------------------
# Logout
# -------------------------
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# -------------------------
# Placeholder
# -------------------------
@app.route('/profile')
def profile():
    return "Profile page coming soon"

@app.route('/history')
def history():
    return "History page coming soon"

@app.route('/library')
def library():
    return "Library page coming soon"

@app.route('/shop')
def shop():
    return "Shop page coming soon"

@app.route('/settings')
def settings():
    return "Settings page coming soon"


if __name__ == "__main__":
    app.run(debug=True)

