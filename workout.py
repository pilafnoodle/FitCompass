import sqlite3
import random

# ---------------- CONNECT TO DATABASE ----------------

db_path = "UserLoginsTest.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# ---------------- CREATE TABLE (MATCHING YOUR DATABASE) ----------------

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    email TEXT,
    password TEXT,
    goal TEXT,
    goal_other TEXT,
    workouts_per_week TEXT,
    body_part TEXT
)
""")

# ---------------- EXERCISE LISTS ----------------

upper_body = ["Push-ups", "V pushups", "Invertaed Rows", "Pull-ups"]
lower_body = ["Squats", "Lunges", "Glute Bridges", "Calf Raises"]
core = ["Sit-ups", "Plank", "Supermans"]
cardio = ["Jumping Jacks", "Jogging in Place", "Running", "Jump Rope", "Burpies"]

push_day = ["Push-ups", "Shoulder Press"]
leg_day = ["Squats", "Lunges", "Calf Raises"]

# ---------------- GENERATE WORKOUT ----------------

new_people_exercises = [
    "Glute Bridges", "Jogging in Place", "Jumping Jacks",
    "Lunges", "Push-ups", "Sit-ups", "Squats", "Supermans"
]

# Rep ranges
rep_ranges = {
    "Beginner": {
        "strength": "2 to 3 sets of 8 to 12 reps",
        "core": "2 to 3 sets of 10 to 15 reps",
        "cardio": "30 to 60 seconds"
    },
}

def pick_random(ex_list, num):
    return random.sample(ex_list, k=min(num, len(ex_list)))

def format_exercise(ex, category):
    text = f"- {ex}\n"
    for level, reps in rep_ranges.items():
        text += f"  - {level}: {reps[category]}\n"
    return text

# ---------------- WORKOUT PLAN FUNCTION ----------------

def generate_workout_plan():

    skill = input(
        "\nChoose Plan:\n"
        "1 - Beginner\n"
        "2 - Strength\n"
        "3 - Weekly Split\n"
        "4 - Exit\n"
        "Enter choice: "
    )

    if skill == "4":
        print("Exiting program.")
        return None

    plan = ""

    # ---------------- BEGINNER ----------------
    if skill == "1":

        plan += "BEGINNER WORKOUT PLAN\n\n"

        for day in ["MON", "WED", "FRI", "SUN"]:
            plan += f"{day}: Full Body\n"

            for ex in pick_random(new_people_exercises, 5):

                category = (
                    "cardio" if ex in cardio
                    else "core" if ex in core
                    else "strength"
                )

                plan += format_exercise(ex, category)

            plan += "\n"

        plan += "TUE: Rest\n\nTHU: Rest\n\nSAT: Rest\n"

    # -------- STRENGTH --------
    elif skill == "2":

        plan += "STRENGTH + ATHLETICISM PLAN\n\n"

        # MON
        plan += "MON: Upper Body + Cardio + Core\n"
        for ex in pick_random(upper_body, 3):
            plan += format_exercise(ex, "strength")

        plan += format_exercise(random.choice(cardio), "cardio")
        plan += format_exercise(random.choice(core), "core")

        # TUE
        plan += "\nTUE: Lower Body + Cardio + Core\n"
        for ex in pick_random(lower_body, 3):
            plan += format_exercise(ex, "strength")

        plan += format_exercise(random.choice(cardio), "cardio")
        plan += format_exercise(random.choice(core), "core")

        # WED
        plan += "\nWED: Rest\n"

        # THU
        plan += "\nTHU: Upper Body\n"
        for ex in pick_random(upper_body, 4):
            plan += format_exercise(ex, "strength")

        # FRI
        plan += "\nFRI: Lower Body\n"
        for ex in pick_random(lower_body, 4):
            plan += format_exercise(ex, "strength")

        # SAT
        plan += "\nSAT: Rest\n"

        # SUN
        plan += "\nSUN: High Cardio\n"
        for ex in pick_random(cardio, 3):
            plan += format_exercise(ex, "cardio")

    # -------- WEEKLY SPLIT --------
    elif skill == "3":

        plan += "WEEKLY SPLIT WORKOUT PLAN\n\n"

        # MON
        plan += "MON: Upper Body + Cardio\n"
        for ex in pick_random(upper_body, 3):
            plan += format_exercise(ex, "strength")
        plan += format_exercise(random.choice(cardio), "cardio")

        # TUE
        plan += "\nTUE: Lower Body + Cardio\n"
        for ex in pick_random(lower_body, 3):
            plan += format_exercise(ex, "strength")
        plan += format_exercise(random.choice(cardio), "cardio")

        # WED
        plan += "\nWED: Rest\n"

        # THU
        plan += "\nTHU: Push Day\n"
        for ex in pick_random(push_day, 3):
            plan += format_exercise(ex, "strength")

        # FRI
        plan += "\nFRI: Leg Day\n"
        for ex in pick_random(leg_day, 3):
            plan += format_exercise(ex, "strength")

        # SAT
        plan += "\nSAT: Upper Body\n"
        for ex in pick_random(upper_body, 4):
            plan += format_exercise(ex, "strength")

        # SUN
        plan += "\nSUN: Rest\n"

    else:
        print("Invalid input.")
        return None

    # Save to file
    with open("random_workout_plan_with_reps.txt", "w", encoding="utf-8") as file:
        file.write(plan)

    print("\nWorkout plan saved successfully.\n")

    return plan

# ---------------- MAIN PROGRAM ----------------

username = input("Enter username: ")
email = input("Enter email: ")
password = input("Enter password: ")
workout_plan = generate_workout_plan()
##goal = input("Enter goal: ")
goal_other = ""
body_part = input("Enter body part focus: ")

# Generate plan


# Stop if user exits
if workout_plan is None:
    conn.close()
    exit()

# ---------------- INSERT INTO DATABASE ----------------

cursor.execute("""
INSERT INTO users (username, email, password, goal, goal_other, workouts_per_week, body_part)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", (
    username,
    email,
    password,
    goal,
    goal_other,
    workout_plan,   # ✅ STORED HERE
    body_part
))

conn.commit()

print("\n✅ Workout Plan Saved Into Database Successfully!")

# Show last inserted user
cursor.execute("SELECT id, username, workouts_per_week FROM users ORDER BY id DESC LIMIT 1")
print("\nLast User Added:\n")
print(cursor.fetchone())

conn.close()
