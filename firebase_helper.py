import os
from datetime import datetime

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, db

# Load environment variables
load_dotenv()

# Firebase initialization
cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
db_url = os.getenv("FIREBASE_DB_URL")

if not cred_path or not os.path.exists(cred_path):
    raise FileNotFoundError(f"Firebase credentials not found at: {cred_path}")

if not db_url:
    raise ValueError("FIREBASE_DB_URL is not set in the .env file.")

print(f"Credential Path: {cred_path}")
print(f"Database URL: {db_url}")

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred, {"databaseURL": db_url})


# Firebase helper function to upload results
def upload_result(result, predicted_reason=None):
    ref = db.reference()

    # Get the current date and time
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S")

    # Update the database with status and predicted reason
    updates = {
        "baby_cry_status/status": "cry" if result == "Baby is crying" else "no cry",
        "babyCryingReason": predicted_reason if predicted_reason else "unknown",
        "Date": [current_date],
        "Time": [current_time],
        "Predictions": predicted_reason if predicted_reason else "unknown",
    }
    ref.update(updates)
    print("Update successful")

    # Update crying reasons and count
    if result == "Baby is crying" and predicted_reason:
        # Increment count for this cry type in Firebase
        cry_count_ref = ref.child("cryingReasonsCount")
        current_counts = cry_count_ref.get() or {}
        new_count = current_counts.get(predicted_reason, 0) + 1

        # Update the counts in Firebase
        cry_count_ref.update({predicted_reason: new_count})
        print(f"Updated Firebase count for '{predicted_reason}': {new_count}")
