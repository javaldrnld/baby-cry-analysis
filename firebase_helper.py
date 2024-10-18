import os
from datetime import datetime

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, db

# Load environment variables
load_dotenv()

# Firebase initialization
cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
db_url = "https://baby-9e20f-default-rtdb.asia-southeast1.firebasedatabase.app/"

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
    print(f"Attempting to update with: {updates}")
    ref.update(updates)
    print("Update successful")

    # Update cryingReasons
    if result == "Baby is crying" and predicted_reason:
        crying_reasons = ref.child("cryingReasons").get() or []
        crying_reasons.append(predicted_reason)
        updates["cryingReasons"] = crying_reasons[-5:]  # Keep only the last 5 reason

    print(f"Attempting to update with: {updates}")
    ref.update(updates)
    print("Update successful")
