import os

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, db

from firebase_helper import upload_result

# Load environment variables
load_dotenv()

# Verify that environment variables are loaded correctly
print(f"FIREBASE_CREDENTIALS_PATH: {os.getenv('FIREBASE_CREDENTIALS_PATH')}")
print(f"FIREBASE_DB_URL: {os.getenv('FIREBASE_DB_URL')}")

# Test case 1: Baby is crying
print("\nTesting: Baby is crying")
try:
    upload_result("Baby is crying", "Hunger")
    print("Test case 1 successful")
except Exception as e:
    print(f"Error in test case 1: {str(e)}")

# Test case 2: Baby is not crying
print("\nTesting: Baby is not crying")
try:
    upload_result("Baby is crying", "tired")
    print("Test case 2 successful")
except Exception as e:
    print(f"Error in test case 2: {str(e)}")

print("\nTest completed. Check your Firebase database to verify the results.")