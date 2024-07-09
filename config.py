from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Access the environment variables
USERNAME = os.getenv('STREAMLIT_USERNAME')
PASSWORD = os.getenv('STREAMLIT_PASSWORD')