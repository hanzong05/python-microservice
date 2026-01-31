"""
Supabase Client Configuration
Reusable connection module for all project files
"""

import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env in the project root
# Handle both direct execution and subprocess execution
current_file = Path(__file__).resolve()
env_path = current_file.parent / '.env'
load_dotenv(env_path)

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    print("X Supabase library not installed!")
    print("Install with: pip install supabase")
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None


def get_supabase_client():
    """
    Get authenticated Supabase client instance

    Returns:
        Client: Authenticated Supabase client or None if connection fails
    """
    if not SUPABASE_AVAILABLE:
        print("X Supabase library not available")
        return None

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not SUPABASE_URL or not SUPABASE_KEY:
        print("X Missing environment variables: SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")
        return None

    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        # Test connection
        client.table('municipalities').select('id').limit(1).execute()
        return client
    except Exception as e:
        print(f"X Failed to connect to Supabase: {e}")
        return None


def test_connection():
    """
    Test Supabase connection

    Returns:
        bool: True if connection successful, False otherwise
    """
    print("Testing Supabase connection...")
    client = get_supabase_client()

    if client:
        print("✓ Successfully connected to Supabase!")
        return True
    else:
        print("X Connection failed")
        return False


if __name__ == "__main__":
    # Test the connection when run directly
    test_connection()
