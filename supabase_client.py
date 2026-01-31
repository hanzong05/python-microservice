"""
Supabase Client Configuration
Reusable connection module for all project files
Works with both .env files (local) and environment variables (production)
"""

import os
from pathlib import Path

# Try to load dotenv, but don't fail if it's not available (production)
try:
    from dotenv import load_dotenv
    
    # Load environment variables from .env in the project root
    current_file = Path(__file__).resolve()
    env_path = current_file.parent / '.env'
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
    else:
        print(f"ℹ No .env file found, using system environment variables")
except ImportError:
    print("ℹ python-dotenv not installed, using system environment variables")

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
    print("✓ Supabase library imported successfully")
except ImportError as e:
    print(f"✗ Supabase library import failed: {e}")
    SUPABASE_AVAILABLE = False
    create_client = None
    Client = None


def get_supabase_client():
    """
    Get authenticated Supabase client instance

    Returns:
        Client: Authenticated Supabase client or None if connection fails
    """
    print("\n=== Attempting Supabase Connection ===")
    
    if not SUPABASE_AVAILABLE:
        print("✗ Supabase library not available")
        return None

    # Check for both possible key names
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")

    # Debug output
    print(f"URL present: {bool(SUPABASE_URL)}")
    print(f"URL value: {SUPABASE_URL[:50]}..." if SUPABASE_URL else "URL: NOT SET")
    print(f"Key present: {bool(SUPABASE_KEY)}")
    print(f"Key length: {len(SUPABASE_KEY) if SUPABASE_KEY else 0}")

    if not SUPABASE_URL:
        print("✗ SUPABASE_URL not set")
        return None
        
    if not SUPABASE_KEY:
        print("✗ SUPABASE_SERVICE_ROLE_KEY not set")
        return None

    try:
        print("Creating Supabase client...")
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Supabase client created successfully!")
        
        # Simple connection test - just try to access storage
        # Don't test tables in case RLS is blocking
        try:
            print("Testing storage access...")
            # This is a minimal test that doesn't require table permissions
            client.storage.list_buckets()
            print("✓ Storage accessible!")
        except Exception as storage_error:
            print(f"⚠ Storage test warning: {storage_error}")
            # Still return client - storage might have different permissions
        
        return client
        
    except Exception as e:
        print(f"✗ Failed to create Supabase client: {e}")
        import traceback
        print("Full error:")
        traceback.print_exc()
        return None


def test_connection():
    """
    Test Supabase connection

    Returns:
        bool: True if connection successful, False otherwise
    """
    print("\n=== Testing Supabase Connection ===")
    client = get_supabase_client()

    if client:
        print("✓ Successfully connected to Supabase!")
        return True
    else:
        print("✗ Connection failed")
        return False


if __name__ == "__main__":
    # Test the connection when run directly
    test_connection()