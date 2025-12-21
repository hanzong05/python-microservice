#!/usr/bin/env python3
"""
Test script for Todo API
Run this before deploying to Render to make sure everything works!
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    print("Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print("✅ Health check passed!\n")

def test_root():
    print("Testing / endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Root endpoint passed!\n")

def test_create_todo():
    print("Testing POST /todos...")
    data = {"task": "Test todo from script"}
    response = requests.post(f"{BASE_URL}/todos", json=data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    print("✅ Create todo passed!\n")
    return result.get("created", {}).get("id")

def test_get_todos():
    print("Testing GET /todos...")
    response = requests.get(f"{BASE_URL}/todos")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Get todos passed!\n")

def test_delete_todo(todo_id):
    print(f"Testing DELETE /todos/{todo_id}...")
    response = requests.delete(f"{BASE_URL}/todos/{todo_id}")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print("✅ Delete todo passed!\n")

if __name__ == "__main__":
    print("=" * 50)
    print("🧪 Testing Todo API Locally")
    print("=" * 50)
    print("\nMake sure the API is running: python main.py\n")
    
    try:
        test_health()
        test_root()
        todo_id = test_create_todo()
        test_get_todos()
        if todo_id:
            test_delete_todo(todo_id)
        test_get_todos()
        
        print("=" * 50)
        print("✅ All tests passed!")
        print("=" * 50)
        print("\n🚀 Ready to deploy to Render!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to API")
        print("Make sure the API is running:")
        print("  cd python-backend")
        print("  python main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")