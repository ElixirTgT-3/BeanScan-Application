#!/usr/bin/env python3
"""
Database Setup Script for BeanScan Backend
This script helps you set up and test your Supabase database connection.
"""

import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Check if required environment variables are set"""
    load_dotenv()
    
    required_vars = [
        "SUPABASE_URL",
        "SUPABASE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease copy env_example.txt to .env and fill in your Supabase credentials.")
        return False
    
    print("✅ Environment variables are configured")
    return True

def test_supabase_connection():
    """Test the Supabase connection"""
    try:
        from database.supabase_client import get_supabase_client
        
        print("🔌 Testing Supabase connection...")
        client = get_supabase_client()
        
        # Test a simple query
        result = client.table("User").select("count", count="exact").execute()
        print("✅ Successfully connected to Supabase!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return False

def create_temp_directory():
    """Create temporary directory for image uploads"""
    try:
        os.makedirs("temp", exist_ok=True)
        print("✅ Created temp directory for image uploads")
        return True
    except Exception as e:
        print(f"❌ Failed to create temp directory: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 BeanScan Database Setup")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Test connection
    if not test_supabase_connection():
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure your Supabase project is running")
        print("2. Verify your API keys are correct")
        print("3. Check if your project has the required tables")
        sys.exit(1)
    
    # Create temp directory
    create_temp_directory()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the database schema: database/schema.sql")
    print("2. Start the API: python main.py")
    print("3. Visit: http://localhost:8000/docs")
    
    print("\n📚 Database Schema:")
    print("- User: User profiles and authentication")
    print("- Bean_Type: Coffee bean classifications")
    print("- Bean_Image: Uploaded bean images")
    print("- Defect: Detected defects in beans")
    print("- Shelf_Life: Shelf life predictions")
    print("- History: Central scan records")

if __name__ == "__main__":
    main()
