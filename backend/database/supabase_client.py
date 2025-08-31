import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

def get_supabase_client() -> Client:
    """Get Supabase client instance"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    
    return create_client(supabase_url, supabase_key)

def get_service_role_client() -> Client:
    """Get Supabase client with service role key for admin operations"""
    supabase_url = os.getenv("SUPABASE_URL")
    service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not service_role_key:
        raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
    
    return create_client(supabase_url, service_role_key)

# Database table schemas - Updated to match ERD design
USER_TABLE = "User"
BEAN_TYPE_TABLE = "Bean_Type"
BEAN_IMAGE_TABLE = "Bean_Image"
DEFECT_TABLE = "Defect"
SHELF_LIFE_TABLE = "Shelf_Life"
HISTORY_TABLE = "History"

# Initialize client
supabase = get_supabase_client()
