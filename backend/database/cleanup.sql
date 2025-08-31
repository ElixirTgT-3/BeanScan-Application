-- Cleanup Script - Drop everything before running new schema
-- Run this FIRST in Supabase SQL Editor

-- Drop all policies
DROP POLICY IF EXISTS "Public read access" ON "User";
DROP POLICY IF EXISTS "Public read access" ON "Bean_Type";
DROP POLICY IF EXISTS "Users manage own data" ON "Bean_Image";
DROP POLICY IF EXISTS "Users manage own data" ON "Defect";
DROP POLICY IF EXISTS "Users manage own data" ON "Shelf_Life";
DROP POLICY IF EXISTS "Users manage own data" ON "History";

-- Drop all triggers
DROP TRIGGER IF EXISTS update_user_updated_at ON "User";
DROP TRIGGER IF EXISTS update_history_updated_at ON "History";

-- Drop all functions
DROP FUNCTION IF EXISTS update_updated_at_column();
DROP FUNCTION IF EXISTS get_user_scan_stats(INTEGER);

-- Drop all views
DROP VIEW IF EXISTS scan_summary;

-- Drop all tables (in reverse dependency order)
DROP TABLE IF EXISTS "History" CASCADE;
DROP TABLE IF EXISTS "Shelf_Life" CASCADE;
DROP TABLE IF EXISTS "Defect" CASCADE;
DROP TABLE IF EXISTS "Bean_Image" CASCADE;
DROP TABLE IF EXISTS "Bean_Type" CASCADE;
DROP TABLE IF EXISTS "User" CASCADE;

-- Drop all indexes (they'll be dropped with tables, but just in case)
-- Drop extensions
DROP EXTENSION IF EXISTS "uuid-ossp";

-- Reset sequences (they'll be recreated)
-- This ensures a clean slate
