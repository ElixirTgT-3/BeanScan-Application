-- BeanScan Database Schema for Supabase (Based on ERD Design)
-- This schema matches the Entity-Relationship Diagram with 6 interconnected tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. User Table
CREATE TABLE IF NOT EXISTS "User" (
    user_id SERIAL PRIMARY KEY,
    "Name" VARCHAR(255) NOT NULL,
    contact_number VARCHAR(20),
    role VARCHAR(50) CHECK (role IN ('admin', 'user', 'analyst')),
    location VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 2. Bean_Type Table
CREATE TABLE IF NOT EXISTS "Bean_Type" (
    bean_type_id SERIAL PRIMARY KEY,
    type_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 3. Bean_Image Table
CREATE TABLE IF NOT EXISTS "Bean_Image" (
    image_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES "User"(user_id) ON DELETE SET NULL,
    bean_type_id INTEGER REFERENCES "Bean_Type"(bean_type_id) ON DELETE SET NULL,
    image_path VARCHAR(500) NOT NULL,
    capture_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    image_url VARCHAR(500),
    file_size INTEGER,
    image_format VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 4. Defect Table
CREATE TABLE IF NOT EXISTS "Defect" (
    defect_id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES "Bean_Image"(image_id) ON DELETE CASCADE,
    defect_type VARCHAR(100) NOT NULL,
    severity_level VARCHAR(20) CHECK (severity_level IN ('low', 'medium', 'high', 'critical')),
    defect_area DECIMAL(10,4),
    defect_percentage DECIMAL(5,2),
    defect_coordinates JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 5. Shelf_Life Table
CREATE TABLE IF NOT EXISTS "Shelf_Life" (
    shelf_life_id SERIAL PRIMARY KEY,
    image_id INTEGER REFERENCES "Bean_Image"(image_id) ON DELETE CASCADE,
    bean_type_id INTEGER REFERENCES "Bean_Type"(bean_type_id) ON DELETE SET NULL,
    defect_id INTEGER REFERENCES "Defect"(defect_id) ON DELETE SET NULL,
    predicted_days INTEGER NOT NULL,
    prediction_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    storage_conditions JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. History Table (Central record for all scans and analyses)
CREATE TABLE IF NOT EXISTS "History" (
    history_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES "User"(user_id) ON DELETE SET NULL,
    image_id INTEGER REFERENCES "Bean_Image"(image_id) ON DELETE SET NULL,
    shelf_life_id INTEGER REFERENCES "Shelf_Life"(shelf_life_id) ON DELETE SET NULL,
    bean_type_id INTEGER REFERENCES "Bean_Type"(bean_type_id) ON DELETE SET NULL,
    defect_id INTEGER REFERENCES "Defect"(defect_id) ON DELETE SET NULL,
    healthy_percent DECIMAL(5,2) CHECK (healthy_percent >= 0 AND healthy_percent <= 100),
    defective_percent DECIMAL(5,2) CHECK (defective_percent >= 0 AND defective_percent <= 100),
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_name ON "User"("Name");
CREATE INDEX IF NOT EXISTS idx_user_role ON "User"(role);
CREATE INDEX IF NOT EXISTS idx_user_location ON "User"(location);

CREATE INDEX IF NOT EXISTS idx_bean_type_name ON "Bean_Type"(type_name);

CREATE INDEX IF NOT EXISTS idx_bean_image_user_id ON "Bean_Image"(user_id);
CREATE INDEX IF NOT EXISTS idx_bean_image_bean_type_id ON "Bean_Image"(bean_type_id);
CREATE INDEX IF NOT EXISTS idx_bean_image_capture_date ON "Bean_Image"(capture_date);

CREATE INDEX IF NOT EXISTS idx_defect_image_id ON "Defect"(image_id);
CREATE INDEX IF NOT EXISTS idx_defect_type ON "Defect"(defect_type);
CREATE INDEX IF NOT EXISTS idx_defect_severity ON "Defect"(severity_level);

CREATE INDEX IF NOT EXISTS idx_shelf_life_image_id ON "Shelf_Life"(image_id);
CREATE INDEX IF NOT EXISTS idx_shelf_life_bean_type_id ON "Shelf_Life"(bean_type_id);
CREATE INDEX IF NOT EXISTS idx_shelf_life_predicted_days ON "Shelf_Life"(predicted_days);

CREATE INDEX IF NOT EXISTS idx_history_user_id ON "History"(user_id);
CREATE INDEX IF NOT EXISTS idx_history_image_id ON "History"(image_id);
CREATE INDEX IF NOT EXISTS idx_history_created_at ON "History"(created_at);
CREATE INDEX IF NOT EXISTS idx_history_bean_type_id ON "History"(bean_type_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for tables with updated_at columns
CREATE TRIGGER update_user_updated_at 
    BEFORE UPDATE ON "User" 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_history_updated_at 
    BEFORE UPDATE ON "History" 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing
INSERT INTO "User" ("Name", contact_number, role, location) VALUES
    ('John Doe', '+1234567890', 'admin', 'New York'),
    ('Jane Smith', '+0987654321', 'user', 'Los Angeles'),
    ('Bob Wilson', '+1122334455', 'analyst', 'Chicago')
ON CONFLICT DO NOTHING;

INSERT INTO "Bean_Type" (type_name, description) VALUES
    ('Arabica', 'High-quality coffee bean with smooth, mild flavor'),
    ('Robusta', 'Strong, bitter coffee bean with high caffeine content'),
    ('Liberica', 'Rare coffee bean with unique, fruity flavor'),
    ('Excelsa', 'Complex coffee bean with tart, fruity notes'),
    ('Other', 'Miscellaneous coffee bean types')
ON CONFLICT DO NOTHING;

-- Create RLS (Row Level Security) policies
ALTER TABLE "User" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "Bean_Type" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "Bean_Image" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "Defect" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "Shelf_Life" ENABLE ROW LEVEL SECURITY;
ALTER TABLE "History" ENABLE ROW LEVEL SECURITY;

-- RLS Policies for User table
CREATE POLICY "Users can view own profile" ON "User"
    FOR SELECT USING (user_id = (auth.uid()::text)::integer);

CREATE POLICY "Users can update own profile" ON "User"
    FOR UPDATE USING (user_id = (auth.uid()::text)::integer);

CREATE POLICY "Admins can manage all users" ON "User"
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM "User" 
            WHERE user_id = (auth.uid()::text)::integer 
            AND role = 'admin'
        )
    );

-- RLS Policies for Bean_Type table (public read, admin write)
CREATE POLICY "Anyone can view bean types" ON "Bean_Type"
    FOR SELECT USING (true);

CREATE POLICY "Admins can manage bean types" ON "Bean_Type"
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM "User" 
            WHERE user_id = (auth.uid()::text)::integer 
            AND role = 'admin'
        )
    );

-- RLS Policies for Bean_Image table
CREATE POLICY "Users can view own images" ON "Bean_Image"
    FOR SELECT USING (
        user_id = (auth.uid()::text)::integer 
        OR user_id IS NULL
    );

CREATE POLICY "Users can insert own images" ON "Bean_Image"
    FOR INSERT WITH CHECK (
        user_id = (auth.uid()::text)::integer 
        OR user_id IS NULL
    );

CREATE POLICY "Users can update own images" ON "Bean_Image"
    FOR UPDATE USING (user_id = (auth.uid()::text)::integer);

CREATE POLICY "Users can delete own images" ON "Bean_Image"
    FOR DELETE USING (user_id = (auth.uid()::text)::integer);

-- RLS Policies for Defect table
CREATE POLICY "Users can view defects from own images" ON "Defect"
    FOR SELECT USING (
        image_id IN (
            SELECT image_id FROM "Bean_Image" 
            WHERE user_id = (auth.uid()::text)::integer 
            OR user_id IS NULL
        )
    );

CREATE POLICY "Users can insert defects for own images" ON "Defect"
    FOR INSERT WITH CHECK (
        image_id IN (
            SELECT image_id FROM "Bean_Image" 
            WHERE user_id = (auth.uid()::text)::integer
        )
    );

-- RLS Policies for Shelf_Life table
CREATE POLICY "Users can view shelf life from own images" ON "Shelf_Life"
    FOR SELECT USING (
        image_id IN (
            SELECT image_id FROM "Bean_Image" 
            WHERE user_id = (auth.uid()::text)::integer 
            OR user_id IS NULL
        )
    );

CREATE POLICY "Users can insert shelf life for own images" ON "Shelf_Life"
    FOR INSERT WITH CHECK (
        image_id IN (
            SELECT image_id FROM "Bean_Image" 
            WHERE user_id = (auth.uid()::text)::integer
        )
    );

-- RLS Policies for History table
CREATE POLICY "Users can view own history" ON "History"
    FOR SELECT USING (
        user_id = (auth.uid()::text)::integer 
        OR user_id IS NULL
    );

CREATE POLICY "Users can insert own history" ON "History"
    FOR INSERT WITH CHECK (
        user_id = (auth.uid()::text)::integer 
        OR user_id IS NULL
    );

CREATE POLICY "Users can update own history" ON "History"
    FOR UPDATE USING (user_id = (auth.uid()::text)::integer);

CREATE POLICY "Users can delete own history" ON "History"
    FOR DELETE USING (user_id = (auth.uid()::text)::integer);

-- Create views for common queries
CREATE OR REPLACE VIEW scan_summary AS
SELECT 
    h.history_id,
    u."Name" as user_name,
    bt.type_name as bean_type,
    bi.image_path,
    h.healthy_percent,
    h.defective_percent,
    h.confidence_score,
    h.created_at
FROM "History" h
LEFT JOIN "User" u ON h.user_id = u.user_id
LEFT JOIN "Bean_Type" bt ON h.bean_type_id = bt.bean_type_id
LEFT JOIN "Bean_Image" bi ON h.image_id = bi.image_id
ORDER BY h.created_at DESC;

-- Create function to get user statistics
CREATE OR REPLACE FUNCTION get_user_scan_stats(user_id_param INTEGER)
RETURNS TABLE(
    total_scans BIGINT,
    total_images BIGINT,
    avg_confidence DECIMAL(5,4),
    bean_types_count BIGINT,
    last_scan_date TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(DISTINCT h.history_id) as total_scans,
        COUNT(DISTINCT h.image_id) as total_images,
        AVG(h.confidence_score) as avg_confidence,
        COUNT(DISTINCT h.bean_type_id) as bean_types_count,
        MAX(h.created_at) as last_scan_date
    FROM "History" h
    WHERE h.user_id = user_id_param;
END;
$$ LANGUAGE plpgsql;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;
