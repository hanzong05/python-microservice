-- Fix boreholes table schema: Replace barangay_id with municipality_id
-- Run these commands in order

-- Step 1: Drop dependent views
DROP VIEW IF EXISTS v_municipality_statistics CASCADE;
DROP VIEW IF EXISTS v_complete_soil_data CASCADE;
DROP VIEW IF EXISTS v_liquefaction_risk_zones CASCADE;
DROP VIEW IF EXISTS v_bearing_capacity_by_layer CASCADE;

-- Step 2: Add new municipality_id column (if it doesn't exist)
ALTER TABLE boreholes ADD COLUMN IF NOT EXISTS municipality_id bigint;

-- Step 3: Update municipality_id from barangays table (if barangays still exist)
-- This maps through barangays to get municipality_id
UPDATE boreholes 
SET municipality_id = (
    SELECT municipality_id 
    FROM barangays 
    WHERE barangays.id = boreholes.barangay_id
    LIMIT 1
)
WHERE barangay_id IS NOT NULL;

-- Step 4: If barangays table is removed, you'll need to set municipality_id manually
-- Or update based on your data mapping

-- Step 5: Make municipality_id NOT NULL (after data is populated)
ALTER TABLE boreholes ALTER COLUMN municipality_id SET NOT NULL;

-- Step 6: Add foreign key constraint
ALTER TABLE boreholes 
ADD CONSTRAINT boreholes_municipality_id_fkey 
FOREIGN KEY (municipality_id) REFERENCES municipalities(id);

-- Step 7: Drop old barangay_id column
ALTER TABLE boreholes DROP COLUMN IF EXISTS barangay_id;

-- Step 8: Recreate views (adjust based on your actual view definitions)
-- Example for v_complete_soil_data (adjust column names as needed):
CREATE OR REPLACE VIEW v_complete_soil_data AS
SELECT 
    m.id as municipality_id,
    m.name as municipality,
    b.id as borehole_record_id,
    b.borehole_id,
    b.elevation,
    b.latitude,
    b.longitude,
    sl.id as layer_id,
    sl.layer_number,
    sl.depth_from_m,
    sl.depth_to_m,
    sl.depth_range,
    sl.spt_n_value,
    sl.spt_n60,
    sl.unit_weight,
    sl.fines_content,
    sl.groundwater_depth_m,
    sl.pga_g,
    sl.csr,
    sl.liquefaction,
    sl.settlement_cm,
    sl.bearing_capacity_kpa,
    sl.friction_angle,
    sl.cohesion_kpa,
    sl.elastic_modulus_es,
    sl.created_at,
    sl.updated_at
FROM municipalities m
JOIN boreholes b ON b.municipality_id = m.id
JOIN soil_layers sl ON sl.borehole_id = b.id;

-- Example for v_municipality_statistics (adjust as needed):
CREATE OR REPLACE VIEW v_municipality_statistics AS
SELECT 
    m.name as municipality,
    COUNT(DISTINCT b.id) as borehole_count,
    COUNT(sl.id) as total_samples,
    AVG(sl.spt_n_value) as avg_spt_n,
    AVG(sl.unit_weight) as avg_unit_weight,
    AVG(sl.bearing_capacity_kpa) as avg_bearing_capacity_kpa,
    COUNT(CASE WHEN sl.liquefaction = true THEN 1 END) as liquefaction_zones,
    ROUND(COUNT(CASE WHEN sl.liquefaction = true THEN 1 END)::numeric / NULLIF(COUNT(sl.id), 0) * 100, 2) as liquefaction_percentage
FROM municipalities m
LEFT JOIN boreholes b ON b.municipality_id = m.id
LEFT JOIN soil_layers sl ON sl.borehole_id = b.id
GROUP BY m.id, m.name;
