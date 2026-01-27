"""
Geotechnical Data Processing Microservice
FastAPI service for Tarlac Liquefaction Prediction System
Deploy to Render.com with Supabase PostgreSQL backend
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import io
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Tarlac Liquefaction Prediction API",
    description="Geotechnical data processing and ML training pipeline",
    version="1.0.0"
)

# CORS configuration for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')  # Set in Render environment
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,  # Render has connection limits
    pool_pre_ping=True,  # Verify connections before using
    echo=False
)

# ============================================================================
# PYDANTIC MODELS (API Request/Response)
# ============================================================================

class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # 'processing', 'completed', 'failed'
    message: str
    records_processed: Optional[int] = None
    timestamp: datetime

class DataStats(BaseModel):
    total_records: int
    liquefiable_layers: int
    avg_settlement_cm: float
    avg_bearing_capacity_kpa: float
    depth_range: Dict[str, float]

class TrainingDataRequest(BaseModel):
    test_size: float = 0.2
    random_state: int = 42
    include_features: Optional[List[str]] = None

class MLDataset(BaseModel):
    X_train_shape: tuple
    X_test_shape: tuple
    y_train_shape: tuple
    y_test_shape: tuple
    feature_names: List[str]
    target_columns: List[str]
    download_url: Optional[str] = None

# ============================================================================
# GEOTECHNICAL CALCULATIONS
# ============================================================================

class GeotechnicalCalculator:
    """Core geotechnical calculations following research methodology"""
    
    @staticmethod
    def calculate_n1_60(spt_n: pd.Series, sigma_prime: pd.Series, depth: pd.Series) -> pd.Series:
        """Calculate corrected SPT N-value"""
        Pa = 100  # kPa
        CN = np.minimum(2.0, (Pa / sigma_prime) ** 0.5)
        
        CR = np.where(depth < 3, 0.75,
             np.where(depth < 4, 0.80,
             np.where(depth < 6, 0.85,
             np.where(depth < 10, 0.95, 1.0))))
        
        return (spt_n * CN * CR).clip(lower=0)
    
    @staticmethod
    def calculate_csr(pga_g: pd.Series, sigma_v: pd.Series, 
                      sigma_prime: pd.Series, depth: pd.Series) -> pd.Series:
        """Calculate Cyclic Stress Ratio (Seed & Idriss 1971)"""
        rd = np.where(depth <= 9.15,
                     1.0 - 0.00765 * depth,
                     1.174 - 0.0267 * depth)
        rd = rd.clip(lower=0.1)
        
        return (0.65 * pga_g * (sigma_v / sigma_prime) * rd).clip(lower=0)
    
    @staticmethod
    def calculate_crr(n1_60: pd.Series, fines_content: pd.Series) -> pd.Series:
        """Calculate Cyclic Resistance Ratio (Idriss & Boulanger 2008)"""
        fc = fines_content.fillna(0)
        
        crr_75 = np.where(n1_60 < 30,
                         np.exp(n1_60/14.1 + (n1_60/126)**2 - 
                               (n1_60/23.6)**3 + (n1_60/25.4)**4 - 2.8),
                         np.nan)
        
        delta_crr = np.where(fc <= 5, 0,
                    np.where(fc <= 35, np.exp(1.76 - 190/fc**2), 0))
        
        return (crr_75 + delta_crr).clip(lower=0)
    
    @staticmethod
    def calculate_settlement(n1_60: pd.Series, csr: pd.Series, 
                           crr: pd.Series, thickness: pd.Series) -> pd.Series:
        """Calculate settlement (Tokimatsu & Seed 1987) in cm"""
        fs = crr / (csr + 1e-10)
        
        volumetric_strain = np.where(
            fs >= 1.0, 0,
            0.01 * np.exp(-0.1 * n1_60) * (csr / 0.2)
        )
        
        settlement_mm = volumetric_strain * thickness * 1000
        return (settlement_mm / 10).clip(lower=0)
    
    @staticmethod
    def calculate_bearing_capacity(gamma: pd.Series, width: pd.Series,
                                  depth: pd.Series, phi: pd.Series,
                                  cohesion: pd.Series) -> pd.Series:
        """Calculate bearing capacity (Terzaghi 1943) in kPa"""
        phi_rad = np.radians(phi)
        c = cohesion.fillna(0)
        
        Nq = np.exp(np.pi * np.tan(phi_rad)) * (np.tan(np.pi/4 + phi_rad/2) ** 2)
        Nc = np.where(phi < 0.001, 5.7, (Nq - 1) / np.tan(phi_rad))
        Ngamma = 2 * (Nq + 1) * np.tan(phi_rad)
        
        # Square footing shape factors
        qult = 1.3 * c * Nc + gamma * depth * Nq + 0.4 * gamma * width * Ngamma
        
        return qult.replace([np.inf, -np.inf], np.nan).clip(lower=0)

# ============================================================================
# DATA PROCESSING PIPELINE
# ============================================================================

class DataProcessor:
    """Process Excel data and insert into Supabase"""
    
    def __init__(self):
        self.calc = GeotechnicalCalculator()
    
    def process_excel(self, file_content: bytes) -> pd.DataFrame:
        """Load and process Excel file"""
        logger.info("Loading Excel file...")
        
        depth_sheets = [
            '0m-1.5m', '1.5m-3.0m', '3.0m-4.5m', '4.5m-6.0m', '6.0m-7.5m',
            '7.5m-9.0m', '9.0m-10.5m', '10.5m-12.0m',
            '12.0m-13.5m', '13.5m-15.0m'
        ]
        
        all_data = []
        excel_file = io.BytesIO(file_content)
        
        for idx, sheet in enumerate(depth_sheets):
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet, header=1)
                
                dmin, dmax = sheet.replace('m', '').split('-')
                df['depth_from_m'] = float(dmin)
                df['depth_to_m'] = float(dmax)
                df['layer_number'] = idx + 1
                
                all_data.append(df)
                logger.info(f"Loaded {len(df)} records from {sheet}")
            except Exception as e:
                logger.warning(f"Skipped {sheet}: {e}")
        
        df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Total records loaded: {len(df)}")
        
        return self.clean_and_calculate(df)
    
    def clean_and_calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data and calculate all geotechnical parameters"""
        logger.info("Processing geotechnical calculations...")
        
        # Remove empty rows
        df = df.dropna(how='all')
        df = df[df['SPT N-Value'].notna()]
        
        # Column mapping to database schema
        column_map = {
            'Borehole ID': 'borehole_id',
            'SPT N-Value': 'spt_n_value',
            'Unit Weight (γ)': 'unit_weight',
            'Natural Water Content (ω)': 'moisture_content',
            'Plasticity Index (PI)': 'plasticity_index',
            'Liquid Limit': 'liquid_limit',
            'Plastic Limit': 'plastic_limit',
            'Fines Content': 'fines_content',
            'Mean Particle Size (D50) (mm)': 'mean_particle_size_d50',
            'Groundwater Level (m)': 'groundwater_depth_m',
            'Internal Friction Angle': 'friction_angle',
            'Cohesion (c)': 'cohesion_kpa',
            'Foundation Width (B)': 'foundation_width_m',
            'Foundation Depth (D)': 'foundation_depth_m',
            'Peak Ground Acceleration': 'pga_g',
            'Effective Overburden Presssure': 'effective_overburden_pressure',
            'Total Overburden Pressure': 'total_overburden_pressure',
        }
        
        df = df.rename(columns=column_map)
        
        # Parse PGA
        if 'pga_g' in df.columns:
            df['pga_g'] = pd.to_numeric(
                df['pga_g'].astype(str).str.extract(r'(\d+\.?\d*)')[0],
                errors='coerce'
            )
        
        # Numeric conversion
        numeric_cols = [
            'spt_n_value', 'unit_weight', 'moisture_content', 'plasticity_index',
            'fines_content', 'friction_angle', 'cohesion_kpa',
            'foundation_width_m', 'foundation_depth_m', 'pga_g',
            'effective_overburden_pressure', 'total_overburden_pressure'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate depth midpoint
        df['depth_mid'] = (df['depth_from_m'] + df['depth_to_m']) / 2
        
        # Geotechnical calculations
        df['spt_n160'] = self.calc.calculate_n1_60(
            df['spt_n_value'],
            df['effective_overburden_pressure'],
            df['depth_mid']
        )
        
        df['csr'] = self.calc.calculate_csr(
            df['pga_g'],
            df['total_overburden_pressure'],
            df['effective_overburden_pressure'],
            df['depth_mid']
        )
        
        df['cyclic_strength_ratio'] = self.calc.calculate_crr(
            df['spt_n160'],
            df['fines_content']
        )
        
        # Liquefaction assessment (DPWH BSDS 2013)
        fs = df['cyclic_strength_ratio'] / (df['csr'] + 1e-10)
        df['liquefaction'] = (
            ((df['csr'] > 0.15) | (df['spt_n_value'] < 15)) & (fs < 1.0)
        )
        
        df['liquefaction_risk_level'] = np.where(
            fs >= 1.5, 'Low',
            np.where(fs >= 1.2, 'Moderate',
            np.where(fs >= 1.0, 'High', 'Very High'))
        )
        
        # Settlement
        thickness = df['depth_to_m'] - df['depth_from_m']
        df['settlement_cm'] = self.calc.calculate_settlement(
            df['spt_n160'], df['csr'], df['cyclic_strength_ratio'], thickness
        )
        
        # Bearing capacity
        df['bearing_capacity_kpa'] = self.calc.calculate_bearing_capacity(
            df['unit_weight'],
            df['foundation_width_m'],
            df['foundation_depth_m'],
            df['friction_angle'],
            df['cohesion_kpa']
        )
        
        df['qa_allowable_kpa'] = df['bearing_capacity_kpa'] / 3.0
        
        logger.info(f"Liquefaction layers: {df['liquefaction'].sum()}")
        logger.info(f"Avg settlement: {df['settlement_cm'].mean():.2f} cm")
        
        return df
    
    def insert_to_database(self, df: pd.DataFrame) -> int:
        """Insert processed data into soil_layers table"""
        logger.info("Inserting data to Supabase...")
        
        # Select columns matching database schema
        db_columns = [
            'borehole_id', 'layer_number', 'depth_from_m', 'depth_to_m',
            'spt_n_value', 'spt_n160', 'unit_weight', 'moisture_content',
            'plasticity_index', 'liquid_limit', 'plastic_limit',
            'fines_content', 'mean_particle_size_d50', 'groundwater_depth_m',
            'friction_angle', 'cohesion_kpa', 'pga_g', 'csr',
            'cyclic_strength_ratio', 'liquefaction', 'liquefaction_risk_level',
            'settlement_cm', 'foundation_width_m', 'foundation_depth_m',
            'bearing_capacity_kpa', 'qa_allowable_kpa',
            'effective_overburden_pressure', 'total_overburden_pressure'
        ]
        
        available_cols = [col for col in db_columns if col in df.columns]
        df_to_insert = df[available_cols].copy()
        
        # Add timestamps
        df_to_insert['created_at'] = datetime.now()
        df_to_insert['updated_at'] = datetime.now()
        
        # Insert to database
        rows_inserted = df_to_insert.to_sql(
            'soil_layers',
            engine,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=500
        )
        
        logger.info(f"Inserted {len(df_to_insert)} rows")
        return len(df_to_insert)

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """API health check"""
    return {
        "service": "Tarlac Liquefaction Prediction API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Check database connectivity"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")

@app.post("/api/upload", response_model=ProcessingStatus)
async def upload_geotechnical_data(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload Excel file with geotechnical data
    Processes and inserts into database
    """
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files allowed")
    
    try:
        # Read file content
        content = await file.read()
        
        # Process data
        processor = DataProcessor()
        df = processor.process_excel(content)
        
        # Insert to database
        records_inserted = processor.insert_to_database(df)
        
        return ProcessingStatus(
            job_id=f"job_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            status="completed",
            message="Data processed and inserted successfully",
            records_processed=records_inserted,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats", response_model=DataStats)
async def get_data_statistics():
    """Get statistics from soil_layers table"""
    try:
        query = text("""
            SELECT 
                COUNT(*) as total_records,
                SUM(CASE WHEN liquefaction = true THEN 1 ELSE 0 END) as liquefiable_layers,
                AVG(settlement_cm) as avg_settlement,
                AVG(bearing_capacity_kpa) as avg_bearing_capacity,
                MIN(depth_from_m) as min_depth,
                MAX(depth_to_m) as max_depth
            FROM soil_layers
            WHERE spt_n_value IS NOT NULL
        """)
        
        with engine.connect() as conn:
            result = conn.execute(query).fetchone()
        
        return DataStats(
            total_records=result[0] or 0,
            liquefiable_layers=result[1] or 0,
            avg_settlement_cm=float(result[2] or 0),
            avg_bearing_capacity_kpa=float(result[3] or 0),
            depth_range={"min": float(result[4] or 0), "max": float(result[5] or 0)}
        )
        
    except Exception as e:
        logger.error(f"Stats query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ml/prepare-training-data", response_model=MLDataset)
async def prepare_ml_training_data(request: TrainingDataRequest):
    """
    Fetch data from ml_training_data view and prepare train/test split
    Returns dataset info and download link for CSV
    """
    try:
        # Fetch training data from view
        query = "SELECT * FROM ml_training_data"
        df = pd.read_sql(query, engine)
        
        logger.info(f"Fetched {len(df)} records for ML training")
        
        # Define feature and target columns
        feature_cols = [
            'latitude', 'longitude', 'elevation', 'layer_number',
            'depth_from_m', 'depth_to_m', 'depth_mid_m',
            'spt_n_value', 'spt_n160', 'unit_weight', 'moisture_content',
            'plasticity_index', 'fines_content', 'friction_angle',
            'effective_overburden_pressure', 'groundwater_depth_m',
            'pga_g', 'csr', 'cyclic_strength_ratio',
            'foundation_width_m', 'foundation_depth_m'
        ]
        
        target_cols = [
            'liquefaction_binary',
            'settlement_cm',
            'bearing_capacity_kpa',
            'qa_allowable_kpa'
        ]
        
        # Filter to requested features if specified
        if request.include_features:
            feature_cols = [f for f in feature_cols if f in request.include_features]
        
        # Remove rows with missing critical data
        required_cols = feature_cols + target_cols
        df_clean = df[required_cols].dropna()
        
        logger.info(f"Clean dataset: {len(df_clean)} records")
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        
        X = df_clean[feature_cols]
        y = df_clean[target_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=request.test_size,
            random_state=request.random_state,
            stratify=df_clean['liquefaction_binary']  # Stratify by liquefaction
        )
        
        # Save datasets to CSV (in production, use cloud storage)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # For now, return shapes and metadata
        # In production, upload to S3/cloud storage and return download URL
        
        return MLDataset(
            X_train_shape=X_train.shape,
            X_test_shape=X_test.shape,
            y_train_shape=y_train.shape,
            y_test_shape=y_test.shape,
            feature_names=list(feature_cols),
            target_columns=list(target_cols),
            download_url=None  # Implement cloud storage integration
        )
        
    except Exception as e:
        logger.error(f"ML data preparation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/feature-importance")
async def get_feature_importance():
    """Get correlation of features with liquefaction potential"""
    try:
        query = """
            SELECT 
                CORR(spt_n_value, liquefaction_binary::int) as spt_corr,
                CORR(spt_n160, liquefaction_binary::int) as n160_corr,
                CORR(csr, liquefaction_binary::int) as csr_corr,
                CORR(fines_content, liquefaction_binary::int) as fines_corr,
                CORR(depth_mid_m, liquefaction_binary::int) as depth_corr,
                CORR(groundwater_depth_m, liquefaction_binary::int) as gwl_corr
            FROM ml_training_data
            WHERE liquefaction_binary IS NOT NULL
        """
        
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
        
        return {
            "correlations": {
                "spt_n_value": float(result[0] or 0),
                "spt_n160": float(result[1] or 0),
                "csr": float(result[2] or 0),
                "fines_content": float(result[3] or 0),
                "depth_mid": float(result[4] or 0),
                "groundwater_depth": float(result[5] or 0)
            },
            "note": "Negative correlation = less likely to liquefy"
        }
        
    except Exception as e:
        logger.error(f"Feature importance query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/clear-data")
async def clear_all_data():
    """Clear all data from soil_layers table (use with caution!)"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("DELETE FROM soil_layers"))
            conn.commit()
        
        return {
            "status": "success",
            "message": "All soil layers data deleted",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Clear data failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# RUN SERVER (for local testing)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)