#!/usr/bin/env python3
"""
Diagnostic script to inspect what features the scaler expects
"""
import os
import io
import json
import joblib
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

def inspect_scaler():
    """Load and inspect the scaler to see what features it expects"""
    
    # Connect to Supabase
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    if not supabase_url or not supabase_key:
        print("ERROR: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")
        return
    
    client = create_client(supabase_url, supabase_key)
    bucket_name = os.getenv('SUPABASE_STORAGE_BUCKET', 'geotechnical-data')
    
    try:
        # Load scaler
        print("Loading scaler from Supabase...")
        scaler_data = client.storage.from_(bucket_name).download('models/scaler.pkl')
        scaler = joblib.load(io.BytesIO(scaler_data))
        
        print("\n" + "="*80)
        print("SCALER INFORMATION")
        print("="*80)
        
        # Check if scaler has feature_names_in_ attribute (sklearn >= 1.0)
        if hasattr(scaler, 'feature_names_in_'):
            print(f"\nNumber of features: {len(scaler.feature_names_in_)}")
            print("\nExpected feature names (in order):")
            for i, name in enumerate(scaler.feature_names_in_):
                print(f"  {i+1:2d}. {name}")
        else:
            print("\nScaler does not have feature_names_in_ attribute")
            print("This scaler was trained with an older sklearn version or without named features")
            
        # Check n_features_in_
        if hasattr(scaler, 'n_features_in_'):
            print(f"\nNumber of features expected: {scaler.n_features_in_}")
        
        # Show mean and scale
        if hasattr(scaler, 'mean_'):
            print(f"\nMean values: {scaler.mean_[:5]}... (showing first 5)")
        if hasattr(scaler, 'scale_'):
            print(f"Scale values: {scaler.scale_[:5]}... (showing first 5)")
        
        print("\n" + "="*80)
        
        # Load metadata to see what features were used during training
        try:
            print("\nLoading metadata...")
            metadata_data = client.storage.from_(bucket_name).download('models/ann_metadata.json')
            metadata = json.loads(metadata_data.decode('utf-8'))
            
            print("\nMETADATA - Training Information:")
            if 'features' in metadata:
                print(f"\nFeatures from metadata ({len(metadata['features'])} total):")
                for i, name in enumerate(metadata['features']):
                    print(f"  {i+1:2d}. {name}")
            
            if 'training_date' in metadata:
                print(f"\nTraining date: {metadata['training_date']}")
            
            if 'model_architecture' in metadata:
                print(f"Model architecture: {metadata['model_architecture']}")
                
        except Exception as e:
            print(f"\nCould not load metadata: {e}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_scaler()