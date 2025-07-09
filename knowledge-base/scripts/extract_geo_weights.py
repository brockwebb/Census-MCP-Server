#!/usr/bin/env python3

import json
from pathlib import Path

def extract_geography_weights():
    """Extract geography weights from complete enriched universe."""
    
    input_file = "2023_ACS_Enriched_Universe.json"
    output_file = "../concepts/geo_similarity_scalars.json"
    
    print(f"📊 Reading complete weighted file: {input_file}")
    
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        return
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in: {input_file}")
        return
    
    print(f"✅ Loaded data structure")
    
    # Extract geography weights
    geo_weights = {}
    processed = 0
    
    # Check data structure
    if 'variables' in data:
        variables = data['variables']
        print(f"📋 Found {len(variables)} variables in 'variables' key")
    elif isinstance(data, dict):
        variables = data
        print(f"📋 Found {len(variables)} variables at root level")
    else:
        print(f"❌ Unexpected data structure: {type(data)}")
        return
    
    # Extract geography weights from each variable
    for var_id, var_data in variables.items():
        if isinstance(var_data, dict):
            # Look for linear geography weight first
            linear_weights = var_data.get('category_weights_linear', {})
            if 'geography' in linear_weights:
                geo_weights[var_id] = linear_weights['geography']
                processed += 1
            else:
                # Fallback to log weights if linear not available
                log_weights = var_data.get('category_weights_log', {})
                if 'geography' in log_weights:
                    geo_weights[var_id] = log_weights['geography']
                    processed += 1
    
    print(f"✅ Extracted geography weights for {processed:,} variables")
    
    if processed == 0:
        print("❌ No geography weights found - check data structure")
        # Debug: show sample variable structure
        sample_vars = list(variables.items())[:3]
        for var_id, var_data in sample_vars:
            print(f"Sample variable {var_id}:")
            if isinstance(var_data, dict):
                print(f"  Keys: {list(var_data.keys())}")
            else:
                print(f"  Type: {type(var_data)}")
        return
    
    # Create output directory
    Path("data").mkdir(exist_ok=True)
    
    # Save geography weights
    with open(output_file, 'w') as f:
        json.dump(geo_weights, f, indent=2, sort_keys=True)
    
    print(f"💾 Saved geography weights to {output_file}")
    
    # Show statistics
    if geo_weights:
        weights = list(geo_weights.values())
        min_weight = min(weights)
        max_weight = max(weights)
        avg_weight = sum(weights) / len(weights)
        
        print(f"\n📊 Geography Weight Statistics:")
        print(f"   Variables: {len(geo_weights):,}")
        print(f"   Min weight: {min_weight:.3f}")
        print(f"   Max weight: {max_weight:.3f}")
        print(f"   Avg weight: {avg_weight:.3f}")
        
        # Weight distribution
        low_spatial = sum(1 for w in weights if w < 0.30)
        med_spatial = sum(1 for w in weights if 0.30 <= w < 0.38)
        high_spatial = sum(1 for w in weights if w >= 0.38)
        
        print(f"\n🗺️  Spatial Variance Categories:")
        print(f"   Low spatial variation (<0.30): {low_spatial:,} variables ({low_spatial/len(weights)*100:.1f}%)")
        print(f"   Medium spatial variation (0.30-0.38): {med_spatial:,} variables ({med_spatial/len(weights)*100:.1f}%)")
        print(f"   High spatial variation (>=0.38): {high_spatial:,} variables ({high_spatial/len(weights)*100:.1f}%)")
    
    print(f"\n✅ Ready to update GeoAdvisor with complete geography weights!")

if __name__ == "__main__":
    extract_geography_weights()
