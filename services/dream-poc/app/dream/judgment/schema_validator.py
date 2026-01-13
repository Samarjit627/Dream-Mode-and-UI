"""
Stage 2 - Schema Validator for Proportion JSONs
Ensures that all proportion band files follow the strict structure required by the logic engine.
"""
import os
import json
from typing import Dict, List, Any

# Required keys in the JSON
REQUIRED_KEYS = ["segment", "archetype", "view", "priority_weights", "proportion_bands"]

# Valid enums
VALID_SEGMENTS = ["microcar", "hatchback", "sedan", "suv", "sportscar", "truck", "van"]
VALID_ARCHETYPES = ["sporty", "balanced", "tall"]
VALID_VIEWS = ["front_3q", "side", "rear_3q"]
VALID_WEIGHTS = ["high", "medium", "low"]


def validate_json_file(file_path: str) -> bool:
    """
    Validates a single JSON file against the schema rules.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # 1. Check required keys
        for key in REQUIRED_KEYS:
            if key not in data:
                print(f"  [Validator] ✗ {os.path.basename(file_path)} missing key: {key}")
                return False
                
        # 2. Check Enums
        if data["segment"] not in VALID_SEGMENTS:
            print(f"  [Validator] ✗ {os.path.basename(file_path)} invalid segment: {data['segment']}")
        
        if data["archetype"] not in VALID_ARCHETYPES:
            print(f"  [Validator] ✗ {os.path.basename(file_path)} invalid archetype: {data['archetype']}")
            return False
            
        if data["view"] not in VALID_VIEWS:
            print(f"  [Validator] ✗ {os.path.basename(file_path)} invalid view: {data['view']}")
            return False
            
        # 3. Check Priority Weights
        for k, v in data["priority_weights"].items():
            if v not in VALID_WEIGHTS:
                print(f"  [Validator] ✗ {os.path.basename(file_path)} invalid weight for {k}: {v}")
                return False
                
        # 4. Check Proportion Bands logic
        for k, band in data["proportion_bands"].items():
            if not isinstance(band, list) or len(band) != 2:
                print(f"  [Validator] ✗ {os.path.basename(file_path)} band {k} must be [min, max]")
                return False
                
            min_val, max_val = band
            
            # Rule: min < max
            if min_val >= max_val:
                print(f"  [Validator] ✗ {os.path.basename(file_path)} band {k} inverted or equal: {min_val} >= {max_val}")
                return False
                
            # Rule: width >= 0.02
            if (max_val - min_val) < 0.02:
                print(f"  [Validator] ✗ {os.path.basename(file_path)} band {k} too tight (<0.02 width)")
                return False
                
        print(f"  [Validator] ✓ {os.path.basename(file_path)} passed")
        return True
        
    except json.JSONDecodeError:
        print(f"  [Validator] ✗ {os.path.basename(file_path)} is not valid JSON")
        return False
    except Exception as e:
        print(f"  [Validator] ✗ {os.path.basename(file_path)} error: {str(e)}")
        return False

def validate_library(directory: str):
    """
    Validates all JSON files in the library directory.
    """
    print("=" * 60)
    print(f"PROPORTION LIBRARY VALIDATION: {directory}")
    print("=" * 60)
    
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    passed = 0
    failed = 0
    
    for filename in sorted(files):
        if validate_json_file(os.path.join(directory, filename)):
            passed += 1
        else:
            failed += 1
            
    print("-" * 60)
    print(f"Summary: {passed} PASSED, {failed} FAILED")
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    # Test run
    current_dir = os.path.dirname(os.path.abspath(__file__))
    proportions_dir = os.path.join(current_dir, "../proportions")
    validate_library(proportions_dir)
