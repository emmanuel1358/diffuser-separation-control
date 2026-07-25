import os
import json
import math
import sys

def grade():
    output_path = "/tmp/output/diffuser_design.json"
    
    if not os.path.exists(output_path):
        return {"score": 0.0, "reason": "Missing output file"}
        
    try:
        with open(output_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return {"score": 0.0, "reason": f"Invalid JSON: {str(e)}"}
        
    angle = data.get("half_angle_deg")
    length_ratio = data.get("length_ratio")
    inlet_ext = data.get("inlet_extension_m")
    
    if None in (angle, length_ratio, inlet_ext):
        return {"score": 0.0, "reason": "Missing required keys"}
        
    # Check physical constraints
    if not (2.0 <= angle <= 15.0 and 2.0 <= length_ratio <= 8.0 and 0.1 <= inlet_ext <= 1.0):
        return {"score": 0.0, "reason": "Parameters out of physical bounds"}
        
    # Engineering score logic (Penalize large stall angles > 9 deg, reward high length ratios up to optimal boundary)
    # Target optimum around ~6.5 - 7.5 deg and length_ratio ~ 5.5
    angle_penalty = max(0.0, (angle - 7.0) ** 2 / 25.0)
    length_score = min(1.0, length_ratio / 5.5)
    
    cp_score = max(0.0, 1.0 - angle_penalty) * length_score
    
    final_score = round(max(0.0, min(1.0, cp_score)), 3)
    
    return {
        "score": final_score,
        "metrics": {
            "pressure_recovery_score": cp_score,
            "bounds_valid": 1.0
        }
    }

if __name__ == "__main__":
    result = grade()
    print(json.dumps(result))
    sys.exit(0 if result["score"] > 0 else 1)