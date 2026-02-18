#!/usr/bin/env python3
"""
Algorithm Validation Script
===========================
Tests the drug repurposing algorithm against known successful cases
to establish scientific validity.

This script should be run BEFORE claiming any performance metrics.
"""

import asyncio
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation_dataset import (
    KNOWN_REPURPOSING_CASES,
    NEGATIVE_CONTROLS,
    get_validation_metrics_target,
    get_test_diseases
)


async def run_validation():
    """
    Run full validation pipeline.
    """
    print("="*80)
    print("DRUG REPURPOSING ALGORITHM - VALIDATION STUDY")
    print("="*80)
    print()
    
    # Import the actual pipeline
    try:
        from backend.pipeline.production_pipeline import ProductionPipeline
        from backend.pipeline.scorer import ProductionScorer
    except ImportError as e:
        print(f"❌ Cannot import pipeline: {e}")
        print("   Make sure you're running from the project root")
        return False
    
    pipeline = ProductionPipeline()
    
    results = {
        "true_positives": [],
        "false_negatives": [],
        "true_negatives": [],
        "false_positives": [],
        "scores": [],
    }
    
    print("📊 PHASE 1: Testing Known Successful Repurposing Cases")
    print("-" * 80)
    print()
    
    # Test positive cases
    for i, case in enumerate(KNOWN_REPURPOSING_CASES, 1):
        drug_name = case["drug_name"]
        disease = case["repurposed_for"]
        expected_min, expected_max = case["expected_score_range"]
        
        print(f"Test {i}/{len(KNOWN_REPURPOSING_CASES)}: {drug_name} → {disease}")
        print(f"  Expected score: {expected_min:.2f} - {expected_max:.2f}")
        
        try:
            # Run the actual analysis
            result = await pipeline.analyze_disease(
                disease_name=disease,
                min_score=0.1,  # Low threshold to catch everything
                max_results=50
            )
            
            if not result['success']:
                print(f"  ❌ Analysis failed: {result.get('error', 'Unknown')}")
                results["false_negatives"].append({
                    "drug": drug_name,
                    "disease": disease,
                    "reason": "analysis_failed"
                })
                continue
            
            # Find the drug in results
            found_drug = None
            for candidate in result['candidates']:
                if drug_name.lower() in candidate['drug_name'].lower():
                    found_drug = candidate
                    break
            
            if found_drug:
                score = found_drug['score']
                results["scores"].append(score)
                
                print(f"  ✅ Found! Score: {score:.3f}")
                
                # Check if within expected range
                if expected_min <= score <= expected_max:
                    print(f"  ✅ Within expected range!")
                    results["true_positives"].append({
                        "drug": drug_name,
                        "disease": disease,
                        "score": score,
                        "expected_range": (expected_min, expected_max),
                        "confidence": found_drug.get('confidence', 'unknown')
                    })
                else:
                    print(f"  ⚠️  Score outside expected range")
                    if score < expected_min:
                        print(f"      (Score too LOW - algorithm may be too conservative)")
                    else:
                        print(f"      (Score too HIGH - algorithm may be too liberal)")
                    
                    results["true_positives"].append({
                        "drug": drug_name,
                        "disease": disease,
                        "score": score,
                        "expected_range": (expected_min, expected_max),
                        "out_of_range": True
                    })
            else:
                print(f"  ❌ Drug not found in top 50 candidates")
                print(f"      This is a FALSE NEGATIVE - algorithm missed known success")
                results["false_negatives"].append({
                    "drug": drug_name,
                    "disease": disease,
                    "reason": "not_in_top_50"
                })
        
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results["false_negatives"].append({
                "drug": drug_name,
                "disease": disease,
                "reason": f"error: {str(e)}"
            })
        
        print()
    
    print()
    print("📊 PHASE 2: Testing Negative Controls")
    print("-" * 80)
    print()
    
    # Test negative cases
    for i, case in enumerate(NEGATIVE_CONTROLS, 1):
        drug_name = case["drug_name"]
        disease = case["disease"]
        expected_min, expected_max = case["expected_score_range"]
        
        print(f"Negative Test {i}/{len(NEGATIVE_CONTROLS)}: {drug_name} → {disease}")
        print(f"  Expected score: {expected_min:.2f} - {expected_max:.2f} (should be LOW)")
        
        try:
            result = await pipeline.analyze_disease(
                disease_name=disease,
                min_score=0.0,
                max_results=100
            )
            
            if not result['success']:
                print(f"  ⚠️  Analysis failed: {result.get('error', 'Unknown')}")
                continue
            
            # Find the drug
            found_drug = None
            for candidate in result['candidates']:
                if drug_name.lower() in candidate['drug_name'].lower():
                    found_drug = candidate
                    break
            
            if found_drug:
                score = found_drug['score']
                
                if score <= expected_max:
                    print(f"  ✅ Correctly scored LOW: {score:.3f}")
                    results["true_negatives"].append({
                        "drug": drug_name,
                        "disease": disease,
                        "score": score
                    })
                else:
                    print(f"  ❌ Scored TOO HIGH: {score:.3f}")
                    print(f"      This is a FALSE POSITIVE - algorithm over-predicts")
                    results["false_positives"].append({
                        "drug": drug_name,
                        "disease": disease,
                        "score": score,
                        "expected_max": expected_max
                    })
            else:
                print(f"  ✅ Not found (correctly excluded)")
                results["true_negatives"].append({
                    "drug": drug_name,
                    "disease": disease,
                    "score": 0.0,
                    "not_found": True
                })
        
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
        
        print()
    
    # Calculate metrics
    print()
    print("="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print()
    
    tp = len(results["true_positives"])
    fn = len(results["false_negatives"])
    tn = len(results["true_negatives"])
    fp = len(results["false_positives"])
    
    total_positive = tp + fn
    total_negative = tn + fp
    
    print(f"True Positives (TP):  {tp} / {total_positive}")
    print(f"False Negatives (FN): {fn} / {total_positive}")
    print(f"True Negatives (TN):  {tn} / {total_negative}")
    print(f"False Positives (FP): {fp} / {total_negative}")
    print()
    
    # Calculate performance metrics
    sensitivity = tp / total_positive if total_positive > 0 else 0
    specificity = tn / total_negative if total_negative > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    print("PERFORMANCE METRICS:")
    print(f"  Sensitivity (Recall): {sensitivity:.2%}")
    print(f"  Specificity:          {specificity:.2%}")
    print(f"  Precision:            {precision:.2%}")
    print(f"  Accuracy:             {accuracy:.2%}")
    print()
    
    # Compare to targets
    targets = get_validation_metrics_target()
    print("TARGET METRICS (for publication):")
    print(f"  Sensitivity: >{targets['sensitivity']:.0%}")
    print(f"  Specificity: >{targets['specificity']:.0%}")
    print(f"  Precision:   >{targets['precision']:.0%}")
    print()
    
    # Score distribution
    if results["scores"]:
        scores_array = np.array(results["scores"])
        print("SCORE DISTRIBUTION (for detected cases):")
        print(f"  Mean:   {np.mean(scores_array):.3f}")
        print(f"  Median: {np.median(scores_array):.3f}")
        print(f"  Std:    {np.std(scores_array):.3f}")
        print(f"  Min:    {np.min(scores_array):.3f}")
        print(f"  Max:    {np.max(scores_array):.3f}")
        print()
    
    # Pass/Fail determination
    print("="*80)
    passed = (
        sensitivity >= targets['sensitivity'] and
        specificity >= targets['specificity'] and
        precision >= targets['precision']
    )
    
    if passed:
        print("✅ VALIDATION PASSED - Algorithm meets publication standards")
        print()
        print("You can now cite these metrics in your paper:")
        print(f"  - Validated on {total_positive} known repurposing successes")
        print(f"  - Sensitivity: {sensitivity:.1%}")
        print(f"  - Specificity: {specificity:.1%}")
        print(f"  - Precision: {precision:.1%}")
    else:
        print("❌ VALIDATION FAILED - Algorithm needs improvement")
        print()
        print("Issues to address:")
        if sensitivity < targets['sensitivity']:
            print(f"  - Sensitivity too low ({sensitivity:.1%} < {targets['sensitivity']:.0%})")
            print("    → Algorithm misses too many true positives")
            print("    → Consider lowering score thresholds or adjusting weights")
        if specificity < targets['specificity']:
            print(f"  - Specificity too low ({specificity:.1%} < {targets['specificity']:.0%})")
            print("    → Algorithm produces too many false positives")
            print("    → Consider raising score thresholds or adding filters")
        if precision < targets['precision']:
            print(f"  - Precision too low ({precision:.1%} < {targets['precision']:.0%})")
            print("    → Too many predicted positives are wrong")
            print("    → Consider refining scoring mechanism")
    
    print("="*80)
    print()
    
    # Save detailed results
    output_file = Path(__file__).parent / "validation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📁 Detailed results saved to: {output_file}")
    print()
    
    await pipeline.close()
    
    return passed


if __name__ == "__main__":
    print()
    print("⚠️  IMPORTANT: This validation should be run BEFORE publication")
    print("   It tests your algorithm against known drug repurposing successes")
    print()
    print("   This will take 5-15 minutes to complete.")
    print()
    
    success = asyncio.run(run_validation())
    sys.exit(0 if success else 1)