import os
import sys
import subprocess

def run_script(script_name, args=[]):
    print("=" * 60)
    print(f"RUNNING: {script_name} {' '.join(args)}")
    print("=" * 60)
    res = subprocess.run([sys.executable, script_name] + args, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("ERRORS:")
        print(res.stderr)
    print("\n")

def main():
    # 1. Audit submissions
    run_script("scratch/audit_submission.py")
    
    # 2. Check class pixel counts for each file
    files = [
        "ensemble_multiclass_w60_c3t50_area128_submission.csv",
        "ensemble_multiclass_w55_c3t50_area128_submission.csv",
        "ensemble_multiclass_w60_c3t50_area64_submission.csv"
    ]
    for f in files:
        if os.path.exists(f):
            run_script("scratch/check_submission_class_pixels.py", [f])
            
    # 3. Compare candidates with baseline
    run_script("scratch/compare_candidates.py")

if __name__ == "__main__":
    main()
