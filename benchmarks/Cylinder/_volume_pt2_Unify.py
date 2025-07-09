import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def normalize_J_minus1(row, noise=1e-12):
    if abs(row['J']) > 1e-8:
        factor = -row['J']  # This makes J = -1 after division
        for k in 'ABCDEFGHIJ':
            row[k] = row[k] / factor
    for k in 'ABCDEFGHIJ':
        if abs(row[k]) < noise:
            row[k] = 0.0
    return row

def unify_coeffs(input_csv, output_csv=None):
    df = pd.read_csv(input_csv)
    df = df.apply(normalize_J_minus1, axis=1)
    
    # Set all coefficients to analytic cylinder [1, 1, 0, 0, 0, 0, 0, 0, 0, -1]
    analytic_coeffs = [1, 1, 0, 0, 0, 0, 0, 0, 0, -1]
    for i, k in enumerate("ABCDEFGHIJ"):
        df[k] = analytic_coeffs[i]
    
    if output_csv is None:
        root, ext = os.path.splitext(input_csv)
        output_csv = root + "_unified.csv"
    df.to_csv(output_csv, index=False)
    print(f"All coefficients set to analytic cylinder. Saved as {output_csv}")
    return output_csv

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python _volume_pt2_Unify.py input.csv [output.csv]")
    else:
        in_csv = sys.argv[1]
        out_csv = sys.argv[2] if len(sys.argv) > 2 else None
        unify_coeffs(in_csv, out_csv)
