"""Run the whole pipeline end to end (works on Windows, macOS, Linux).

    python run_all.py              # full run (about 2-4 minutes)
    python run_all.py --quick      # small datasets, smoke run (about 20 seconds)
"""
import argparse
import subprocess
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
args = ap.parse_args()

n_h, n_f, n_b = (800, 100, 300) if args.quick else (6000, 800, 1000)
steps = [
    ["-m", "otaml.generate_dataset", "--n-healthy", str(n_h), "--n-per-fault", str(n_f)],
    ["-m", "otaml.train_surrogate"],
    ["-m", "otaml.train_classifier"],
    ["-m", "otaml.benchmark", "--n", str(n_b)],
    ["-m", "otaml.optimize"],
]
t0 = time.time()
for s in steps:
    print(f"\n=== python {' '.join(s)} ===", flush=True)
    subprocess.run([sys.executable, *s], check=True)
print(f"\nDone in {time.time() - t0:.0f}s.  Now run:  streamlit run app/app.py")
