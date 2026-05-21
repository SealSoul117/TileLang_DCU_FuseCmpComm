import re
import csv
import argparse
import sys

def parse_log(log_path):
    pattern = re.compile(
        r"Size:\s*(\d+)x(\d+)x(\d+),\s*TFLOPS:\s*([\d.]+)"
    )

    results = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                m, n, k, tflops = match.groups()
                results.append({
                    "M": int(m),
                    "N": int(n),
                    "K": int(k),
                    "TFLOPS": float(tflops)
                })
    return results

def write_csv(data, output_path):
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["M", "N", "K", "TFLOPS"]
        )
        writer.writeheader()
        writer.writerows(data)

def main():
    parser = argparse.ArgumentParser(
        description="Parse TileLang GEMM log and export CSV"
    )
    parser.add_argument(
        "logfile",
        default="gemm.log",
        help="Path to the log file (e.g. gemm.log)"
    )
    parser.add_argument(
        "-o", "--output",
        default="gemm_results.csv",
        help="Output CSV file name"
    )

    args = parser.parse_args()

    try:
        data = parse_log(args.logfile)
    except FileNotFoundError:
        print(f"[ERROR] Log file not found: {args.logfile}", file=sys.stderr)
        sys.exit(1)

    if not data:
        print("[WARN] No Size/TFLOPS entries found.", file=sys.stderr)
        return

    write_csv(data, args.output)
    print(f"✅ CSV written to: {args.output}")

if __name__ == "__main__":
    main()