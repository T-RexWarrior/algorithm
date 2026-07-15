from __future__ import annotations

import argparse

from gpp_inversion.promotion import evaluate_promotion, write_promotion_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the fixed GPP candidate promotion gates")
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("output")
    parser.add_argument("--high-target-threshold", type=float, required=True)
    args = parser.parse_args()
    report = evaluate_promotion(
        args.baseline, args.candidate,
        high_target_threshold=args.high_target_threshold,
    )
    write_promotion_report(report, args.output)
    print("PASS" if report["passed"] else "FAIL")


if __name__ == "__main__":
    main()
