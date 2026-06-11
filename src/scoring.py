import json
import re
import os
from constants import OUTPUT_PATH

# Goal: Score extracted values against ground_truth.csv


def normalize_numeric(value):
    """
    Try to extract a number from a string for comparison.
    Strips commas, dollar signs, and common suffixes.
    """
    cleaned = re.sub(r"[,$%]", "", str(value).strip())
    # Try to pull out the first number (int or float)
    match = re.search(r"-?[\d]+\.?[\d]*", cleaned)
    if match:
        return float(match.group())
    return None


def score_numeric(expected, extracted, tolerance=0.05):
    """
    Score a numeric extraction. Returns 1.0 for exact match,
    partial credit if within tolerance, 0.0 otherwise.
    """
    expected_num = normalize_numeric(expected)
    extracted_num = normalize_numeric(extracted)

    if expected_num is None or extracted_num is None:
        return 0.0

    if expected_num == 0:
        return 1.0 if extracted_num == 0 else 0.0

    error = abs(expected_num - extracted_num) / abs(expected_num)
    if error == 0:
        return 1.0
    elif error <= tolerance:
        return 0.5
    else:
        return 0.0


def score_categorical(expected, extracted):
    """
    Score a categorical extraction. Case-insensitive exact match.
    """
    expected_clean = str(expected).strip().lower()
    extracted_clean = str(extracted).strip().lower()

    if expected_clean == extracted_clean:
        return 1.0

    # Partial credit if expected is contained in extracted or vice versa
    if expected_clean in extracted_clean or extracted_clean in expected_clean:
        return 0.5

    return 0.0


def score_results(results_path):
    """
    Score all extraction results. Returns a summary dict.
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    numeric_vars = ["Total Revenues", "Net Income"]
    categorical_vars = ["Stock Exchange"]
    scores = []

    for result in results:
        variable = result["variable"]
        expected = result["expected_value"]
        extracted = result["extracted_value"]

        if variable in numeric_vars:
            score = score_numeric(expected, extracted)
        else:
            score = score_categorical(expected, extracted)

        scores.append({
            "query": result["query"],
            "variable": variable,
            "expected": expected,
            "extracted": extracted,
            "score": score,
        })

    total = len(scores)
    avg_score = sum(s["score"] for s in scores) / total if total > 0 else 0

    return {
        "scores": scores,
        "total_queries": total,
        "average_score": round(avg_score, 3),
    }


if __name__ == "__main__":
    results_path = os.path.join(OUTPUT_PATH, "extraction_results.json")
    summary = score_results(results_path)

    print(f"Total queries: {summary['total_queries']}")
    print(f"Average score: {summary['average_score']}")
    print()

    for s in summary["scores"]:
        status = "PASS" if s["score"] == 1.0 else "PARTIAL" if s["score"] > 0 else "FAIL"
        print(f"  [{status}] {s['variable']}: expected={s['expected']} | extracted={s['extracted']}")

    # Write scoring output
    output_path = os.path.join(OUTPUT_PATH, "scoring_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote scoring results to {output_path}")
