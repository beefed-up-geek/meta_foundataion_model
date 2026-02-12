"""WorkBench Answer Checker

평가 디렉터리와 정답 디렉터리를 비교하여 정확도를 측정합니다.

사용법:
    python check_answer.py <prediction_dir> <answer_dir>

예시:
    python check_answer.py _temp_workspace/20260205_abc123 dataset/answer_csv/calendar/query_001
"""

import os
import sys
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List


def compare_csv_files(pred_path: Path, answer_path: Path) -> Tuple[bool, str]:
    """두 CSV 파일을 비교합니다.

    Args:
        pred_path: 예측 CSV 파일 경로
        answer_path: 정답 CSV 파일 경로

    Returns:
        (일치 여부, 상세 메시지)
    """
    try:
        pred_df = pd.read_csv(pred_path)
        answer_df = pd.read_csv(answer_path)
    except Exception as e:
        return False, f"CSV 읽기 오류: {e}"

    # 컬럼 비교
    if set(pred_df.columns) != set(answer_df.columns):
        return False, f"컬럼 불일치 - 예측: {list(pred_df.columns)}, 정답: {list(answer_df.columns)}"

    # 컬럼 순서 맞추기
    pred_df = pred_df[answer_df.columns]

    # 행 수 비교
    if len(pred_df) != len(answer_df):
        return False, f"행 수 불일치 - 예측: {len(pred_df)}, 정답: {len(answer_df)}"

    # 정렬 후 비교 (행 순서가 다를 수 있음)
    sort_cols = list(answer_df.columns)

    try:
        pred_sorted = pred_df.sort_values(by=sort_cols).reset_index(drop=True)
        answer_sorted = answer_df.sort_values(by=sort_cols).reset_index(drop=True)
    except Exception:
        # 정렬 실패 시 첫 번째 컬럼으로만 정렬
        pred_sorted = pred_df.sort_values(by=sort_cols[0]).reset_index(drop=True)
        answer_sorted = answer_df.sort_values(by=sort_cols[0]).reset_index(drop=True)

    # 값 비교
    if pred_sorted.equals(answer_sorted):
        return True, "완전 일치"

    # 차이점 찾기
    diff_count = 0
    diff_details = []

    for idx in range(len(answer_sorted)):
        for col in answer_sorted.columns:
            pred_val = pred_sorted.iloc[idx][col]
            ans_val = answer_sorted.iloc[idx][col]

            # NaN 비교
            if pd.isna(pred_val) and pd.isna(ans_val):
                continue

            # 문자열 비교 (타입 변환 후)
            if str(pred_val).strip() != str(ans_val).strip():
                diff_count += 1
                if len(diff_details) < 5:  # 최대 5개만 출력
                    diff_details.append(f"  행 {idx}, 컬럼 '{col}': 예측={pred_val}, 정답={ans_val}")

    if diff_count > 0:
        detail_str = "\n".join(diff_details)
        if diff_count > 5:
            detail_str += f"\n  ... 외 {diff_count - 5}개 차이"
        return False, f"값 불일치 ({diff_count}개)\n{detail_str}"

    return True, "완전 일치"


def check_answer(prediction_dir: str, answer_dir: str) -> Dict:
    """정답 디렉터리의 파일들이 예측 디렉터리에 일치하는지 확인합니다.

    - 정답 디렉터리에 있는 파일만 평가 대상
    - 평가 디렉터리에만 있고 정답 디렉터리에 없는 파일은 평가에서 제외

    Args:
        prediction_dir: 평가할 디렉터리 경로
        answer_dir: 정답 디렉터리 경로

    Returns:
        {
            "total": 정답 파일 수,
            "matched": 일치한 파일 수,
            "accuracy": 정확도 (0.0 ~ 1.0),
            "results": {파일명: (일치여부, 상세메시지), ...},
            "excluded": [평가 제외된 파일명들]
        }
    """
    pred_path = Path(prediction_dir)
    answer_path = Path(answer_dir)

    if not pred_path.exists():
        raise FileNotFoundError(f"예측 디렉터리가 존재하지 않습니다: {prediction_dir}")

    if not answer_path.exists():
        raise FileNotFoundError(f"정답 디렉터리가 존재하지 않습니다: {answer_dir}")

    # 정답 디렉터리의 CSV 파일들 찾기
    answer_files = list(answer_path.glob("*.csv"))
    answer_filenames = {f.name for f in answer_files}

    # 평가 디렉터리의 CSV 파일들 찾기
    pred_files = list(pred_path.glob("*.csv"))
    pred_filenames = {f.name for f in pred_files}

    # 평가에서 제외되는 파일들 (평가 디렉터리에만 있는 파일)
    excluded_files = list(pred_filenames - answer_filenames)

    if not answer_files:
        return {
            "total": 0,
            "matched": 0,
            "accuracy": 1.0,
            "results": {},
            "excluded": excluded_files,
            "message": "정답 디렉터리에 CSV 파일이 없습니다."
        }

    results = {}
    matched_count = 0

    for answer_file in answer_files:
        filename = answer_file.name
        pred_file = pred_path / filename

        if not pred_file.exists():
            results[filename] = (False, f"파일이 예측 디렉터리에 없음")
            continue

        is_match, detail = compare_csv_files(pred_file, answer_file)
        results[filename] = (is_match, detail)

        if is_match:
            matched_count += 1

    total = len(answer_files)
    accuracy = matched_count / total if total > 0 else 0.0

    return {
        "total": total,
        "matched": matched_count,
        "accuracy": accuracy,
        "results": results,
        "excluded": excluded_files
    }


def print_results(result: Dict):
    """결과를 출력합니다."""
    print("=" * 60)
    print("WorkBench Answer Check Results")
    print("=" * 60)

    print(f"\n평가 대상 파일 수: {result['total']}")
    print(f"일치 파일 수: {result['matched']}")
    print(f"정확도: {result['accuracy'] * 100:.2f}%")

    # 평가에서 제외된 파일 출력
    excluded = result.get("excluded", [])
    if excluded:
        print(f"\n평가 제외 파일 수: {len(excluded)}")
        print(f"  (정답 디렉터리에 없는 파일: {', '.join(sorted(excluded))})")

    print("\n" + "-" * 60)
    print("파일별 결과:")
    print("-" * 60)

    for filename, (is_match, detail) in result["results"].items():
        if is_match:
            print(f"\n{filename}: 일치")
        else:
            print(f"\n{filename}: 불일치")
            print(f"    사유: {detail}")

    print("\n" + "=" * 60)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    prediction_dir = sys.argv[1]
    answer_dir = sys.argv[2]

    print(f"예측 디렉터리: {prediction_dir}")
    print(f"정답 디렉터리: {answer_dir}")

    try:
        result = check_answer(prediction_dir, answer_dir)
        print_results(result)

        # 정확도가 100%가 아니면 exit code 1
        sys.exit(0 if result["accuracy"] == 1.0 else 1)

    except FileNotFoundError as e:
        print(f"오류: {e}")
        sys.exit(2)
    except Exception as e:
        print(f"예상치 못한 오류: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
