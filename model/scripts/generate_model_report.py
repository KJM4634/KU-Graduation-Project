"""
파인튜닝 모델과 베이스라인(사전학습, 파인튜닝 전) 모델을 동일한 test 서브셋들에 대해
평가하고, 비교 표 + 막대그래프를 곁들인 MODEL_REPORT.md를 생성한다.

사전 조건: model/scripts/build_eval_subsets.py 로 서브셋을 미리 만들어둘 것,
baseline 평가는 model/experiments/eval_baseline/n_baseline_<subset>.json 에 미리 저장돼
있으면 재사용(없으면 새로 실행).

사용 예:
    python generate_model_report.py --weights model/experiments/train/n_full/weights/best.pt \
        --model-name n_full --baseline-weights model/weights/yolo11n.pt
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate_model import evaluate

SUBSETS = [
    ("test_full", "전체 test"),
    ("test_public", "공개데이터 test"),
    ("test_synthetic", "합성 test (전체)"),
    ("test_synthetic_occ30", "합성 occlusion 30%"),
    ("test_synthetic_occ50", "합성 occlusion 50%"),
    ("test_synthetic_occ70", "합성 occlusion 70%"),
    ("test_own_capture", "자체촬영 test"),
]

# dataviz 스킬 참조 팔레트: 베이스라인=무채색(muted ink), 파인튜닝=categorical slot 1(blue)
# "이전 대비 개선"을 표현하는 표준 관례 (기준선은 중립, 강조 대상은 accent color)
COLOR_BASELINE = "#898781"
COLOR_FINETUNED = "#2a78d6"


def load_or_run_baseline(subset_key, baseline_weights, subsets_dir, cache_dir, imgsz, batch, device):
    cache_path = Path(cache_dir) / f"n_baseline_{subset_key}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    result = evaluate(baseline_weights, Path(subsets_dir) / f"{subset_key}.txt", imgsz, batch, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def make_bar_chart(labels, baseline_vals, finetuned_vals, ylabel, title, out_path):
    x = range(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars1 = ax.bar([i - width / 2 for i in x], baseline_vals, width, label="베이스라인(사전학습)", color=COLOR_BASELINE)
    bars2 = ax.bar([i + width / 2 for i in x], finetuned_vals, width, label="파인튜닝", color=COLOR_FINETUNED)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color="#0b0b0b")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c3c2b7")
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.grid(axis="y", alpha=0.3, color="#e1e0d9")
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="파인튜닝된 best.pt 경로")
    parser.add_argument("--model-name", required=True, help="예: n_full")
    parser.add_argument("--baseline-weights", required=True, help="예: model/weights/yolo11n.pt")
    parser.add_argument("--subsets-dir", default="model/data/splits/eval_subsets")
    parser.add_argument("--baseline-cache", default="model/experiments/eval_baseline")
    parser.add_argument("--out-dir", default="model/experiments/report_assets")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for key, label in SUBSETS:
        print(f"평가 중: {label} ({key})")
        baseline = load_or_run_baseline(key, args.baseline_weights, args.subsets_dir,
                                         args.baseline_cache, args.imgsz, args.batch, args.device)
        finetuned = evaluate(args.weights, Path(args.subsets_dir) / f"{key}.txt",
                              args.imgsz, args.batch, args.device)
        results[key] = {"label": label, "baseline": baseline, "finetuned": finetuned}

    (out_dir / f"{args.model_name}_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 차트 1: occlusion 버킷별 recall 비교
    occ_keys = ["test_synthetic_occ30", "test_synthetic_occ50", "test_synthetic_occ70"]
    make_bar_chart(
        ["30%", "50%", "70%"],
        [results[k]["baseline"]["recall"] for k in occ_keys],
        [results[k]["finetuned"]["recall"] for k in occ_keys],
        "Recall", f"{args.model_name}: Occlusion 비율별 Recall (합성 test)",
        out_dir / f"{args.model_name}_occlusion_recall.png",
    )
    make_bar_chart(
        ["30%", "50%", "70%"],
        [results[k]["baseline"]["precision"] for k in occ_keys],
        [results[k]["finetuned"]["precision"] for k in occ_keys],
        "Precision", f"{args.model_name}: Occlusion 비율별 Precision (합성 test)",
        out_dir / f"{args.model_name}_occlusion_precision.png",
    )

    # 차트 2: 소스별(공개/자체촬영) mAP50 비교
    src_keys = ["test_public", "test_own_capture"]
    make_bar_chart(
        ["공개데이터", "자체촬영"],
        [results[k]["baseline"]["map50"] for k in src_keys],
        [results[k]["finetuned"]["map50"] for k in src_keys],
        "mAP50", f"{args.model_name}: 공개 vs 자체촬영 test mAP50",
        out_dir / f"{args.model_name}_source_map50.png",
    )

    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n결과 저장: {out_dir / f'{args.model_name}_results.json'}")


if __name__ == "__main__":
    main()
