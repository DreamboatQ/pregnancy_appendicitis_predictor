"""
基于六个模型的多重插补 pooled OOF 概率执行校准与决策曲线分析。

实验输入：
    XGBoostbest_MI_CV_outputs/model_evaluation_metrics.xlsx
    工作表：OOF_predictions

实验输出（全部位于 newpt 一级目录）：
    calibration_metrics.csv
    calibration_points.csv
    decision_curve_values.csv
    calibration_metrics_table.png
    calibration_curve.png
    decision_curve_analysis.png
    analysis_metadata.json

重要说明：
1. 本脚本不重新训练模型，而是严格复用主程序生成的逐患者 pooled OOF 概率。
2. 每位患者的 pooled OOF 概率已经是20个匹配插补模型预测概率的算术平均值。
3. DCA结局是观察到的手术治疗选择，因此只能作探索性解释，不能证明最佳治疗或临床获益。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==================== 【实验设置｜开始】 ====================
# 修改内容：新建独立实验路径，不复用或覆盖主程序的历史输出目录。
# 代码逻辑：所有输入、输出均相对于本脚本所在目录解析，兼容当前macOS路径。
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_WORKBOOK = SCRIPT_DIR / "XGBoostbest_MI_CV_outputs" / "model_evaluation_metrics.xlsx"
INPUT_SHEET = "OOF_predictions"
OUTPUT_DIR = SCRIPT_DIR / "newpt"

# 修改内容：按预先约定固定实验参数，避免查看结果后再调整分析设定。
# 代码逻辑：1000次分层患者bootstrap用于指标区间；校准图固定5个等人数分组；
# DCA固定展示0.05至0.95的阈值概率。
BOOTSTRAP_REPLICATES = 1_000
BOOTSTRAP_SEED = 42
N_CALIBRATION_BINS = 5
DCA_THRESHOLDS = np.linspace(0.05, 0.95, 181)
PROBABILITY_EPSILON = 1e-6

MODEL_COLUMNS = {
    "XGBoost": "XGBoost_pooled_OOF_probability",
    "Random Forest": "Random Forest_pooled_OOF_probability",
    "Decision Tree": "Decision Tree_pooled_OOF_probability",
    "MLP": "MLP_pooled_OOF_probability",
    "AdaBoost": "AdaBoost_pooled_OOF_probability",
    "LightGBM": "LightGBM_pooled_OOF_probability",
}

MODEL_COLORS = {
    "XGBoost": "#1f77b4",
    "Random Forest": "#d62728",
    "Decision Tree": "#2ca02c",
    "MLP": "#9467bd",
    "AdaBoost": "#ff7f0e",
    "LightGBM": "#17becf",
}
# ==================== 【实验设置｜结束】 ====================


def _expit(values: np.ndarray) -> np.ndarray:
    """稳定计算logistic函数，避免极端logit发生数值溢出。"""
    clipped = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    """将概率截断后转换为logit，避免0或1造成无穷值。"""
    clipped = np.clip(
        np.asarray(probabilities, dtype=float),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return np.log(clipped / (1.0 - clipped))


def calibration_in_the_large(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """
    计算calibration-in-the-large（校准截距）。

    代码逻辑：将预测logit作为系数固定为1的offset，只估计一个截距；理想值为0。
    """
    y_array = np.asarray(y_true, dtype=float)
    prediction_logit = _logit(probabilities)
    intercept = 0.0

    for _ in range(100):
        fitted_probability = _expit(prediction_logit + intercept)
        score = np.sum(y_array - fitted_probability)
        information = np.sum(fitted_probability * (1.0 - fitted_probability))
        if information <= 1e-12:
            raise RuntimeError("校准截距估计失败：有效信息量接近0。")
        update = score / information
        intercept += update
        if abs(update) < 1e-10:
            return float(intercept)

    raise RuntimeError("校准截距估计在100次迭代内未收敛。")


def calibration_slope(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """
    计算校准斜率。

    代码逻辑：无惩罚拟合 logit(Y)=alpha+beta*logit(p)，返回beta；理想值为1。
    """
    y_array = np.asarray(y_true, dtype=float)
    prediction_logit = _logit(probabilities)
    design = np.column_stack([np.ones(len(y_array)), prediction_logit])
    coefficients = np.array([0.0, 1.0], dtype=float)

    for _ in range(100):
        fitted_probability = _expit(design @ coefficients)
        weights = np.maximum(
            fitted_probability * (1.0 - fitted_probability),
            1e-10,
        )
        score = design.T @ (y_array - fitted_probability)
        information = design.T @ (weights[:, None] * design)
        information += np.eye(2) * 1e-10
        update = np.linalg.solve(information, score)
        coefficients += update
        if np.max(np.abs(update)) < 1e-10:
            return float(coefficients[1])

    raise RuntimeError("校准斜率估计在100次迭代内未收敛。")


def make_stratified_bootstrap_indices(
    y_true: np.ndarray,
    n_replicates: int,
    seed: int,
) -> np.ndarray:
    """
    生成患者层面的分层bootstrap索引。

    代码逻辑：分别在手术组和保守组内有放回抽样，保持每次抽样的两组样本量不变，
    再合并并随机打乱；六个模型共用同一批索引，保证比较具有配对性。
    """
    y_array = np.asarray(y_true, dtype=int)
    negative_indices = np.flatnonzero(y_array == 0)
    positive_indices = np.flatnonzero(y_array == 1)
    rng = np.random.default_rng(seed)
    bootstrap_indices = np.empty((n_replicates, len(y_array)), dtype=int)

    for replicate in range(n_replicates):
        sampled_negative = rng.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )
        sampled_positive = rng.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )
        sampled_indices = np.concatenate([sampled_negative, sampled_positive])
        bootstrap_indices[replicate] = rng.permutation(sampled_indices)

    return bootstrap_indices


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """返回有限bootstrap估计的2.5%和97.5%百分位数。"""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        return np.nan, np.nan
    lower, upper = np.percentile(finite_values, [2.5, 97.5])
    return float(lower), float(upper)


def equal_count_bin_labels(probabilities: np.ndarray, n_bins: int) -> np.ndarray:
    """
    为校准图建立固定数量的等人数分组。

    代码逻辑：按预测概率稳定排序后依次分组。355名患者分为5组时每组恰为71人；
    若预测概率相同，则按原始行顺序确定其所在组，从而保证结果可复现。
    """
    probabilities = np.asarray(probabilities, dtype=float)
    order = np.argsort(probabilities, kind="mergesort")
    sorted_bin_labels = np.floor(np.arange(len(probabilities)) * n_bins / len(probabilities))
    sorted_bin_labels = np.minimum(sorted_bin_labels.astype(int), n_bins - 1)
    bin_labels = np.empty(len(probabilities), dtype=int)
    bin_labels[order] = sorted_bin_labels
    return bin_labels


def validate_input(frame: pd.DataFrame) -> None:
    """检查输入完整性，防止错误列、缺失概率或越界概率进入分析。"""
    required_columns = {"observed_outcome", *MODEL_COLUMNS.values()}
    missing_columns = sorted(required_columns.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"OOF_predictions缺少必要列：{missing_columns}")

    if frame[list(required_columns)].isna().any().any():
        raise ValueError("OOF_predictions存在缺失的结局或预测概率。")

    observed_values = set(frame["observed_outcome"].astype(int).unique())
    if observed_values != {0, 1}:
        raise ValueError(f"observed_outcome必须同时包含0和1，当前为：{observed_values}")

    for model_name, column_name in MODEL_COLUMNS.items():
        probabilities = frame[column_name].to_numpy(dtype=float)
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise ValueError(f"{model_name}存在超出[0, 1]范围的预测概率。")


def compute_metrics_and_calibration_points(
    frame: pd.DataFrame,
    bootstrap_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算六模型校准指标、95%区间和5个等人数校准点。"""
    y_true = frame["observed_outcome"].to_numpy(dtype=int)
    prevalence = float(np.mean(y_true))
    null_brier = float(np.mean((prevalence - y_true) ** 2))
    bootstrap_y = y_true[bootstrap_indices]
    bootstrap_prevalence = bootstrap_y.mean(axis=1)
    bootstrap_null_brier = np.mean(
        (bootstrap_y - bootstrap_prevalence[:, None]) ** 2,
        axis=1,
    )

    metric_rows: list[dict[str, float | int | str]] = []
    calibration_rows: list[dict[str, float | int | str]] = []

    for model_name, column_name in MODEL_COLUMNS.items():
        probabilities = frame[column_name].to_numpy(dtype=float)
        brier_score = float(np.mean((probabilities - y_true) ** 2))
        scaled_brier = float(1.0 - brier_score / null_brier)
        intercept = calibration_in_the_large(y_true, probabilities)
        slope = calibration_slope(y_true, probabilities)

        bootstrap_probabilities = probabilities[bootstrap_indices]
        bootstrap_brier = np.mean(
            (bootstrap_probabilities - bootstrap_y) ** 2,
            axis=1,
        )
        bootstrap_scaled_brier = 1.0 - bootstrap_brier / bootstrap_null_brier

        bootstrap_intercepts = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=float)
        bootstrap_slopes = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=float)
        for replicate in range(BOOTSTRAP_REPLICATES):
            replicate_y = bootstrap_y[replicate]
            replicate_probability = bootstrap_probabilities[replicate]
            try:
                bootstrap_intercepts[replicate] = calibration_in_the_large(
                    replicate_y,
                    replicate_probability,
                )
                bootstrap_slopes[replicate] = calibration_slope(
                    replicate_y,
                    replicate_probability,
                )
            except (RuntimeError, np.linalg.LinAlgError, FloatingPointError):
                # 修改内容：单个异常bootstrap样本不终止全部实验。
                # 代码逻辑：保留NaN并在结果中报告成功次数，区间只使用有限估计。
                continue

        brier_lower, brier_upper = percentile_interval(bootstrap_brier)
        scaled_lower, scaled_upper = percentile_interval(bootstrap_scaled_brier)
        intercept_lower, intercept_upper = percentile_interval(bootstrap_intercepts)
        slope_lower, slope_upper = percentile_interval(bootstrap_slopes)

        metric_rows.append(
            {
                "Model": model_name,
                "N": len(y_true),
                "Events": int(np.sum(y_true)),
                "Event rate": prevalence,
                "Brier score": brier_score,
                "Brier 95% CI lower": brier_lower,
                "Brier 95% CI upper": brier_upper,
                "Null Brier": null_brier,
                "Scaled Brier": scaled_brier,
                "Scaled Brier 95% CI lower": scaled_lower,
                "Scaled Brier 95% CI upper": scaled_upper,
                "Calibration intercept": intercept,
                "Calibration intercept 95% CI lower": intercept_lower,
                "Calibration intercept 95% CI upper": intercept_upper,
                "Calibration slope": slope,
                "Calibration slope 95% CI lower": slope_lower,
                "Calibration slope 95% CI upper": slope_upper,
                "Bootstrap replicates": BOOTSTRAP_REPLICATES,
                "Successful intercept replicates": int(np.isfinite(bootstrap_intercepts).sum()),
                "Successful slope replicates": int(np.isfinite(bootstrap_slopes).sum()),
            }
        )

        bin_labels = equal_count_bin_labels(probabilities, N_CALIBRATION_BINS)
        for bin_index in range(N_CALIBRATION_BINS):
            in_bin = bin_labels == bin_index
            bin_bootstrap_observed = np.full(BOOTSTRAP_REPLICATES, np.nan, dtype=float)
            for replicate in range(BOOTSTRAP_REPLICATES):
                sampled_original_indices = bootstrap_indices[replicate]
                sampled_in_bin = bin_labels[sampled_original_indices] == bin_index
                if np.any(sampled_in_bin):
                    bin_bootstrap_observed[replicate] = np.mean(
                        y_true[sampled_original_indices[sampled_in_bin]]
                    )
            observed_lower, observed_upper = percentile_interval(bin_bootstrap_observed)
            calibration_rows.append(
                {
                    "Model": model_name,
                    "Bin": bin_index + 1,
                    "N in bin": int(np.sum(in_bin)),
                    "Mean predicted probability": float(np.mean(probabilities[in_bin])),
                    "Observed event rate": float(np.mean(y_true[in_bin])),
                    "Observed rate 95% CI lower": observed_lower,
                    "Observed rate 95% CI upper": observed_upper,
                }
            )

    return pd.DataFrame(metric_rows), pd.DataFrame(calibration_rows)


def compute_decision_curve(frame: pd.DataFrame) -> pd.DataFrame:
    """
    计算六模型、treat-all和treat-none的净获益。

    公式：Net benefit = TP/N - FP/N * threshold/(1-threshold)。
    """
    y_true = frame["observed_outcome"].to_numpy(dtype=int)
    prevalence = float(np.mean(y_true))
    dca_frame = pd.DataFrame({"Threshold probability": DCA_THRESHOLDS})

    for model_name, column_name in MODEL_COLUMNS.items():
        probabilities = frame[column_name].to_numpy(dtype=float)
        net_benefits = []
        for threshold in DCA_THRESHOLDS:
            predicted_positive = probabilities >= threshold
            true_positive = np.sum(predicted_positive & (y_true == 1))
            false_positive = np.sum(predicted_positive & (y_true == 0))
            threshold_weight = threshold / (1.0 - threshold)
            net_benefit = (
                true_positive / len(y_true)
                - false_positive / len(y_true) * threshold_weight
            )
            net_benefits.append(net_benefit)
        dca_frame[model_name] = net_benefits

    dca_frame["Treat all"] = prevalence - (1.0 - prevalence) * (
        DCA_THRESHOLDS / (1.0 - DCA_THRESHOLDS)
    )
    dca_frame["Treat none"] = 0.0
    return dca_frame


def save_metric_table_figure(metrics: pd.DataFrame, output_path: Path) -> None:
    """把核心校准指标输出为可直接审核的高分辨率表格图片。"""
    display_rows = []
    for _, row in metrics.iterrows():
        display_rows.append(
            [
                row["Model"],
                f'{row["Brier score"]:.3f} '
                f'({row["Brier 95% CI lower"]:.3f}–{row["Brier 95% CI upper"]:.3f})',
                f'{row["Null Brier"]:.3f}',
                f'{row["Scaled Brier"]:.3f} '
                f'({row["Scaled Brier 95% CI lower"]:.3f}–'
                f'{row["Scaled Brier 95% CI upper"]:.3f})',
                f'{row["Calibration intercept"]:.3f} '
                f'({row["Calibration intercept 95% CI lower"]:.3f}–'
                f'{row["Calibration intercept 95% CI upper"]:.3f})',
                f'{row["Calibration slope"]:.3f} '
                f'({row["Calibration slope 95% CI lower"]:.3f}–'
                f'{row["Calibration slope 95% CI upper"]:.3f})',
            ]
        )

    columns = [
        "Model",
        "Brier score (95% CI)",
        "Null Brier",
        "Scaled Brier (95% CI)",
        "Calibration intercept (95% CI)",
        "Calibration slope (95% CI)",
    ]
    figure, axis = plt.subplots(figsize=(16, 4.6))
    axis.axis("off")
    table = axis.table(
        cellText=display_rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.14, 0.19, 0.11, 0.20, 0.20, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.2)
    table.scale(1.0, 1.65)

    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D7DE")
        cell.set_linewidth(0.6)
        if row_index == 0:
            cell.set_facecolor("#1F4E78")
            cell.set_text_props(color="white", weight="bold")
        elif row_index % 2 == 0:
            cell.set_facecolor("#F4F7FA")
        else:
            cell.set_facecolor("white")

    axis.set_title(
        "Calibration metrics based on pooled five-fold out-of-fold predictions",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=14,
    )
    figure.text(
        0.01,
        0.04,
        "Probabilities were averaged across 20 imputations. "
        "Intervals are 95% percentile intervals from 1,000 stratified patient-level bootstrap samples.",
        fontsize=8.8,
        color="#404040",
    )
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_calibration_curve(calibration_points: pd.DataFrame, output_path: Path) -> None:
    """绘制六模型5个等人数分组的校准曲线及观察率bootstrap区间。"""
    figure, axis = plt.subplots(figsize=(9.2, 7.5))
    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        linewidth=1.5,
        color="#303030",
        label="Ideal calibration",
        zorder=1,
    )

    for model_name in MODEL_COLUMNS:
        model_points = calibration_points.loc[
            calibration_points["Model"] == model_name
        ].sort_values("Bin")
        x_values = model_points["Mean predicted probability"].to_numpy(dtype=float)
        y_values = model_points["Observed event rate"].to_numpy(dtype=float)
        lower = model_points["Observed rate 95% CI lower"].to_numpy(dtype=float)
        upper = model_points["Observed rate 95% CI upper"].to_numpy(dtype=float)
        y_error = np.vstack([y_values - lower, upper - y_values])
        axis.errorbar(
            x_values,
            y_values,
            yerr=y_error,
            marker="o",
            markersize=4.5,
            linewidth=1.7,
            capsize=2.5,
            color=MODEL_COLORS[model_name],
            label=model_name,
            zorder=2,
        )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean predicted probability", fontsize=11)
    axis.set_ylabel("Observed surgical-treatment rate", fontsize=11)
    axis.set_title(
        "Calibration curves for six models",
        fontsize=13,
        fontweight="bold",
    )
    axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axis.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_decision_curve(dca_frame: pd.DataFrame, output_path: Path) -> None:
    """绘制六模型DCA以及treat-all和treat-none参照线。"""
    figure, axis = plt.subplots(figsize=(9.5, 7.5))
    thresholds = dca_frame["Threshold probability"].to_numpy(dtype=float)

    for model_name in MODEL_COLUMNS:
        axis.plot(
            thresholds,
            dca_frame[model_name],
            color=MODEL_COLORS[model_name],
            linewidth=1.8,
            label=model_name,
        )

    axis.plot(
        thresholds,
        dca_frame["Treat all"],
        color="#555555",
        linewidth=1.6,
        linestyle="--",
        label="Treat all / classify all as surgical",
    )
    axis.plot(
        thresholds,
        dca_frame["Treat none"],
        color="#111111",
        linewidth=1.5,
        linestyle=":",
        label="Treat none / classify none as surgical",
    )

    axis.set_xlim(0.05, 0.95)
    # 修改内容：完整净获益仍写入CSV，图中限制纵轴以避免高阈值下treat-all极端负值压缩模型曲线。
    # 代码逻辑：DCA常用的可读窗口用于展示模型间差异，不删除任何原始阈值结果。
    axis.set_ylim(-0.10, 0.68)
    axis.set_xlabel("Threshold probability", fontsize=11)
    axis.set_ylabel("Net benefit", fontsize=11)
    axis.set_title(
        "Exploratory decision-curve analysis",
        fontsize=13,
        fontweight="bold",
    )
    axis.grid(color="#D9D9D9", linewidth=0.6, alpha=0.75)
    axis.legend(loc="upper right", fontsize=8.1, frameon=False, ncol=1)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    """运行全部实验并保存审计数据与图片。"""
    if not INPUT_WORKBOOK.exists():
        raise FileNotFoundError(f"未找到输入文件：{INPUT_WORKBOOK}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_excel(INPUT_WORKBOOK, sheet_name=INPUT_SHEET)
    validate_input(frame)
    y_true = frame["observed_outcome"].to_numpy(dtype=int)

    bootstrap_indices = make_stratified_bootstrap_indices(
        y_true,
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    metrics, calibration_points = compute_metrics_and_calibration_points(
        frame,
        bootstrap_indices,
    )
    dca_frame = compute_decision_curve(frame)

    # 修改内容：同时保存机器可读数据和可直接审核的高分辨率图表。
    # 代码逻辑：CSV保留完整精度，图表按论文审核需求格式化显示。
    metrics.to_csv(OUTPUT_DIR / "calibration_metrics.csv", index=False, encoding="utf-8-sig")
    calibration_points.to_csv(
        OUTPUT_DIR / "calibration_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    dca_frame.to_csv(
        OUTPUT_DIR / "decision_curve_values.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_metric_table_figure(metrics, OUTPUT_DIR / "calibration_metrics_table.png")
    save_calibration_curve(calibration_points, OUTPUT_DIR / "calibration_curve.png")
    save_decision_curve(dca_frame, OUTPUT_DIR / "decision_curve_analysis.png")

    metadata = {
        "source_workbook": str(INPUT_WORKBOOK),
        "source_sheet": INPUT_SHEET,
        "sample_size": int(len(frame)),
        "events_observed_surgery": int(np.sum(y_true)),
        "nonevents_conservative_management": int(np.sum(y_true == 0)),
        "event_rate": float(np.mean(y_true)),
        "probability_source": (
            "Pooled five-fold out-of-fold probabilities; each patient probability is the "
            "arithmetic mean across 20 matched multiply imputed datasets/models."
        ),
        "bootstrap": {
            "method": "Stratified patient-level bootstrap with replacement",
            "replicates": BOOTSTRAP_REPLICATES,
            "random_seed": BOOTSTRAP_SEED,
            "confidence_interval": "2.5th and 97.5th percentiles",
            "paired_across_models": True,
        },
        "calibration": {
            "bins": N_CALIBRATION_BINS,
            "binning": "Equal-count stable rank groups",
            "calibration_intercept": (
                "Calibration-in-the-large with prediction logit included as an offset "
                "with coefficient fixed at 1"
            ),
            "calibration_slope": (
                "Unpenalized logistic recalibration model: "
                "logit(Y)=alpha+beta*logit(predicted probability)"
            ),
            "ideal_intercept": 0.0,
            "ideal_slope": 1.0,
        },
        "dca": {
            "threshold_min": float(DCA_THRESHOLDS.min()),
            "threshold_max": float(DCA_THRESHOLDS.max()),
            "threshold_step": float(DCA_THRESHOLDS[1] - DCA_THRESHOLDS[0]),
            "reference_strategies": ["Treat all", "Treat none"],
            "interpretation_limit": (
                "Outcome is the observed surgical-treatment decision, not true surgical need, "
                "optimal treatment, or causal clinical benefit."
            ),
        },
        "models": list(MODEL_COLUMNS.keys()),
    }
    (OUTPUT_DIR / "analysis_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Calibration and DCA analysis completed.")
    print(f"Input: {INPUT_WORKBOOK}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(metrics[["Model", "Brier score", "Scaled Brier", "Calibration intercept", "Calibration slope"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
