import atexit
import os
import re
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib



matplotlib.use("Agg")
# ==================== 【4-macOS/跨平台环境｜结束】 ====================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.experimental import (
    enable_iterative_imputer,  # noqa: F401  # IterativeImputer 必需
)
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "all.xlsx"
RUN_STAMP = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S_%f")
OUTPUT_ROOT = SCRIPT_DIR / "XGBoostbest_MI_CV_outputs"
OUTPUT_DIR = OUTPUT_ROOT / RUN_STAMP
IMPUTED_DATA_DIR = OUTPUT_DIR / "imputed_datasets"
MODEL_DIR = OUTPUT_DIR / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
IMPUTED_DATA_DIR.mkdir(parents=True, exist_ok=False)
MODEL_DIR.mkdir(parents=True, exist_ok=False)


def output_path(filename):
    """返回本次运行目录中的跨平台输出路径。"""
    return OUTPUT_DIR / filename


# 内容：控制台的 print、警告和搜索进度同步写入 run.log。
# 代码逻辑：Tee 保留终端可见输出，同时补足“所有输出均落入新目录”的审计要求。
class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


_original_stdout = sys.stdout
_original_stderr = sys.stderr
_run_log_handle = output_path("run.log").open("w", encoding="utf-8", buffering=1)
sys.stdout = _TeeStream(_original_stdout, _run_log_handle)
sys.stderr = _TeeStream(_original_stderr, _run_log_handle)


def _restore_console_streams():
    """程序结束时先恢复系统流，再安全关闭日志文件。"""
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
    _run_log_handle.close()


atexit.register(_restore_console_streams)


print(f"✅ 本次运行的全部结果将保存到: {OUTPUT_DIR}")

matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "STHeiti",
    "Arial Unicode MS",
    "Microsoft YaHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

if not DATA_PATH.exists():
    raise FileNotFoundError(f"未找到输入文件：{DATA_PATH}。请将 all.xlsx 与 XGBoostbest.py 放在同一目录。")

df = pd.read_excel(DATA_PATH)
df.columns = df.columns.str.strip()
print(df.columns.tolist())

TARGET_COLUMN = "Treatment type (Surgery vs. Conservative)"

f = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int)
print(df[TARGET_COLUMN].unique())

y = df[TARGET_COLUMN].reset_index(drop=True)
print(y.unique())
print(y.isnull().sum())


binary_cols = [
    "Diabetes (Yes/No)",
    "Hypertension (Yes/No)",
    "Placental abnormalities (Yes/No)",
    "Fever (Yes/No)",
    "Right lower quadrant effusion (Yes/No)",
    "Primiparity (Yes/No)",
    "Appendiceal swelling (Yes/No)",
    "Abscess (Yes/No)",
]

continous_cols = [
    "Age",
    "Gestational age",
    "Body temperature",
    "Duration of abdominal pain",
    "White blood cell (WBC) count",
    "C-reactive protein (CRP) level",
    "Neutrophil percentage",
]

# 保持原主模型的 15 个预测变量及其顺序不变。
model_features = continous_cols + binary_cols


# 内容：将原来一次、确定性的分组填补改为默认 M=20 次随机多重插补（MICE-like）。
# 代码逻辑：BayesianRidge 支持 posterior sampling；每次使用不同种子，正式分析默认形成20个完成数据集。
#          二元变量采用线性概率近似，插补值限制到 [0, 1] 后按0.5阈值化；原始已观测值不变。
N_IMPUTATIONS = int(os.getenv("XGBOOSTBEST_N_IMPUTATIONS", "20"))
N_TUNING_IMPUTATIONS = min(
    int(os.getenv("XGBOOSTBEST_N_TUNING_IMPUTATIONS", "5")),
    N_IMPUTATIONS,
)
IMPUTATION_MAX_ITER = int(os.getenv("XGBOOSTBEST_IMPUTATION_MAX_ITER", "10"))
IMPUTATION_BASE_SEED = 42
if N_IMPUTATIONS < 2:
    raise ValueError("多重插补至少需要 2 个完成数据集；默认值为 20。")
if N_TUNING_IMPUTATIONS < 1 or IMPUTATION_MAX_ITER < 1:
    raise ValueError("调参插补集数和插补迭代次数必须至少为 1。")

# 内容：插补模型严格使用原主代码最终进入 X_imputed 的同一组15个变量。
# 代码逻辑：既往阑尾炎史在原 concat 后未进入建模；这里继续排除，避免改变原研究变量框架。
AUXILIARY_BINARY_COLUMNS = binary_cols.copy()
imputation_columns = model_features.copy()

required_columns = set(model_features + AUXILIARY_BINARY_COLUMNS)
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    raise KeyError(f"输入文件缺少插补所需列：{missing_columns}")

X_raw = df[imputation_columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

# ==================== 【4-macOS/跨平台数据质控｜开始】 ====================
# 内容：只报告、不自动改写原始数据中的可疑范围和零方差列。
# 代码逻辑：迁移环境后先把可能影响插补/模型的原表问题落盘，具体修正必须由临床含义确认。
temperature_outlier_mask = X_raw["Body temperature"].notna() & ~X_raw[
    "Body temperature"
].between(34, 43)
temperature_quality_flags = df.loc[
    temperature_outlier_mask,
    ["Body temperature", TARGET_COLUMN],
].copy()
temperature_quality_flags.insert(0, "source_excel_row", temperature_quality_flags.index + 2)
constant_columns = [
    column for column in X_raw.columns if X_raw[column].dropna().nunique() <= 1
]

with pd.ExcelWriter(output_path("data_quality_flags.xlsx")) as writer:
    temperature_quality_flags.to_excel(writer, sheet_name="temperature_outliers", index=False)
    pd.DataFrame({"constant_or_empty_column": constant_columns}).to_excel(
        writer,
        sheet_name="constant_columns",
        index=False,
    )

if not temperature_quality_flags.empty:
    print(
        "⚠️ 数据质控提示：Body temperature 存在常规摄氏范围34–43之外的观测；"
        "未自动，请核对 data_quality_flags.xlsx。"
    )
if constant_columns:
    print(f"⚠️ 数据质控提示：以下列为零方差/全空列，未自动删除：{constant_columns}")
# ==================== 【4-macOS/跨平台数据质控｜结束】 ====================


def _imputation_bounds(columns):
    """按列顺序生成临床变量的插补边界；边界只约束新插补值。"""
    min_values = np.zeros(len(columns), dtype=float)
    max_values = np.full(len(columns), np.inf, dtype=float)
    for column in AUXILIARY_BINARY_COLUMNS:
        if column in columns:
            max_values[columns.index(column)] = 1.0
    if "Neutrophil percentage" in columns:
        max_values[columns.index("Neutrophil percentage")] = 100.0
    return min_values, max_values


def _postprocess_imputed_frame(values, index, columns):
    """恢复 DataFrame 结构，并把二元插补变量统一约束为 0/1。"""
    frame = pd.DataFrame(values, index=index, columns=columns)
    binary_present = [column for column in AUXILIARY_BINARY_COLUMNS if column in frame.columns]
    frame.loc[:, binary_present] = frame[binary_present].clip(0, 1).round().astype(int)
    return frame


def create_multiple_imputations(
    X_train_raw,
    X_validation_raw=None,
    n_imputations=N_IMPUTATIONS,
    seed_offset=0,
):
    """
    仅在训练数据上拟合 M 个随机 IterativeImputer；若提供验证数据，则用对应插补器转换验证数据。

    返回：训练完成集列表、验证完成集列表（可为 None）、已拟合插补器列表。
    """
    columns = list(X_train_raw.columns)
    min_values, max_values = _imputation_bounds(columns)
    train_datasets = []
    validation_datasets = [] if X_validation_raw is not None else None
    fitted_imputers = []

    for imputation_index in range(n_imputations):
        seed = IMPUTATION_BASE_SEED + seed_offset + imputation_index
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=True,
            random_state=seed,
            max_iter=IMPUTATION_MAX_ITER,
            initial_strategy="median",
            min_value=min_values,
            max_value=max_values,
        )
        train_values = imputer.fit_transform(X_train_raw[columns])
        train_frame = _postprocess_imputed_frame(train_values, X_train_raw.index, columns)
        train_datasets.append(train_frame)

        if X_validation_raw is not None:
            validation_values = imputer.transform(X_validation_raw[columns])
            validation_frame = _postprocess_imputed_frame(
                validation_values,
                X_validation_raw.index,
                columns,
            )
            validation_datasets.append(validation_frame)

        fitted_imputers.append(imputer)

    return train_datasets, validation_datasets, fitted_imputers


# 先按原研究框架保留 70%/30% 开发-测试划分；加入 stratify 以维持两类比例。
all_indices = np.arange(len(y))
train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.3,
    random_state=42,
    stratify=y,
)
X_train_raw = X_raw.iloc[train_indices].reset_index(drop=True)
X_test_raw = X_raw.iloc[test_indices].reset_index(drop=True)
y_train = y.iloc[train_indices].reset_index(drop=True)
y_test = y.iloc[test_indices].reset_index(drop=True)

# 测试集从不参与插补器拟合；每个测试完成集与同编号训练完成集严格配对。
imputed_train_datasets, imputed_test_datasets, holdout_imputers = create_multiple_imputations(
    X_train_raw,
    X_test_raw,
    n_imputations=N_IMPUTATIONS,
    seed_offset=10_000,
)

print(
    f"✅ 多重插补完成：BayesianRidge-MICE-like，M={N_IMPUTATIONS}，"
    f"每次最多迭代 {IMPUTATION_MAX_ITER} 轮"
)
# ==================== 【1-多重插补｜结束】 ====================


# =============== 5️⃣ 定义六模型及原参数网格 ===============
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 4, 5],
}

rf_pram_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}

dt_param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [3, 4, 5, 6, None],
    "min_samples_split": [2, 5, 10],
}

mlp_param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
    "activation": ["relu", "tanh"],
    "solver": ["adam", "sgd"],
    "alpha": [0.0001, 0.001],
    "learning_rate_init": [0.001, 0.01],
}

# ==================== 【4-macOS/跨平台环境｜开始】 ====================
# 代码逻辑：新版 sklearn（>=1.4）使用 estimator，旧环境仍能回退，不改变弱学习器网格原意。
ADA_ESTIMATOR_PARAMETER = (
    "estimator" if "estimator" in AdaBoostClassifier().get_params() else "base_estimator"
)
ada_param_grid = {
    ADA_ESTIMATOR_PARAMETER: [
        DecisionTreeClassifier(max_depth=1, random_state=42),
        DecisionTreeClassifier(max_depth=2, random_state=42),
    ],
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1.0],
}
# ==================== 【4-macOS/跨平台环境｜结束】 ====================

lgb_param_grid = {
    "num_leaves": [15, 31, 63],
    "max_depth": [-1, 3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [50, 100, 200],
}


# ==================== 【2-六模型统一分层5折｜开始】 ====================
# 内容：六个模型及后文 LogisticRegression 共用同一种显式分层、打乱、固定种子的 5 折。
# 代码逻辑：所有模型看到完全相同的折，结果可做病例配对比较；不再使用 sklearn 隐式 cv=5。
N_CV_SPLITS = 5
SHARED_CV = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=42)


# ==================== 【本轮5-全模型StandardScaler｜开始】 ====================
# 内容：所有预测模型统一包装为 StandardScaler -> model 的两步管道。
# 代码逻辑：传入管道的列就是该模型实际使用的特征；StandardScaler 对这些列逐列计算训练集均值和
#          标准差。树模型虽然理论上对线性缩放不敏感，但仍按本轮要求使用同一预处理框架。
STANDARDIZER_STEP = "standardizer"
MODEL_STEP = "model"


def build_standardized_model(estimator):
    """返回可被 sklearn clone 的标准化模型管道。"""
    return Pipeline(
        steps=[
            (STANDARDIZER_STEP, StandardScaler()),
            (MODEL_STEP, clone(estimator)),
        ]
    )


def _prefix_model_search_params(parameter_grid):
    """把原模型参数网格映射到 Pipeline 的 model__ 参数命名空间。"""
    return {f"{MODEL_STEP}__{key}": value for key, value in parameter_grid.items()}


def _strip_model_search_params(pipeline_parameters):
    """输出日志和Excel时恢复原参数名，保持与旧版结果表的字段结构一致。"""
    prefix = f"{MODEL_STEP}__"
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in pipeline_parameters.items()
    }


# ==================== 【本轮5-全模型StandardScaler｜结束】 ====================


model_specs = {
    "XGBoost": {
        "estimator": XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1),
        "params": param_grid,
        "search": "grid",
        "verbose": 2,
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=1),
        "params": rf_pram_grid,
        "search": "randomized",
        "n_iter": 30,
        "verbose": 1,
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "params": dt_param_grid,
        "search": "grid",
        "verbose": 2,
    },
    "MLP": {
        "estimator": MLPClassifier(max_iter=1000, random_state=42),
        "params": mlp_param_grid,
        "search": "grid",
        "verbose": 2,
    },
    "AdaBoost": {
        "estimator": AdaBoostClassifier(random_state=42),
        "params": ada_param_grid,
        "search": "grid",
        "verbose": 2,
    },
    "LightGBM": {
        "estimator": lgb.LGBMClassifier(
            objective="binary",
            random_state=42,
            n_jobs=1,
            verbosity=-1,
        ),
        "params": lgb_param_grid,
        "search": "grid",
        "verbose": 2,
    },
}


def _safe_excel_value(value):
    """把 estimator 等对象安全转换为 Excel 可写值。"""
    if value is None or isinstance(value, (str, int, float, bool, np.number)):
        return value
    return repr(value)


def tune_model_across_imputations(model_name, spec, X_datasets, y_values):
    """
    在前 N_TUNING_IMPUTATIONS 个训练完成集上分别执行相同 5 折搜索，
    再按候选参数的跨插补平均准确率选出一组共同参数。
    """
    score_rows = []
    candidate_params = None
    reference_search = None
    standardized_estimator = build_standardized_model(spec["estimator"])
    standardized_search_params = _prefix_model_search_params(spec["params"])

    for imputation_index, dataset in enumerate(X_datasets[:N_TUNING_IMPUTATIONS]):
        if spec["search"] == "randomized":
            search = RandomizedSearchCV(
                # 【本轮5-全模型StandardScaler】调参时每个内层训练折独立拟合标准化器。
                estimator=clone(standardized_estimator),
                param_distributions=standardized_search_params,
                n_iter=spec["n_iter"],
                scoring="accuracy",
                cv=SHARED_CV,
                random_state=42,
                n_jobs=-1,
                verbose=spec["verbose"],
            )
        else:
            search = GridSearchCV(
                # 【本轮5-全模型StandardScaler】参数搜索对象改为标准化管道，模型网格原意不变。
                estimator=clone(standardized_estimator),
                param_grid=standardized_search_params,
                scoring="accuracy",
                cv=SHARED_CV,
                n_jobs=-1,
                verbose=spec["verbose"],
            )

        print(
            f"\n🔎 {model_name} 参数搜索：插补集 "
            f"{imputation_index + 1}/{N_TUNING_IMPUTATIONS}"
        )
        search.fit(dataset[model_features], y_values)

        current_params = list(search.cv_results_["params"])
        if candidate_params is None:
            candidate_params = current_params
            reference_search = search
        elif [repr(item) for item in current_params] != [repr(item) for item in candidate_params]:
            raise RuntimeError(f"{model_name} 在不同插补集上的候选参数顺序不一致，无法安全汇总。")

        score_rows.append(np.asarray(search.cv_results_["mean_test_score"], dtype=float))

    score_matrix = np.vstack(score_rows)
    pooled_scores = np.nanmean(score_matrix, axis=0)
    best_index = int(np.nanargmax(pooled_scores))
    best_pipeline_params = candidate_params[best_index]
    best_params = _strip_model_search_params(best_pipeline_params)
    # 【本轮5-全模型StandardScaler】正式模板保留完整管道；对外仍报告不带 model__ 前缀的原参数名。
    best_template = clone(standardized_estimator).set_params(**best_pipeline_params)

    table_rows = []
    for candidate_index, pipeline_params in enumerate(candidate_params):
        params = _strip_model_search_params(pipeline_params)
        row = {
            "params": repr(params),
            "pooled_mean_test_score": pooled_scores[candidate_index],
            "between_imputation_score_sd": (
                np.nanstd(score_matrix[:, candidate_index], ddof=1)
                if score_matrix.shape[0] > 1
                else 0.0
            ),
        }
        for key, value in params.items():
            row[f"param_{key}"] = _safe_excel_value(value)
        table_rows.append(row)

    print(f"✅ {model_name} 跨插补汇总最优参数: {best_params}")
    # 【2-六模型统一分层交叉验证】内容：日志使用实际折数；代码逻辑：正式默认值仍为5，冒烟测试改折数时不再误报。
    print(
        f"✅ {model_name} 汇总{N_CV_SPLITS}折准确率: "
        f"{pooled_scores[best_index]:.4f}"
    )
    return {
        "template": best_template,
        "best_params": best_params,
        "best_score": pooled_scores[best_index],
        "search_table": pd.DataFrame(table_rows),
        "reference_search": reference_search,
    }


tuning_artifacts = {}
best_templates = {}
for model_name, model_spec in model_specs.items():
    artifact = tune_model_across_imputations(
        model_name,
        model_spec,
        imputed_train_datasets,
        y_train,
    )
    tuning_artifacts[model_name] = artifact
    best_templates[model_name] = artifact["template"]

# 保留原代码中的搜索器变量名，便于审核旧逻辑与新逻辑的对应关系。
grid_search = tuning_artifacts["XGBoost"]["reference_search"]
rf_grid_search = tuning_artifacts["Random Forest"]["reference_search"]
dt_gtid_search = tuning_artifacts["Decision Tree"]["reference_search"]
mlp_grid_search = tuning_artifacts["MLP"]["reference_search"]
ada_grid_search = tuning_artifacts["AdaBoost"]["reference_search"]
lgb_grid_search = tuning_artifacts["LightGBM"]["reference_search"]

# 内容：把全部调参结果写入新目录中的一个多工作表文件。
# 代码逻辑：每张表记录候选参数、跨插补平均分及插补间标准差，便于复核共同参数的来源。
with pd.ExcelWriter(output_path("hyperparameter_search_results.xlsx")) as writer:
    for model_name, artifact in tuning_artifacts.items():
        artifact["search_table"].to_excel(
            writer,
            sheet_name=model_name[:31],
            index=False,
        )

# 原 XGBoost 参数热图改为落盘，不再 plt.show() 阻塞。
xgb_search_results = tuning_artifacts["XGBoost"]["search_table"]
xgb_pivot_table = xgb_search_results.pivot_table(
    index="param_n_estimators",
    columns="param_learning_rate",
    values="pooled_mean_test_score",
)
plt.figure(figsize=(8, 6))
sns.heatmap(xgb_pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Accuracy by n_estimators & learning_rate")
plt.xlabel("learning_rate")
plt.ylabel("n_estimators")
plt.tight_layout()
plt.savefig(output_path("xgboost_parameter_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()


def build_cv_imputation_cache(X_values, y_values, cv_splits=N_CV_SPLITS):
    """
    为共同的分层5折预先构建 fold-specific 多重插补数据。

    每折的插补器只拟合该折训练行，再转换该折验证行；六模型和 LogisticRegression
    复用同一缓存，既避免验证泄漏，也保证比较时折与插补编号完全一致。
    """
    splitter = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    cache = []
    for fold_index, (fold_train_index, fold_valid_index) in enumerate(
        splitter.split(X_values, y_values),
        start=1,
    ):
        fold_train_raw = X_values.iloc[fold_train_index]
        fold_valid_raw = X_values.iloc[fold_valid_index]
        fold_train_sets, fold_valid_sets, _ = create_multiple_imputations(
            fold_train_raw,
            fold_valid_raw,
            n_imputations=N_IMPUTATIONS,
            seed_offset=fold_index * 1_000,
        )
        cache.append(
            {
                "fold": fold_index,
                "train_index": fold_train_index,
                "valid_index": fold_valid_index,
                "train_sets": fold_train_sets,
                "valid_sets": fold_valid_sets,
            }
        )
        print(f"✅ 已建立第 {fold_index}/{cv_splits} 折的 {N_IMPUTATIONS} 套无泄漏插补数据")
    return cache


cv_imputation_cache = build_cv_imputation_cache(X_raw, y, cv_splits=N_CV_SPLITS)


def _positive_class_probability(fitted_model, X_values):
    """按类别标签定位正类1，避免假定 predict_proba 的第二列永远是正类。"""
    positive_positions = np.flatnonzero(np.asarray(fitted_model.classes_) == 1)
    if len(positive_positions) != 1:
        raise ValueError(f"模型类别中未唯一找到正类1：{fitted_model.classes_}")
    return fitted_model.predict_proba(X_values)[:, int(positive_positions[0])]


def evaluate_model_cv_mi(model_template, cache, y_values, feature_list, model_name):
    """用 M 次插补 × 分层交叉验证产生逐病例 OOF 概率，并汇总 ROC/PR 曲线与波动。"""
    y_array = np.asarray(y_values)
    probability_by_imputation = np.full(
        (N_IMPUTATIONS, len(y_array)),
        np.nan,
        dtype=float,
    )
    fold_rows = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    interpolated_tprs = []
    interpolated_precisions = []

    for fold_data in cache:
        train_index = fold_data["train_index"]
        valid_index = fold_data["valid_index"]
        fold_probabilities = np.zeros((N_IMPUTATIONS, len(valid_index)), dtype=float)

        for imputation_index in range(N_IMPUTATIONS):
            fitted_model = clone(model_template)
            fitted_model.fit(
                fold_data["train_sets"][imputation_index][feature_list],
                y_array[train_index],
            )
            fold_probabilities[imputation_index] = _positive_class_probability(
                fitted_model,
                fold_data["valid_sets"][imputation_index][feature_list],
            )
            probability_by_imputation[imputation_index, valid_index] = fold_probabilities[
                imputation_index
            ]

        # 同一病例先在 M 个插补模型间平均概率，再作为这一折唯一一次验证预测。
        pooled_fold_probability = fold_probabilities.mean(axis=0)
        fold_y = y_array[valid_index]

        fold_fpr, fold_tpr, _ = roc_curve(fold_y, pooled_fold_probability)
        fold_precision, fold_recall, _ = precision_recall_curve(
            fold_y,
            pooled_fold_probability,
        )
        fold_roc_auc = auc(fold_fpr, fold_tpr)
        fold_pr_auc = auc(fold_recall, fold_precision)

        interp_tpr = np.interp(mean_fpr, fold_fpr, fold_tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interpolated_tprs.append(interp_tpr)
        interpolated_precisions.append(
            np.interp(mean_recall, fold_recall[::-1], fold_precision[::-1])
        )
        fold_rows.append(
            {
                "Model": model_name,
                "Fold": fold_data["fold"],
                "ROC-AUC": fold_roc_auc,
                "PR-AUC": fold_pr_auc,
                "N validation": len(valid_index),
            }
        )

    if np.isnan(probability_by_imputation).any():
        raise RuntimeError(f"{model_name} 的 OOF 预测不完整。")

    pooled_oof_probability = probability_by_imputation.mean(axis=0)
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_precision = np.mean(interpolated_precisions, axis=0)

    imputation_rows = []
    for imputation_index in range(N_IMPUTATIONS):
        imputation_probability = probability_by_imputation[imputation_index]
        imputation_prediction = (imputation_probability >= 0.5).astype(int)
        imputation_precision, imputation_recall, _ = precision_recall_curve(
            y_array,
            imputation_probability,
        )
        imputation_rows.append(
            {
                "Model": model_name,
                "Imputation": imputation_index + 1,
                "ROC-AUC": roc_auc_score(y_array, imputation_probability),
                "PR-AUC": auc(imputation_recall, imputation_precision),
                "Accuracy@0.5": accuracy_score(y_array, imputation_prediction),
            }
        )

    return {
        "oof_probability": pooled_oof_probability,
        "oof_prediction": (pooled_oof_probability >= 0.5).astype(int),
        "probability_by_imputation": probability_by_imputation,
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "mean_recall": mean_recall,
        "mean_precision": mean_precision,
        "mean_curve_roc_auc": auc(mean_fpr, mean_tpr),
        "mean_fold_pr_auc": float(np.mean([row["PR-AUC"] for row in fold_rows])),
        "fold_metrics": pd.DataFrame(fold_rows),
        "imputation_metrics": pd.DataFrame(imputation_rows),
    }


machine_cv_results = {}
for model_name, best_template in best_templates.items():
    print(f"\n🔁 {model_name}：开始执行 {N_IMPUTATIONS} 次插补 × {N_CV_SPLITS} 折验证")
    machine_cv_results[model_name] = evaluate_model_cv_mi(
        best_template,
        cv_imputation_cache,
        y,
        model_features,
        model_name,
    )
    result = machine_cv_results[model_name]
    # 【2-六模型统一分层交叉验证】内容：输出实际折数；代码逻辑：保证日志与当前 StratifiedKFold 配置一致。
    print(
        f"📊 {model_name} {N_CV_SPLITS}折平均 ROC AUC: "
        f"{result['mean_curve_roc_auc']:.4f}"
    )
    print(
        f"📊 {model_name} {N_CV_SPLITS}折平均 PR AUC: "
        f"{result['mean_fold_pr_auc']:.4f}"
    )
# ==================== 【2-六模型统一分层5折｜结束】 ====================


# ==================== 【1+2-多重插补训练与独立测试｜开始】 ====================
# 内容：保留原70/30测试框架，但每个算法在 M 个训练完成集上各拟合一次，测试概率取平均。
# 代码逻辑：测试集不参与插补、调参或训练；平均概率反映插补不确定性，避免纵向堆叠伪增样本量。
def fit_and_predict_holdout_mi(model_template, train_sets, test_sets, feature_list):
    probabilities = []
    fitted_models = []
    for train_set, test_set in zip(train_sets, test_sets):
        fitted_model = clone(model_template)
        fitted_model.fit(train_set[feature_list], y_train)
        probabilities.append(_positive_class_probability(fitted_model, test_set[feature_list]))
        fitted_models.append(fitted_model)
    probability_matrix = np.vstack(probabilities)
    return probability_matrix.mean(axis=0), probability_matrix, fitted_models


holdout_probabilities = {}
holdout_probabilities_by_imputation = {}
holdout_model_ensembles = {}
for model_name, best_template in best_templates.items():
    pooled_probability, probability_matrix, fitted_models = fit_and_predict_holdout_mi(
        best_template,
        imputed_train_datasets,
        imputed_test_datasets,
        model_features,
    )
    holdout_probabilities[model_name] = pooled_probability
    holdout_probabilities_by_imputation[model_name] = probability_matrix
    holdout_model_ensembles[model_name] = fitted_models
    holdout_prediction = (pooled_probability >= 0.5).astype(int)
    print(f"\n📊 {model_name} 多重插补汇总测试集准确率: {accuracy_score(y_test, holdout_prediction):.4f}")
    print(
        f"📋 {model_name} 多重插补汇总测试集分类报告:\n",
        classification_report(y_test, holdout_prediction, zero_division=0),
    )
# ==================== 【1+2-多重插补训练与独立测试｜结束】 ====================


# ==================== 【1-最终多重插补模型与数据集｜开始】 ====================
# 内容：在全数据上另行生成 M 个完成集并重训 M 个最终模型，用于保存与SHAP，不复用任何CV折模型。
# 代码逻辑：每个最终模型与同编号插补器配对；未来预测时应分别转换/预测，再平均 M 个正类概率。
full_imputed_auxiliary_datasets, _, full_imputers = create_multiple_imputations(
    X_raw,
    n_imputations=N_IMPUTATIONS,
    seed_offset=20_000,
)
full_imputed_model_datasets = [
    dataset[model_features].copy() for dataset in full_imputed_auxiliary_datasets
]

for imputation_index, dataset in enumerate(full_imputed_auxiliary_datasets, start=1):
    export_dataset = dataset.copy()
    export_dataset[TARGET_COLUMN] = y.to_numpy()
    export_dataset.to_excel(
        IMPUTED_DATA_DIR / f"imputed_dataset_{imputation_index:02d}.xlsx",
        index=False,
    )

final_model_ensembles = {}
# 【本轮5-全模型StandardScaler】记录每个最终插补模型的训练均值、标准差和方差，便于完整复核。
standardization_parameter_rows = []
for model_name, best_template in best_templates.items():
    fitted_models = []
    for imputation_index, dataset in enumerate(full_imputed_model_datasets, start=1):
        fitted_model = clone(best_template)
        fitted_model.fit(dataset, y)
        fitted_models.append(fitted_model)

        # 【本轮5-全模型StandardScaler】从已拟合管道导出标准化器参数；不改写原始或插补数据。
        fitted_scaler = fitted_model.named_steps[STANDARDIZER_STEP]
        for feature_index, feature_name in enumerate(model_features):
            standardization_parameter_rows.append(
                {
                    "Model": model_name,
                    "Imputation": imputation_index,
                    "Feature": feature_name,
                    "Training mean": fitted_scaler.mean_[feature_index],
                    "Training scale": fitted_scaler.scale_[feature_index],
                    "Training variance": fitted_scaler.var_[feature_index],
                }
            )
    final_model_ensembles[model_name] = fitted_models

    safe_model_name = re.sub(r"[^A-Za-z0-9]+", "_", model_name).strip("_").lower()
    joblib.dump(
        {
            "model_name": model_name,
            "models": fitted_models,
            "imputers": full_imputers,
            "imputation_columns": imputation_columns,
            "model_features": model_features,
            "binary_columns": AUXILIARY_BINARY_COLUMNS,
            "n_imputations": N_IMPUTATIONS,
            "probability_pooling": "arithmetic mean across matched imputer-model pairs",
            "best_params": tuning_artifacts[model_name]["best_params"],
            # 【本轮5-全模型StandardScaler】每个 models 元素均为含 StandardScaler 的完整Pipeline。
            "preprocessing": "StandardScaler fitted separately inside each matched model pipeline",
        },
        MODEL_DIR / f"{safe_model_name}_mi_ensemble.pkl",
    )

    # 内容：同时保存仅由70%开发集拟合的测试复现模型包。
    # 代码逻辑：它与 holdout_imputers 配对，可重现独立30%测试结果；全数据包继续用于最终部署。
    joblib.dump(
        {
            "model_name": model_name,
            "models": holdout_model_ensembles[model_name],
            "imputers": holdout_imputers,
            "imputation_columns": imputation_columns,
            "model_features": model_features,
            "binary_columns": AUXILIARY_BINARY_COLUMNS,
            "n_imputations": N_IMPUTATIONS,
            "training_scope": "stratified 70% development set only",
            "best_params": tuning_artifacts[model_name]["best_params"],
            # 【本轮5-全模型StandardScaler】测试复现包同样保存完整标准化管道，防止部署时漏做转换。
            "preprocessing": "StandardScaler fitted on the corresponding imputed development dataset",
        },
        MODEL_DIR / f"{safe_model_name}_holdout_evaluation_bundle.pkl",
    )

# 【本轮5-全模型StandardScaler】新增审计表：覆盖六模型×全部插补集×全部特征的缩放参数。
pd.DataFrame(standardization_parameter_rows).to_excel(
    output_path("standardization_parameters.xlsx"),
    index=False,
)
print(f"✅ 全模型标准化参数已保存为: {output_path('standardization_parameters.xlsx')}")

# 内容：保留原文件名的“单个可直接predict模型”格式，同时把MI模型包另存为 xgboost_mi_ensemble.pkl。
# 代码逻辑：旧调用方加载 best_xgb_model_acc.pkl 后仍可直接 predict；正式MI预测应使用上方完整模型包。
joblib.dump(
    holdout_model_ensembles["XGBoost"][0],
    MODEL_DIR / "best_xgb_model_acc.pkl",
)
print(f"\n✅ 六模型的多重插补最终模型包已保存到: {MODEL_DIR}")

# 保留原变量名供后文解释代码使用；这些别名仅代表第1个最终插补模型，不用于性能评价。
best_model = final_model_ensembles["XGBoost"][0]
best_rf_model = final_model_ensembles["Random Forest"][0]
best_dt_model = final_model_ensembles["Decision Tree"][0]
best_mlp_model = final_model_ensembles["MLP"][0]
best_ada_model = final_model_ensembles["AdaBoost"][0]
best_lgb_model = final_model_ensembles["LightGBM"][0]

# X_imputed 只作为汇总展示/SHAP横轴数据；训练和评价始终使用上面的20套数据，绝不使用此均值表代替MI。
X_imputed = sum(full_imputed_model_datasets) / N_IMPUTATIONS
X_imputed.loc[:, binary_cols] = X_imputed[binary_cols].round().astype(int)
# ==================== 【1-最终多重插补模型与数据集｜结束】 ====================


# =============== 六模型 ROC/PR 曲线 ===============
# ==================== 【2+3-交叉验证结果统一输出｜开始】 ====================
# 内容：原单次测试集曲线改为 M次插补汇总后的分层5折平均曲线，并全部写入新目录。
# 代码逻辑：CV图表消费同一套病例级 pooled OOF 概率；DeLong保留原意并使用独立测试概率。
model_colors = {
    "XGBoost": "darkorange",
    "Random Forest": "blue",
    "Decision Tree": "green",
    "MLP": "purple",
    "AdaBoost": "red",
    "LightGBM": "cyan",
}

plt.figure(figsize=(8, 6))
for model_name, result in machine_cv_results.items():
    plt.plot(
        result["mean_fpr"],
        result["mean_tpr"],
        color=model_colors[model_name],
        lw=2,
        label=f"{model_name} (AUC={result['mean_curve_roc_auc']:.2f})",
    )
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves: Multiple Imputation + 5-Fold CV")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(output_path("6 machines roc_curves.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
for model_name, result in machine_cv_results.items():
    plt.plot(
        result["mean_recall"],
        result["mean_precision"],
        color=model_colors[model_name],
        lw=2,
        label=f"{model_name} (AUC={result['mean_fold_pr_auc']:.2f})",
    )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curves: Multiple Imputation + 5-Fold CV")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(output_path("6 machines pr_curves.png"), dpi=300, bbox_inches="tight")
plt.close()

accuracy_scores = {
    model_name: accuracy_score(y, result["oof_prediction"])
    for model_name, result in machine_cv_results.items()
}
acc_df = pd.DataFrame(accuracy_scores.items(), columns=["Model", "Accuracy"])
plt.figure(figsize=(10, 6))
sns.barplot(
    x="Model",
    y="Accuracy",
    hue="Model",
    data=acc_df,
    palette="Set2",
    legend=False,
)
plt.ylim(0, 1)
plt.title("Model Accuracy: Multiple Imputation + 5-Fold CV")
plt.xlabel("Model")
plt.ylabel("Accuracy")
for index, value in enumerate(acc_df["Accuracy"]):
    plt.text(index, value + 0.01, f"{value:.3f}", ha="center")
plt.tight_layout()
plt.savefig(output_path("model_accuracy_comparison.png"), dpi=300, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for axis, (model_name, result) in zip(axes.flatten(), machine_cv_results.items()):
    matrix = confusion_matrix(y, result["oof_prediction"], labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    display.plot(cmap=plt.cm.Blues, ax=axis, colorbar=False)
    axis.set_title(f"{model_name} Confusion Matrix")
plt.tight_layout()
plt.savefig(output_path("model_confusion_matrices.png"), dpi=300, bbox_inches="tight")
plt.close()


def calculate_binary_metrics(y_true, y_probability, evaluation_threshold=0.5):
    """计算原指标；Youden最优点仅作描述，分类指标统一使用预先给定阈值（默认0.5）。"""
    y_true_array = np.asarray(y_true)
    probability = np.asarray(y_probability)
    curve_fpr, curve_tpr, thresholds = roc_curve(y_true_array, probability)
    roc_auc_value = auc(curve_fpr, curve_tpr)
    youden_values = curve_tpr - curve_fpr

    best_index = int(np.argmax(youden_values))
    best_threshold = float(thresholds[best_index])
    youden_j = float(youden_values[best_index])
    evaluation_threshold = float(evaluation_threshold)

    # 内容：准确率/混淆矩阵/表格统一固定0.5阈值；不再用同一OOF或测试数据选阈值后自评。
    # 代码逻辑：Youden阈值仍输出供探索，但阈值相关性能与前述图表保持一致且更少乐观偏差。
    prediction = (probability >= evaluation_threshold).astype(int)
    tn, fp, fn, _ = confusion_matrix(y_true_array, prediction, labels=[0, 1]).ravel()
    precision_values, recall_values, _ = precision_recall_curve(y_true_array, probability)

    return {
        "Youden's J": youden_j,
        "Best Threshold": best_threshold,
        "Evaluation Threshold": evaluation_threshold,
        "Accuracy": accuracy_score(y_true_array, prediction),
        "F1 Score": f1_score(y_true_array, prediction, zero_division=0),
        "Sensitivity": recall_score(y_true_array, prediction, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "PPV": precision_score(y_true_array, prediction, zero_division=0),
        "NPV": tn / (tn + fn) if (tn + fn) else 0.0,
        "AUC": roc_auc_value,
        "PR-AUC": auc(recall_values, precision_values),
    }


machine_metric_rows = {}
machine_youden_metric_rows = {}
holdout_metric_rows = {}
holdout_youden_metric_rows = {}
machine_fold_tables = []
machine_imputation_tables = []
for model_name, result in machine_cv_results.items():
    row = calculate_binary_metrics(y, result["oof_probability"], evaluation_threshold=0.5)
    row["Mean fold ROC-AUC"] = result["fold_metrics"]["ROC-AUC"].mean()
    row["Fold ROC-AUC SD"] = result["fold_metrics"]["ROC-AUC"].std(ddof=1)
    row["MI ROC-AUC SD"] = result["imputation_metrics"]["ROC-AUC"].std(ddof=1)
    machine_metric_rows[model_name] = row
    machine_youden_metric_rows[model_name] = calculate_binary_metrics(
        y,
        result["oof_probability"],
        evaluation_threshold=row["Best Threshold"],
    )

    # 独立30%测试集固定0.5阈值，避免在测试集自身选阈值造成乐观偏差。
    holdout_row = calculate_binary_metrics(
        y_test,
        holdout_probabilities[model_name],
        evaluation_threshold=0.5,
    )
    holdout_metric_rows[model_name] = holdout_row
    holdout_youden_metric_rows[model_name] = calculate_binary_metrics(
        y_test,
        holdout_probabilities[model_name],
        evaluation_threshold=holdout_row["Best Threshold"],
    )
    machine_fold_tables.append(result["fold_metrics"])
    machine_imputation_tables.append(result["imputation_metrics"])

metrics_df = pd.DataFrame(machine_metric_rows).T
youden_metrics_df = pd.DataFrame(machine_youden_metric_rows).T
holdout_metrics_df = pd.DataFrame(holdout_metric_rows).T
holdout_youden_metrics_df = pd.DataFrame(holdout_youden_metric_rows).T
cv_prediction_df = pd.DataFrame(
    {
        "source_excel_row": np.arange(len(y)) + 2,
        "observed_outcome": y.to_numpy(),
        **{
            f"{model_name}_pooled_OOF_probability": result["oof_probability"]
            for model_name, result in machine_cv_results.items()
        },
    }
)
holdout_prediction_df = pd.DataFrame(
    {
        "source_excel_row": test_indices + 2,
        "observed_outcome": y_test.to_numpy(),
        **{
            f"{model_name}_pooled_probability": probability
            for model_name, probability in holdout_probabilities.items()
        },
    }
)
# 【2-六模型统一分层交叉验证】内容：输出实际折数；代码逻辑：默认5折，测试覆盖时仍保持描述准确。
print(f"\n📊 六模型多重插补+{N_CV_SPLITS}折OOF评估指标：")
print(metrics_df.round(4))

with pd.ExcelWriter(output_path("model_evaluation_metrics.xlsx")) as writer:
    metrics_df.round(4).to_excel(writer, sheet_name="OOF_at_0.5", index=True)
    youden_metrics_df.round(4).to_excel(writer, sheet_name="OOF_at_Youden_exploratory", index=True)
    holdout_metrics_df.round(4).to_excel(writer, sheet_name="holdout_at_0.5", index=True)
    holdout_youden_metrics_df.round(4).to_excel(
        writer,
        sheet_name="holdout_Youden_exploratory",
        index=True,
    )
    cv_prediction_df.to_excel(writer, sheet_name="OOF_predictions", index=False)
    holdout_prediction_df.to_excel(writer, sheet_name="holdout_predictions", index=False)
    pd.concat(machine_fold_tables, ignore_index=True).round(4).to_excel(
        writer,
        sheet_name="fold_details",
        index=False,
    )
    pd.concat(machine_imputation_tables, ignore_index=True).round(4).to_excel(
        writer,
        sheet_name="imputation_details",
        index=False,
    )
print(f"✅ 六模型指标已保存为: {output_path('model_evaluation_metrics.xlsx')}")
# ==================== 【2+3-交叉验证结果统一输出｜结束】 ====================


# ==================== 【2-配对DeLong检验修正｜开始】 ====================
# 内容：后续比较改用独立30%测试集中同一病例的 pooled MI 概率，并修复原实现未计算协方差的问题。
# 代码逻辑：标准 paired fast-DeLong 同时输入两组测试预测，保留模型间协方差后计算双侧P值。
def compute_midrank(values):
    order = np.argsort(values)
    sorted_values = values[order]
    count = len(values)
    ranks = np.zeros(count, dtype=float)
    start = 0
    while start < count:
        end = start
        while end < count and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + end - 1)
        start = end
    restored = np.empty(count, dtype=float)
    restored[order] = ranks + 1
    return restored


def fast_delong(predictions_sorted_by_label, positive_count):
    model_count, total_count = predictions_sorted_by_label.shape
    negative_count = total_count - positive_count
    positive_examples = predictions_sorted_by_label[:, :positive_count]
    negative_examples = predictions_sorted_by_label[:, positive_count:]

    positive_ranks = np.empty((model_count, positive_count), dtype=float)
    negative_ranks = np.empty((model_count, negative_count), dtype=float)
    all_ranks = np.empty((model_count, total_count), dtype=float)
    for model_index in range(model_count):
        positive_ranks[model_index] = compute_midrank(positive_examples[model_index])
        negative_ranks[model_index] = compute_midrank(negative_examples[model_index])
        all_ranks[model_index] = compute_midrank(predictions_sorted_by_label[model_index])

    auc_values = (
        all_ranks[:, :positive_count].sum(axis=1) / positive_count / negative_count
        - (positive_count + 1.0) / (2.0 * negative_count)
    )
    v01 = (all_ranks[:, :positive_count] - positive_ranks) / negative_count
    v10 = 1.0 - (all_ranks[:, positive_count:] - negative_ranks) / positive_count
    covariance = np.atleast_2d(np.cov(v01)) / positive_count + np.atleast_2d(
        np.cov(v10)
    ) / negative_count
    return auc_values, covariance


def paired_delong_test(ground_truth, prediction_one, prediction_two):
    ground_truth = np.asarray(ground_truth, dtype=int)
    order = np.argsort(-ground_truth)
    positive_count = int(ground_truth.sum())
    stacked_predictions = np.vstack([prediction_one, prediction_two])[:, order]
    auc_values, covariance = fast_delong(stacked_predictions, positive_count)
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast.T)
    if variance <= 0 or not np.isfinite(variance):
        return auc_values, np.nan, np.nan
    z_value = float((auc_values[0] - auc_values[1]) / np.sqrt(variance))
    p_value = float(2.0 * stats.norm.sf(abs(z_value)))
    return auc_values, z_value, p_value


model_probs = {
    model_name: probability
    for model_name, probability in holdout_probabilities.items()
}
delong_rows = []
for model_one, model_two in combinations(model_probs.keys(), 2):
    auc_values, z_value, p_value = paired_delong_test(
        y_test,
        model_probs[model_one],
        model_probs[model_two],
    )
    delong_rows.append(
        {
            "Model 1": model_one,
            "Model 2": model_two,
            "AUC 1": auc_values[0],
            "AUC 2": auc_values[1],
            "Z": z_value,
            "P-Value": p_value,
            "Prediction basis": "independent holdout probability pooled across imputations",
        }
    )

delong_df = pd.DataFrame(delong_rows)
# 六个模型共有 C(6, 2)=15 次两两比较。Bonferroni 校正用于控制整组
# DeLong 检验的家族错误率（family-wise error rate）。保留原始 P 值，
# 同时报告校正后的 P 值和在 alpha=0.05 下的校正后显著性判断。
DELONG_FAMILY_ALPHA = 0.05
delong_comparison_count = len(delong_df)
delong_df["Bonferroni-adjusted P-Value"] = np.minimum(
    delong_df["P-Value"] * delong_comparison_count,
    1.0,
)
delong_df["Bonferroni significance threshold"] = (
    DELONG_FAMILY_ALPHA / delong_comparison_count
)
delong_df["Significant after Bonferroni (alpha=0.05)"] = (
    delong_df["Bonferroni-adjusted P-Value"] < DELONG_FAMILY_ALPHA
)
delong_df = delong_df[
    [
        "Model 1",
        "Model 2",
        "AUC 1",
        "AUC 2",
        "Z",
        "P-Value",
        "Bonferroni-adjusted P-Value",
        "Bonferroni significance threshold",
        "Significant after Bonferroni (alpha=0.05)",
        "Prediction basis",
    ]
]

print(
    f"\n📊 配对DeLong检验（Bonferroni校正："
    f"{delong_comparison_count}次比较，校正后阈值="
    f"{DELONG_FAMILY_ALPHA / delong_comparison_count:.6f}）："
)
print(delong_df.round(4))
delong_df.round(4).to_excel(output_path("delong_test_results.xlsx"), index=False)
print(f"✅ DeLong检验已保存为: {output_path('delong_test_results.xlsx')}")
# ==================== 【2-配对DeLong检验修正｜结束】 ====================


# =============== Basic+ Logistic Regression ===============
basic = [
    "Age",
    "Gestational age",
    "Primiparity (Yes/No)",
    "Diabetes (Yes/No)",
    "Hypertension (Yes/No)",
    "Placental abnormalities (Yes/No)",
]
symptoms = ["Fever (Yes/No)", "Body temperature", "Duration of abdominal pain"]
imaging = [
    "Appendiceal swelling (Yes/No)",
    "Abscess (Yes/No)",
    "Right lower quadrant effusion (Yes/No)",
]
labs = [
    "White blood cell (WBC) count",
    "C-reactive protein (CRP) level",
    "Neutrophil percentage",
]

feature_sets = {
    "Basic+Symptoms": basic + symptoms,
    "Basic+Imaging": basic + imaging,
    "Basic+Labs": basic + labs,
    "Basic+Symptoms+Imaging": basic + symptoms + imaging,
    "Basic+Symptoms+Labs": basic + symptoms + labs,
    "Basic+Imaging+Labs": basic + imaging + labs,
    "Basic+Symptoms+Imaging+Labs": basic + symptoms + imaging + labs,
}


# ==================== 【1+2-Logistic后续统一｜开始】 ====================
# 内容：删除原先7次单一插补/单次切分的重复训练，复用六模型同一套“M次插补×分层5折”缓存。
# 代码逻辑：每种特征组合只训练一次完整CV流程，ROC、PR、混淆矩阵与表格共用同一OOF概率。
logistic_cv_results = {}
for label, features in feature_sets.items():
    logistic_cv_results[label] = evaluate_model_cv_mi(
        # 【本轮5-全模型StandardScaler】每个特征组合、插补集和CV折均在训练部分拟合自己的标准化器。
        build_standardized_model(LogisticRegression(max_iter=5000, solver="lbfgs")),
        cv_imputation_cache,
        y,
        features,
        label,
    )

plt.figure(figsize=(8, 6))
for label, result in logistic_cv_results.items():
    plt.plot(
        result["mean_fpr"],
        result["mean_tpr"],
        lw=2,
        label=f"{label} (AUC={result['mean_curve_roc_auc']:.2f})",
    )
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic ROC: Multiple Imputation + 5-Fold CV")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(output_path("basic_logistic_roc_curves_jc.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 6))
for label, result in logistic_cv_results.items():
    plt.plot(
        result["mean_recall"],
        result["mean_precision"],
        lw=2,
        label=f"{label} (AUC={result['mean_fold_pr_auc']:.2f})",
    )
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Logistic PR: Multiple Imputation + 5-Fold CV")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(output_path("basic_logistic_pr_curves_jc.png"), dpi=300, bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(3, 3, figsize=(16, 15))
flat_axes = axes.flatten()
for axis, (label, result) in zip(flat_axes, logistic_cv_results.items()):
    matrix = confusion_matrix(y, result["oof_prediction"], labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    display.plot(cmap=plt.cm.Blues, ax=axis, colorbar=False)
    axis.set_title(f"{label} Confusion Matrix")
for axis in flat_axes[len(logistic_cv_results) :]:
    axis.axis("off")
plt.tight_layout()
plt.savefig(
    output_path("basic_logistic_confusion_matrices.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

basic_metric_rows = {}
basic_youden_metric_rows = {}
basic_fold_tables = []
basic_imputation_tables = []
for label, result in logistic_cv_results.items():
    row = calculate_binary_metrics(y, result["oof_probability"], evaluation_threshold=0.5)
    row["Mean fold ROC-AUC"] = result["fold_metrics"]["ROC-AUC"].mean()
    row["Fold ROC-AUC SD"] = result["fold_metrics"]["ROC-AUC"].std(ddof=1)
    row["MI ROC-AUC SD"] = result["imputation_metrics"]["ROC-AUC"].std(ddof=1)
    basic_metric_rows[label] = row
    basic_youden_metric_rows[label] = calculate_binary_metrics(
        y,
        result["oof_probability"],
        evaluation_threshold=row["Best Threshold"],
    )
    basic_fold_tables.append(result["fold_metrics"])
    basic_imputation_tables.append(result["imputation_metrics"])

basic_metrics_df = pd.DataFrame(basic_metric_rows).T
basic_youden_metrics_df = pd.DataFrame(basic_youden_metric_rows).T
print(f"\n📊 Logistic特征组合的多重插补+{N_CV_SPLITS}折OOF评估指标：")
print(basic_metrics_df.round(4))
with pd.ExcelWriter(output_path("basic_evaluation_metrics.xlsx")) as writer:
    basic_metrics_df.round(4).to_excel(writer, sheet_name="OOF_at_0.5", index=True)
    basic_youden_metrics_df.round(4).to_excel(
        writer,
        sheet_name="OOF_at_Youden_exploratory",
        index=True,
    )
    pd.concat(basic_fold_tables, ignore_index=True).round(4).to_excel(
        writer,
        sheet_name="fold_details",
        index=False,
    )
    pd.concat(basic_imputation_tables, ignore_index=True).round(4).to_excel(
        writer,
        sheet_name="imputation_details",
        index=False,
    )
print(f"✅ Logistic评估指标已保存为: {output_path('basic_evaluation_metrics.xlsx')}")
# ==================== 【1+2-Logistic后续统一｜结束】 ====================


# ==================== 【1-SHAP多重插补汇总｜开始】 ====================
# 代码逻辑：全局图展示最终MI集成的平均贡献；单病例图也使用平均base value、SHAP和展示特征值。
def extract_shap_2d(shap_result, X_values, explainer, class_index=1):
    """兼容 SHAP 新旧版本，提取一个类别的 (n_samples, n_features) 矩阵与基线。"""
    if isinstance(shap_result, (list, tuple)):
        arrays = [np.asarray(item) for item in shap_result]
        shap_values = arrays[class_index]
        base = np.asarray(getattr(explainer, "expected_value", 0.0))
        if base.ndim > 0 and base.size > class_index:
            base = base[class_index]
        return shap_values, base

    values = np.asarray(getattr(shap_result, "values", shap_result))
    base = np.asarray(
        getattr(shap_result, "base_values", getattr(explainer, "expected_value", 0.0))
    )
    n_samples, n_features = X_values.shape

    if values.ndim == 2:
        if base.ndim == 1 and base.size == 2:
            base = base[class_index]
        return values, np.squeeze(base)

    if values.ndim == 3 and values.shape[:2] == (n_samples, n_features):
        output_count = values.shape[2]
        if class_index >= output_count:
            raise IndexError(f"class_index {class_index} 越界，只有 {output_count} 个输出")
        selected_base = base
        if base.ndim >= 1 and base.shape[-1] == output_count:
            selected_base = base[..., class_index]
        elif base.ndim == 1 and base.size == output_count:
            selected_base = base[class_index]
        return values[..., class_index], np.squeeze(selected_base)

    if values.ndim == 3 and values.shape[1:] == (n_samples, n_features):
        output_count = values.shape[0]
        if class_index >= output_count:
            raise IndexError(f"class_index {class_index} 越界，只有 {output_count} 个输出")
        selected_base = base[class_index] if base.ndim > 0 and base.size == output_count else base
        return values[class_index], np.squeeze(selected_base)

    raise ValueError(f"无法识别 SHAP values 形状：{values.shape}")


def base_as_sample_vector(base_value, sample_count):
    """把标量或逐样本基线统一成长度为 n_samples 的向量。"""
    array = np.asarray(base_value)
    if array.ndim == 0 or array.size == 1:
        return np.full(sample_count, float(array.reshape(-1)[0]))
    squeezed = np.squeeze(array)
    if squeezed.ndim == 1 and squeezed.size == sample_count:
        return squeezed.astype(float)
    raise ValueError(f"无法把 SHAP base value 形状 {array.shape} 对齐到 {sample_count} 个样本")


class_idx = 1
rf_models = final_model_ensembles["Random Forest"]
shap_matrices = []
base_vectors = []
per_imputation_importance = []

for imputation_index, (rf_model, imputed_dataset) in enumerate(
    zip(rf_models, full_imputed_model_datasets),
    start=1,
):
    # ==================== 【本轮5-全模型StandardScaler｜SHAP适配】 ====================
    # 内容：RF 现在是 StandardScaler -> RandomForest 的 Pipeline；TreeExplainer 解释其中的树模型本体。
    # 代码逻辑：先用与该插补模型配对且仅在该完成集上拟合的标准化器转换特征，再计算SHAP。
    #          pooled SHAP仍按原特征逐列汇总，后续图横轴继续使用未标准化的临床原始量纲，便于解释。
    rf_scaler = rf_model.named_steps[STANDARDIZER_STEP]
    rf_estimator = rf_model.named_steps[MODEL_STEP]
    standardized_dataset = pd.DataFrame(
        rf_scaler.transform(imputed_dataset),
        index=imputed_dataset.index,
        columns=imputed_dataset.columns,
    )
    explainer = shap.TreeExplainer(rf_estimator)
    shap_result = explainer(standardized_dataset)
    shap_matrix, base_value = extract_shap_2d(
        shap_result,
        standardized_dataset,
        explainer,
        class_index=class_idx,
    )
    # ==================== 【本轮5-全模型StandardScaler｜SHAP适配结束】 ====================
    shap_matrices.append(shap_matrix)
    base_vectors.append(base_as_sample_vector(base_value, len(imputed_dataset)))
    per_imputation_importance.append(
        pd.DataFrame(
            {
                "Imputation": imputation_index,
                "feature": imputed_dataset.columns,
                "mean_abs_shap": np.abs(shap_matrix).mean(axis=0),
            }
        )
    )

shap_2d = np.mean(np.stack(shap_matrices, axis=0), axis=0)
base_for_plot = np.mean(np.stack(base_vectors, axis=0), axis=0)
print("shap pooled shape =", shap_2d.shape)
print("shap display data shape =", X_imputed.shape)

shap.summary_plot(
    shap_2d,
    X_imputed,
    show=False,
    max_display=X_imputed.shape[1],
)
plt.savefig(output_path("shap_summary_plot.png"), dpi=300, bbox_inches="tight")
plt.close()
print("✅ 已保存多重插补汇总 shap_summary_plot.png")


def save_pooled_sample_explanations(sample_index):
    """保存指定病例的多重插补汇总 waterfall 与 force plot。"""
    if sample_index < 0 or sample_index >= shap_2d.shape[0]:
        raise IndexError(f"sample_index {sample_index} 越界，样本数为 {shap_2d.shape[0]}")

    # ==================== 【本轮6-SHAP数值统一两位小数｜开始】 ====================
    # 内容：Force Plot 中展示的全部特征取值统一保留两位小数，并恢复SHAP默认图例位置。
    # 代码逻辑：这里只生成供绘图使用的字符串副本；模型输入、SHAP值、基线值和预测值均不做舍入，
    #          因而不会改变任何计算结果。静态图的坐标轴刻度另用两位小数格式器固定显示精度。
    force_display_values = X_imputed.iloc[sample_index].map(
        lambda value: f"{float(value):.2f}" if pd.notna(value) else "NA"
    )
    # ==================== 【本轮6-SHAP数值统一两位小数｜结束】 ====================

    explanation = shap.Explanation(
        values=shap_2d[sample_index],
        base_values=base_for_plot[sample_index],
        data=X_imputed.iloc[sample_index].to_numpy(),
        feature_names=list(X_imputed.columns),
    )
    shap.plots.waterfall(explanation, show=False)
    plt.title(f"Pooled SHAP Waterfall (class={class_idx})")
    plt.savefig(
        output_path(f"shap_waterfall_sample{sample_index}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    force_plot = shap.force_plot(
        base_for_plot[sample_index],
        shap_2d[sample_index],
        force_display_values,
        matplotlib=False,
    )
    shap.save_html(
        str(output_path(f"shap_force_plot_sample{sample_index}.html")),
        force_plot,
    )

    # ==================== 【本轮5-标准化后SHAP静态图｜开始】 ====================
    # 内容：除交互式HTML外，同时输出可直接插入paper.tex的静态Force Plot PNG。
    # 代码逻辑：静态图与HTML使用完全相同的 pooled base value、pooled SHAP 和原量纲展示值；
    #          对样本20/56额外保留论文旧文件名，替换图片时无需重画或手工截图。
    shap.force_plot(
        base_for_plot[sample_index],
        shap_2d[sample_index],
        force_display_values,
        matplotlib=True,
        show=False,
        figsize=(24, 4),
        text_rotation=25,
    )
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    static_force_path = output_path(f"shap_force_plot_sample{sample_index}.png")
    plt.savefig(static_force_path, dpi=300, bbox_inches="tight")
    legacy_force_filename = {20: "20do.png", 56: "56notdo.png"}.get(sample_index)
    if legacy_force_filename is not None:
        plt.savefig(output_path(legacy_force_filename), dpi=300, bbox_inches="tight")
    plt.close()
    # ==================== 【本轮5-标准化后SHAP静态图｜结束】 ====================
    print(
        f"✅ 已保存样本 {sample_index} 的多重插补汇总 Waterfall、Force Plot HTML与静态PNG"
    )


save_pooled_sample_explanations(20)
save_pooled_sample_explanations(56)

# 单独示例图使用不会与循环图（尤其macOS大小写不敏感路径）冲突的文件名。
plt.figure()
shap.dependence_plot("Gestational age", shap_2d, X_imputed, show=False)
plt.title("Pooled SHAP Dependence Plot: Gestational age")
plt.savefig(
    output_path("shap_selected_gestational_age.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.close()

shap_df = pd.DataFrame(shap_2d, columns=X_imputed.columns)
mi_importance_df = pd.concat(per_imputation_importance, ignore_index=True)
with pd.ExcelWriter(output_path("shap_values.xlsx")) as writer:
    shap_df.to_excel(writer, sheet_name="pooled_SHAP_values", index=False)
    mi_importance_df.to_excel(writer, sheet_name="importance_by_imputation", index=False)
print(f"✅ 已导出多重插补汇总 SHAP 值到: {output_path('shap_values.xlsx')}")

feature_importance = np.abs(shap_2d).mean(axis=0)
feat_imp_df = pd.DataFrame(
    {"feature": X_imputed.columns, "importance": feature_importance}
).sort_values("importance", ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x="importance", y="feature", data=feat_imp_df, color="#008AFA")
plt.title(f"Pooled SHAP Feature Importance (class={class_idx})")
plt.tight_layout()
plt.savefig(output_path("shap_barplot.png"), dpi=300, bbox_inches="tight")
plt.close()

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
clusters = kmeans.fit_predict(shap_2d)
shap_df_clustered = pd.DataFrame(shap_2d, columns=X_imputed.columns)
shap_df_clustered["cluster"] = clusters
shap_df_sorted = shap_df_clustered.sort_values("cluster").drop(columns="cluster")

plt.figure(figsize=(12, 8))
sns.heatmap(shap_df_sorted, cmap="RdBu_r", center=0)
plt.title(f"Pooled SHAP Heatmap with KMeans (n_clusters={n_clusters})")
plt.xlabel("Features")
plt.ylabel("Samples (sorted by cluster)")
plt.tight_layout()
plt.savefig(output_path("shap_cluster_heatmap.png"), dpi=300, bbox_inches="tight")
plt.close()

# 内容：兼容 pandas 3.0，不再依赖 GroupBy.apply 自动包含分组列的旧行为。
# 代码逻辑：先移除 cluster，再按外部 cluster 标签分组计算平均绝对SHAP。
cluster_feat_imp = (
    shap_df_clustered.drop(columns="cluster")
    .abs()
    .groupby(shap_df_clustered["cluster"])
    .mean()
    .reset_index()
    .melt(
        id_vars="cluster",
        var_name="feature",
        value_name="mean_abs_shap",
    )
)

custom_colors = {0: "#345F9C", 1: "#A6A4D4", 2: "#B3CCEC"}
plt.figure(figsize=(12, 6))
sns.barplot(
    x="mean_abs_shap",
    y="feature",
    hue="cluster",
    data=cluster_feat_imp,
    palette=custom_colors,
)
plt.title(f"Pooled SHAP Feature Importance by Cluster (class={class_idx})")
plt.xlabel("Mean |SHAP value|")
plt.ylabel("Feature")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig(output_path("shap_barplot_by_cluster.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\n✅ 随机森林跨插补汇总最优参数:", tuning_artifacts["Random Forest"]["best_params"])

for column in X_imputed.columns:
    plt.figure()
    shap.dependence_plot(
        column,
        shap_2d,
        X_imputed,
        interaction_index=None,
        show=False,
    )
    plt.title(f"Pooled SHAP Dependence Plot: {column}")
    safe_column = re.sub(r"[\\/*?:\"<>|]", "_", column)
    plt.savefig(
        output_path(f"shap_dependence_{safe_column}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
# ==================== 【1-SHAP多重插补汇总｜结束】 ====================


# ==================== 【3-运行摘要输出｜开始】 ====================
# 内容：将关键运行配置与统计口径另存文本，确保控制台之外也可追溯本次结果。
# 代码逻辑：摘要与所有图、表、模型位于同一时间戳目录，不与旧结果混放。
missingness_summary = pd.DataFrame(
    {
        "missing_count": X_raw.isna().sum(),
        "missing_rate": X_raw.isna().mean(),
    }
)
missingness_summary.to_excel(output_path("missingness_summary.xlsx"), index=True)

run_summary = f"""XGBoostbest.py 运行摘要
输入文件: {DATA_PATH}
输出目录: {OUTPUT_DIR}
样本量: {len(y)}
开发集/测试集: {len(y_train)}/{len(y_test)}（分层70%/30%）
插补模型: IterativeImputer + BayesianRidge posterior sampling（MICE-like）
插补次数: {N_IMPUTATIONS}
每次最大迭代: {IMPUTATION_MAX_ITER}
调参使用的插补集数: {N_TUNING_IMPUTATIONS}
交叉验证: StratifiedKFold(n_splits={N_CV_SPLITS}, shuffle=True, random_state=42)
CV插补原则: 每折只在训练部分拟合插补器，再转换验证部分
标准化: 所有六个机器学习模型及LogisticRegression均使用StandardScaler
标准化防泄漏原则: StandardScaler嵌入Pipeline，仅在每次对应的训练折/训练完成集上拟合
标准化范围: 模型实际输入的全部特征列（包括二元0/1变量）；插补导出表保留原量纲
多插补预测合并: 同一病例的{N_IMPUTATIONS}个正类概率取算术平均
主CV指标来源: 病例级 pooled out-of-fold probabilities
独立测试指标阈值: 固定0.5
SHAP来源: {N_IMPUTATIONS}个StandardScaler+Random Forest管道及其对应完成集的逐病例/逐特征平均
"""
output_path("run_summary.txt").write_text(run_summary, encoding="utf-8")
print(f"\n✅ 全部运行产物已保存到新目录: {OUTPUT_DIR}")
# ==================== 【3-运行摘要输出｜结束】 ====================
