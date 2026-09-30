from __future__ import annotations

import base64
import copy
import io
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


MODEL_PATH = Path(__file__).resolve().parent / "model" / "random_forest_mi_ensemble.pkl"
CLASSIFICATION_THRESHOLD = 0.5


@dataclass(frozen=True)
class FieldSpec:
    key: str
    feature: str
    label: str
    kind: str
    observed_min: float
    observed_max: float
    hard_min: float | None = None
    hard_max: float | None = None


FIELD_SPECS = (
    FieldSpec("age", "Age", "Maternal age", "continuous", 18.0, 44.0, 18.0, None),
    FieldSpec("gestational_age", "Gestational age", "Gestational age", "continuous", 4.0, 41.0, 0.0, 45.0),
    FieldSpec("body_temperature", "Body temperature", "Body temperature", "continuous", 36.0, 39.2, 25.0, 45.0),
    FieldSpec("pain_duration", "Duration of abdominal pain", "Abdominal pain duration", "continuous", 0.0, 1440.0, 0.0, None),
    FieldSpec("wbc", "White blood cell (WBC) count", "WBC count", "continuous", 3.0, 29.49, 0.0, None),
    FieldSpec("crp", "C-reactive protein (CRP) level", "CRP level", "continuous", 0.3, 172.22, 0.0, None),
    FieldSpec("neutrophils", "Neutrophil percentage", "Neutrophil percentage", "continuous", 5.3, 97.6, 0.0, 100.0),
    FieldSpec("diabetes", "Diabetes (Yes/No)", "Diabetes", "binary", 0.0, 1.0),
    FieldSpec("hypertension", "Hypertension (Yes/No)", "Hypertension", "binary", 0.0, 1.0),
    FieldSpec("placental_abnormalities", "Placental abnormalities (Yes/No)", "Placental abnormalities", "binary", 0.0, 0.0),
    FieldSpec("fever", "Fever (Yes/No)", "Fever", "binary", 0.0, 1.0),
    FieldSpec("rlq_effusion", "Right lower quadrant effusion (Yes/No)", "Right lower quadrant effusion", "binary", 0.0, 1.0),
    FieldSpec("primiparity", "Primiparity (Yes/No)", "Primiparity", "binary", 0.0, 1.0),
    FieldSpec("appendiceal_swelling", "Appendiceal swelling (Yes/No)", "Appendiceal swelling", "binary", 0.0, 1.0),
    FieldSpec("abscess", "Abscess (Yes/No)", "Appendiceal abscess", "binary", 0.0, 1.0),
)

SHORT_FEATURE_NAMES = {
    "Age": "Maternal age",
    "Gestational age": "Gestational age",
    "Body temperature": "Body temperature",
    "Duration of abdominal pain": "Pain duration",
    "White blood cell (WBC) count": "WBC count",
    "C-reactive protein (CRP) level": "CRP level",
    "Neutrophil percentage": "Neutrophil percentage",
    "Diabetes (Yes/No)": "Diabetes",
    "Hypertension (Yes/No)": "Hypertension",
    "Placental abnormalities (Yes/No)": "Placental abnormalities",
    "Fever (Yes/No)": "Fever",
    "Right lower quadrant effusion (Yes/No)": "RLQ effusion",
    "Primiparity (Yes/No)": "Primiparity",
    "Appendiceal swelling (Yes/No)": "Appendiceal swelling",
    "Abscess (Yes/No)": "Appendiceal abscess",
}


class ModelInputError(ValueError):
    pass


class PregnancyAppendicitisPredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {model_path}")

        bundle = joblib.load(model_path)
        self.model_name = str(bundle["model_name"])
        self.models = tuple(bundle["models"])
        self.imputers = tuple(bundle["imputers"])
        self.imputation_columns = tuple(bundle["imputation_columns"])
        self.model_features = tuple(bundle["model_features"])
        self.binary_columns = tuple(bundle["binary_columns"])
        self.n_imputations = int(bundle["n_imputations"])
        self.best_params = dict(bundle["best_params"])

        if self.model_name != "Random Forest":
            raise RuntimeError(f"Unexpected model in deployment bundle: {self.model_name}")
        if self.n_imputations != 20:
            raise RuntimeError(f"Expected 20 imputations, found {self.n_imputations}")
        if not (len(self.models) == len(self.imputers) == self.n_imputations):
            raise RuntimeError("The model and imputer counts do not match the bundle metadata.")
        if tuple(spec.feature for spec in FIELD_SPECS) != self.model_features:
            raise RuntimeError("The website field order does not match the saved model feature order.")

        self.explainers = tuple(
            shap.TreeExplainer(pipeline.named_steps["model"])
            for pipeline in self.models
        )
        self._plot_lock = threading.Lock()

    @staticmethod
    def _coerce_value(spec: FieldSpec, raw_value: Any) -> float:
        if raw_value is None or raw_value == "":
            return float("nan")

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ModelInputError(f"{spec.label} must be numeric or left unavailable.") from error

        if not math.isfinite(value):
            raise ModelInputError(f"{spec.label} must be a finite number.")
        if spec.kind == "binary" and value not in (0.0, 1.0):
            raise ModelInputError(f"{spec.label} must be Yes, No, or Unavailable.")
        if spec.hard_min is not None and value < spec.hard_min:
            raise ModelInputError(f"{spec.label} must be at least {spec.hard_min:g}.")
        if spec.hard_max is not None and value > spec.hard_max:
            raise ModelInputError(f"{spec.label} must not exceed {spec.hard_max:g}.")
        return value

    def _parse_payload(self, payload: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
        values: list[float] = []
        missing_fields: list[str] = []
        warnings: list[str] = []

        for spec in FIELD_SPECS:
            value = self._coerce_value(spec, payload.get(spec.key))
            values.append(value)
            if math.isnan(value):
                missing_fields.append(spec.label)
            elif value < spec.observed_min or value > spec.observed_max:
                warnings.append(
                    f"{spec.label} is outside the development-cohort range "
                    f"({spec.observed_min:g}–{spec.observed_max:g})."
                )

        if len(missing_fields) == len(FIELD_SPECS):
            raise ModelInputError("Enter at least one patient variable before predicting.")

        row = pd.DataFrame([values], columns=self.model_features, dtype=float)
        return row, missing_fields, warnings

    @staticmethod
    def _positive_probability(pipeline: Any, frame: pd.DataFrame) -> float:
        classes = np.asarray(pipeline.classes_)
        positive_positions = np.flatnonzero(classes == 1)
        if positive_positions.size != 1:
            raise RuntimeError("The saved model does not contain a unique positive class labelled 1.")
        return float(pipeline.predict_proba(frame)[0, int(positive_positions[0])])

    @staticmethod
    def _extract_positive_class_shap(
        explanation: Any,
        feature_count: int,
    ) -> tuple[np.ndarray, float]:
        values = np.asarray(explanation.values)
        base_values = np.asarray(explanation.base_values)

        if values.ndim == 3 and values.shape[:2] == (1, feature_count):
            positive_values = values[0, :, 1]
            if base_values.ndim == 2:
                positive_base = base_values[0, 1]
            else:
                positive_base = base_values.reshape(-1)[1]
            return np.asarray(positive_values, dtype=float), float(positive_base)

        if values.ndim == 2 and values.shape == (1, feature_count):
            return values[0].astype(float), float(base_values.reshape(-1)[0])

        raise RuntimeError(f"Unsupported SHAP output shape: {values.shape}")

    def _render_waterfall(
        self,
        shap_values: np.ndarray,
        base_value: float,
        display_values: np.ndarray,
        missing_features: set[str],
    ) -> str:
        names = [
            SHORT_FEATURE_NAMES[feature] + (" (imputed)" if feature in missing_features else "")
            for feature in self.model_features
        ]
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=np.round(display_values, 2),
            feature_names=names,
        )

        with self._plot_lock:
            plt.close("all")
            shap.plots.waterfall(explanation, max_display=len(names), show=False)
            figure = plt.gcf()
            figure.set_size_inches(10.5, 8.2)
            figure.suptitle(
                "Patient-specific pooled SHAP explanation",
                x=0.52,
                y=0.985,
                fontsize=16,
                fontweight="bold",
                color="#13243a",
            )
            figure.text(
                0.52,
                0.018,
                "Red shifts the output toward surgery; blue shifts it toward conservative management.",
                ha="center",
                fontsize=9,
                color="#5d6c80",
            )
            figure.tight_layout(rect=(0.02, 0.045, 0.98, 0.955))
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", dpi=165, bbox_inches="tight", facecolor="white")
            plt.close(figure)

        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw_row, missing_fields, warnings = self._parse_payload(payload)
        missing_features = {
            spec.feature for spec in FIELD_SPECS if spec.label in missing_fields
        }

        probabilities: list[float] = []
        shap_rows: list[np.ndarray] = []
        base_values: list[float] = []
        imputed_rows: list[np.ndarray] = []

        for imputer, pipeline, explainer in zip(
            self.imputers,
            self.models,
            self.explainers,
            strict=True,
        ):
            # IterativeImputer with posterior sampling advances its RNG during transform.
            # A per-request copy makes repeated requests deterministic and thread-safe.
            imputed_array = copy.deepcopy(imputer).transform(raw_row[list(self.imputation_columns)])
            imputed_frame = pd.DataFrame(imputed_array, columns=self.imputation_columns)
            imputed_frame.loc[:, list(self.binary_columns)] = (
                imputed_frame.loc[:, list(self.binary_columns)].clip(0, 1).round()
            )
            model_frame = imputed_frame.loc[:, list(self.model_features)]

            probabilities.append(self._positive_probability(pipeline, model_frame))
            standardized = pipeline.named_steps["standardizer"].transform(model_frame)
            explanation = explainer(standardized)
            shap_row, base_value = self._extract_positive_class_shap(
                explanation,
                len(self.model_features),
            )
            shap_rows.append(shap_row)
            base_values.append(base_value)
            imputed_rows.append(model_frame.iloc[0].to_numpy(dtype=float))

        probability = float(np.mean(probabilities))
        pooled_shap = np.mean(np.stack(shap_rows), axis=0)
        pooled_base = float(np.mean(base_values))
        pooled_display_values = np.mean(np.stack(imputed_rows), axis=0)

        reconstructed = pooled_base + float(np.sum(pooled_shap))
        if not np.isclose(reconstructed, probability, atol=1e-8):
            raise RuntimeError(
                "The pooled SHAP explanation does not reconstruct the pooled probability."
            )

        ranked_indices = np.argsort(np.abs(pooled_shap))[::-1]
        contributions = [
            {
                "feature": SHORT_FEATURE_NAMES[self.model_features[index]],
                "value": round(float(pooled_display_values[index]), 2),
                "shap": round(float(pooled_shap[index]), 6),
                "direction": "surgery" if pooled_shap[index] >= 0 else "conservative",
                "imputed": self.model_features[index] in missing_features,
            }
            for index in ranked_indices
        ]

        return {
            "probability": round(probability, 8),
            "probability_percent": round(probability * 100, 2),
            "threshold": CLASSIFICATION_THRESHOLD,
            "classification": (
                "Above the prespecified 0.50 classification threshold"
                if probability >= CLASSIFICATION_THRESHOLD
                else "Below the prespecified 0.50 classification threshold"
            ),
            "missing_fields": missing_fields,
            "warnings": warnings,
            "n_imputations": self.n_imputations,
            "base_value": round(pooled_base, 8),
            "contributions": contributions,
            "shap_plot": self._render_waterfall(
                pooled_shap,
                pooled_base,
                pooled_display_values,
                missing_features,
            ),
        }


predictor = PregnancyAppendicitisPredictor()
