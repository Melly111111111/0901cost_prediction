from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


STRICT_BUNDLE_FORMAT = "strict_price_experiment_v1"
STRICT_TARGET = "单价"

_BINARY_FEATURES = {
    "是否EO灭菌",
    "内包装材质_纸+纸",
    "内包装材质_纸+塑",
    "是否有中包装",
    "中包装材质_白卡",
}
_RATIO_FEATURES = {"粘胶配比", "涤纶配比"}
_NONNEGATIVE_FEATURES = {
    "克重g/㎡",
    "长",
    "宽（cm）",
    "层数",
    "粘胶配比",
    "涤纶配比",
    "内包装数量",
    "内包装印刷色数",
    "中包装数量",
    "中包装纸张克重",
    "中包装印刷色数",
    "外包装数量",
    "外包装印刷色数",
    "产品面积cm2",
    "理论材料重量g",
    "理论粘胶重量g",
    "理论涤纶重量g",
    "长宽比",
    "每外包装总装量",
    "每外包装总片数",
    "内包装成本分摊系数",
    "中包装成本分摊系数",
    "外包装成本分摊系数",
    "外包装数量_log1p",
}


def _is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return True
    return str(value).strip() in {"", "/", "／", "nan", "None"}


def _number(value: object, default: float = 0.0) -> float:
    if _is_empty(value):
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else default


def _ratio(value: object) -> float:
    number = _number(value)
    return number / 100.0 if number > 1 else number


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _paper_weight(material: str) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*g\s*/?\s*㎡", material, flags=re.I)
    return float(match.group(1)) if match else 0.0


def build_strict_price_row(
    input_data: dict[str, object],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Convert website inputs to the exact 29-feature training contract."""
    length = _number(input_data.get("长（cm）"))
    width = _number(input_data.get("宽（cm）"))
    layers = _number(input_data.get("层数"))
    gram = _number(input_data.get("克重g/㎡"))
    viscose = _ratio(input_data.get("粘胶配比%"))
    polyester = _ratio(input_data.get("涤纶配比%"))

    sterilization = _text(input_data.get("灭菌方式"))
    is_eo = float(sterilization == "EO灭菌")

    inner_quantity = _number(input_data.get("内包装 装量"))
    inner_material = _text(input_data.get("内包装 方式（材质）"))
    inner_colors = _number(input_data.get("内包装 印刷色数"))

    middle_quantity = _number(input_data.get("中包装 装量"))
    middle_material = _text(input_data.get("中包装 方式（材质）"))
    middle_colors = _number(input_data.get("中包装 印刷色数"))
    no_middle = middle_quantity <= 0 or middle_material in {
        "",
        "/",
        "／",
        "无",
        "无中包装",
    }
    if no_middle:
        middle_quantity = 0.0
        middle_colors = 0.0
        middle_paper_weight = 0.0
        middle_is_white_card = 0.0
    else:
        middle_paper_weight = _paper_weight(middle_material)
        middle_is_white_card = float(
            "白卡" in middle_material and "灰底白板" not in middle_material
        )

    outer_quantity = _number(input_data.get("外包装 装量"))
    outer_colors = _number(input_data.get("外包装 印刷色数"))

    area = length * width
    material_weight = gram * area * layers / 10000.0
    total_package_count = (
        middle_quantity * outer_quantity if not no_middle else outer_quantity
    )
    total_pieces = inner_quantity * total_package_count

    values = {
        "克重g/㎡": gram,
        "长": length,
        "宽（cm）": width,
        "层数": layers,
        "粘胶配比": viscose,
        "涤纶配比": polyester,
        "是否EO灭菌": is_eo,
        "内包装数量": inner_quantity,
        "内包装印刷色数": inner_colors,
        "内包装材质_纸+纸": float(inner_material == "纸+纸"),
        "内包装材质_纸+塑": float(inner_material == "纸+塑"),
        "是否有中包装": float(not no_middle),
        "中包装数量": middle_quantity,
        "中包装纸张克重": middle_paper_weight,
        "中包装材质_白卡": middle_is_white_card,
        "中包装印刷色数": middle_colors,
        "外包装数量": outer_quantity,
        "外包装印刷色数": outer_colors,
        "产品面积cm2": area,
        "理论材料重量g": material_weight,
        "理论粘胶重量g": material_weight * viscose,
        "理论涤纶重量g": material_weight * polyester,
        "长宽比": length / width if width > 0 else 0.0,
        "每外包装总装量": total_package_count,
        "每外包装总片数": total_pieces,
        "内包装成本分摊系数": 1.0 / inner_quantity if inner_quantity > 0 else 0.0,
        "中包装成本分摊系数": (
            1.0 / (inner_quantity * middle_quantity)
            if not no_middle and inner_quantity > 0 and middle_quantity > 0
            else 0.0
        ),
        "外包装成本分摊系数": 1.0 / total_pieces if total_pieces > 0 else 0.0,
        "外包装数量_log1p": float(np.log1p(max(outer_quantity, 0.0))),
    }

    missing = [column for column in feature_columns if column not in values]
    if missing:
        raise ValueError(f"网站未生成模型所需特征：{', '.join(missing)}")

    model_row = pd.DataFrame(
        [[values[column] for column in feature_columns]],
        columns=feature_columns,
        dtype=float,
    )
    if not np.isfinite(model_row.to_numpy(dtype=float)).all():
        raise ValueError("模型输入包含无效数值，请检查包装装量和产品规格。")

    derived = {
        "产品面积cm2": area,
        "理论材料重量g": material_weight,
        "每外包装总装量": total_package_count,
        "每外包装总片数": total_pieces,
        "中包装纸张克重": middle_paper_weight,
        "内包装成本分摊系数": values["内包装成本分摊系数"],
        "中包装成本分摊系数": values["中包装成本分摊系数"],
        "外包装成本分摊系数": values["外包装成本分摊系数"],
    }
    return model_row, derived


def augment_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the feature augmentation used by the locked SVR component."""
    result = frame.copy()
    positive_log_fields = [
        "内包装数量",
        "中包装数量",
        "中包装纸张克重",
        "产品面积cm2",
        "理论材料重量g",
        "每外包装总装量",
        "每外包装总片数",
        "内包装成本分摊系数",
        "中包装成本分摊系数",
        "外包装成本分摊系数",
    ]
    for field in positive_log_fields:
        result[f"log1p_{field}"] = np.log1p(
            np.clip(result[field].astype(float), 0, None)
        )
    result["内包装印刷摊销"] = result["内包装印刷色数"] * result["内包装成本分摊系数"]
    result["中包装印刷摊销"] = result["中包装印刷色数"] * result["中包装成本分摊系数"]
    result["外包装印刷摊销"] = result["外包装印刷色数"] * result["外包装成本分摊系数"]
    result["中包装纸张摊销"] = result["中包装纸张克重"] * result["中包装成本分摊系数"]
    result["EO材料交互"] = result["是否EO灭菌"] * result["理论材料重量g"]
    result["EO面积交互"] = result["是否EO灭菌"] * result["产品面积cm2"]
    result["包装印刷复杂度"] = (
        result["内包装印刷色数"]
        + result["中包装印刷色数"]
        + result["外包装印刷色数"]
    )
    return result


def validate_strict_bundle(bundle: dict[str, Any]) -> None:
    if bundle.get("bundle_format") != STRICT_BUNDLE_FORMAT:
        raise ValueError("模型包格式不是 strict_price_experiment_v1。")
    required = {
        "original_feature_columns",
        "augmented_feature_columns",
        "locked_model",
        "components",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"最优模型包缺少字段：{', '.join(missing)}")


def _predict_component(
    artifact: dict[str, Any],
    frames: dict[str, pd.DataFrame],
) -> np.ndarray:
    spec = artifact["spec"]
    frame = frames[str(spec["feature_set"])]
    if not spec.get("routed", False):
        return np.asarray(
            artifact["estimators"]["global"].predict(frame), dtype=float
        )

    routes = frames["original"]["是否有中包装"].to_numpy(dtype=int)
    prediction = np.full(len(frame), np.nan, dtype=float)
    for route in (0, 1):
        mask = routes == route
        if np.any(mask):
            prediction[mask] = artifact["estimators"][str(route)].predict(
                frame.loc[mask]
            )
    return prediction


def predict_strict_price_model(
    bundle: dict[str, Any],
    original_frame: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run the locked ensemble exactly as evaluated in the experiment."""
    validate_strict_bundle(bundle)
    original_columns = list(bundle["original_feature_columns"])
    missing = [column for column in original_columns if column not in original_frame]
    if missing:
        raise ValueError(f"模型输入缺少特征：{', '.join(missing)}")

    original = original_frame[original_columns].astype(float)
    augmented = augment_features(original)
    augmented_columns = list(bundle["augmented_feature_columns"])
    if list(augmented.columns) != augmented_columns:
        raise ValueError("增强特征顺序与训练模型不一致。")
    frames = {"original": original, "augmented": augmented}

    locked = bundle["locked_model"]
    component_names = list(locked["components"])
    weights = np.asarray(locked["weights"], dtype=float)
    if len(component_names) != len(weights) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("融合模型的组件和权重配置无效。")

    component_predictions = {
        name: _predict_component(bundle["components"][name], frames)
        for name in component_names
    }
    prediction_matrix = np.column_stack(
        [component_predictions[name] for name in component_names]
    )
    if locked.get("ensemble_mode", "arithmetic") == "geometric":
        blended = np.exp(
            np.sum(
                np.log(np.clip(prediction_matrix, 1e-12, None)) * weights,
                axis=1,
            )
        )
    else:
        blended = prediction_matrix @ weights
    calibrated = blended * float(locked["calibration_factor"])
    return np.asarray(calibrated, dtype=float), component_predictions


def local_sensitivity(
    bundle: dict[str, Any],
    original_frame: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    """Measure local prediction changes for transparent nonlinear-model context."""
    if len(original_frame) != 1:
        raise ValueError("局部敏感度分析只支持单条预测。")
    original_columns = list(bundle["original_feature_columns"])
    base_frame = original_frame[original_columns].astype(float)
    base_prediction = float(predict_strict_price_model(bundle, base_frame)[0][0])

    scale_by_feature: dict[str, float] = {}
    for component in bundle["components"].values():
        if component["spec"].get("feature_set") != "augmented":
            continue
        estimator = component["estimators"].get("global")
        pipeline = getattr(estimator, "regressor_", None)
        if pipeline is None or "scale" not in pipeline.named_steps:
            continue
        scales = np.asarray(pipeline.named_steps["scale"].scale_, dtype=float)
        augmented_columns = list(bundle["augmented_feature_columns"])
        scale_by_feature.update(dict(zip(augmented_columns, scales)))
        break

    records: list[dict[str, object]] = []
    for feature in original_columns:
        value = float(base_frame.iloc[0][feature])
        candidates: list[float]
        if feature in _BINARY_FEATURES:
            candidates = [1.0 - value]
        else:
            reference_scale = abs(scale_by_feature.get(feature, value))
            step = max(abs(value) * 0.05, reference_scale * 0.05, 1e-6)
            lower = value - step
            upper = value + step
            if feature in _NONNEGATIVE_FEATURES:
                lower = max(0.0, lower)
            if feature in _RATIO_FEATURES:
                lower = min(max(lower, 0.0), 1.0)
                upper = min(max(upper, 0.0), 1.0)
            candidates = [lower, upper]

        best_change = 0.0
        best_value = value
        for candidate in candidates:
            if np.isclose(candidate, value):
                continue
            changed = base_frame.copy()
            changed.loc[changed.index[0], feature] = candidate
            prediction = float(predict_strict_price_model(bundle, changed)[0][0])
            change = prediction - base_prediction
            if abs(change) > abs(best_change):
                best_change = change
                best_value = candidate
        records.append(
            {
                "特征": feature,
                "当前输入值": value,
                "扰动后值": best_value,
                "最大预测变化": best_change,
                "绝对敏感度": abs(best_change),
                "影响方向": "推高预测" if best_change >= 0 else "降低预测",
            }
        )
    return (
        pd.DataFrame(records)
        .sort_values("绝对敏感度", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
