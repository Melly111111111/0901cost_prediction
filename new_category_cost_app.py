from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    from sqlalchemy import create_engine, text
except ImportError as exc:
    create_engine = None
    text = None
    SQLALCHEMY_IMPORT_ERROR = exc
else:
    SQLALCHEMY_IMPORT_ERROR = None

from price_model_runtime import (
    STRICT_BUNDLE_FORMAT,
    STRICT_TARGET,
    build_strict_price_row,
    local_sensitivity,
    predict_strict_price_model,
    validate_strict_bundle,
)


APP_DIR = Path(__file__).resolve().parent
SEARCHABLE_SELECT_COMPONENT = components.declare_component(
    "searchable_select",
    path=str(APP_DIR / "components" / "searchable_select"),
)
NEW_CATEGORY_BUNDLE_FORMAT = "new_category_experiment_v1"
MODEL_PATH = APP_DIR / "new_category_cost_model.joblib"
DB_PATH = APP_DIR / "new_category_cost_feedback.db"
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if not DATABASE_URL:
    with suppress(Exception):
        DATABASE_URL = str(st.secrets.get("DATABASE_URL", "")).strip()


def normalize_database_url(database_url: str) -> str:
    """Use psycopg 3 for standard PostgreSQL URLs on Streamlit Cloud."""
    prefixes = (
        ("postgresql+psycopg2://", "postgresql+psycopg://"),
        ("postgresql://", "postgresql+psycopg://"),
        ("postgres://", "postgresql+psycopg://"),
    )
    for source_prefix, target_prefix in prefixes:
        if database_url.startswith(source_prefix):
            return target_prefix + database_url[len(source_prefix) :]
    return database_url


DATABASE_URL = normalize_database_url(DATABASE_URL)
QUOTE_PARAMETER_FILE = "产品配置数据.xlsx"
RAW_HISTORY_FILE = "新建 Microsoft Excel 工作表.xlsx"
MIDDLE_PRINT_COLOR_FEATURE = "中包装印刷色数"
MIDDLE_PRINT_COLOR_LEVELS = np.arange(5, dtype=float)
# Training-data-derived cumulative print increments per allocation unit.
MIDDLE_PRINT_INCREMENT_PER_ALLOCATION_UNIT = np.array(
    [0.0, 0.0572439785, 0.0784875901, 0.1087624229, 0.1164446875],
    dtype=float,
)
OUTER_PRINT_COLOR_FEATURE = "外包装印刷色数"
OUTER_PRINT_COLOR_LEVELS = np.arange(5, dtype=float)
# Use the same cumulative print-cost basis as middle packaging, allocated per PCS.
OUTER_PRINT_INCREMENT_PER_ALLOCATION_UNIT = np.array(
    [0.0, 0.0572439785, 0.0784875901, 0.1087624229, 0.1164446875],
    dtype=float,
)
STERILIZATION_OPTIONS = ["EO灭菌", "EO预处理", "伽马灭菌", "蒸汽灭菌", "无灭菌"]
LOW_SAMPLE_STERILIZATION = {"EO预处理", "蒸汽灭菌", "伽马灭菌"}
MIDDLE_MATERIAL_OPTIONS = [
    "中盒-255g/㎡白卡",
    "中盒-275g/㎡白卡",
    "中盒-300g/㎡白卡",
    "中盒-350g/㎡白卡",
    "中盒-250g/㎡灰底白板",
    "中盒-300g/㎡灰底白板",
    "中盒-350g/㎡灰底白板",
    "中盒-400g/㎡灰底白板",
    "E瓦楞",
    "中箱",
    "/",
    "塑袋（贴标）",
]
LOW_SAMPLE_MIDDLE_MATERIALS = {"E瓦楞", "中盒-E瓦楞", "中箱", "塑袋", "塑袋（贴标）"}

NEW_CATEGORY_BINARY_FEATURES = {
    "是否EO灭菌",
    "内包装材质_纸+纸",
    "内包装材质_纸+塑",
    "是否有中包装",
    "中包装材质_白卡",
    "灭菌方式_EO预处理",
    "灭菌方式_蒸汽灭菌",
    "灭菌方式_伽马灭菌",
    "中包装材质_E瓦楞",
    "中包装材质_塑袋（贴标）",
    "中包装材质_中箱",
    "中包装纸张克重_适用",
    "中包装纸张克重_缺失",
}
NEW_CATEGORY_RATIO_FEATURES = {"粘胶配比", "涤纶配比"}

# Show sensitivity using the original website inputs instead of engineered model columns.
NEW_CATEGORY_INPUT_FEATURE_GROUPS = {
    "克重g/㎡": ("克重g/㎡",),
    "长（cm）": ("长",),
    "宽（cm）": ("宽（cm）",),
    "层数": ("层数",),
    "粘胶配比%": ("粘胶配比",),
    "涤纶配比%": ("涤纶配比",),
    "灭菌方式": (
        "是否EO灭菌",
        "灭菌方式_EO预处理",
        "灭菌方式_蒸汽灭菌",
        "灭菌方式_伽马灭菌",
    ),
    "内包装 装量": ("内包装数量",),
    "内包装 方式（材质）": (
        "内包装材质_纸+纸",
        "内包装材质_纸+塑",
    ),
    "内包装 印刷色数": ("内包装印刷色数",),
    "中包装 装量": ("中包装数量",),
    "中包装 方式（材质）": (
        "中包装材质_白卡",
        "中包装材质_E瓦楞",
        "中包装材质_塑袋（贴标）",
        "中包装材质_中箱",
    ),
    "中包装 印刷色数": ("中包装印刷色数",),
    "外包装 装量": ("外包装数量",),
    "外包装 印刷色数": ("外包装印刷色数",),
}

INPUT_PARAMETER_COLUMNS = [
    "产品标准配置",
    "包装标准配置",
    "长*宽（cm）",
    "长（cm）",
    "宽（cm）",
    "层数",
    "克重g/㎡",
    "粘胶配比%",
    "涤纶配比%",
    "灭菌方式",
    "内包装 装量",
    "内包装 方式（材质）",
    "内包装 印刷色数",
    "中包装 装量",
    "中包装 方式（材质）",
    "中包装 印刷色数",
    "外包装 装量",
    "外包装 方式（材质）",
    "外包装 印刷色数",
]


PAGE_CSS = """
<style>
    .stApp { background-color: #f7f9fb; }
    h1 {
        color: #0f3d5e;
        border-bottom: 2px solid #0f3d5e;
        padding-bottom: 10px;
        text-align: center;
        letter-spacing: 0;
    }
    .section-header {
        background-color: #eaf2f8;
        color: #0f3d5e;
        font-weight: 700;
        padding: 8px 12px;
        border-left: 5px solid #0f3d5e;
        margin: 12px 0 16px 0;
        text-align: center;
    }
    .price-box {
        background-color: white;
        border: 2px solid #0f3d5e;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-bottom: 0;
    }
    .price-box h2 {
        color: #0f3d5e;
        margin: 8px 0 0 0;
        font-size: 34px;
        letter-spacing: 0;
    }
    .judgment-box {
        padding: 15px;
        border-radius: 8px;
        margin-top: 16px;
        text-align: center;
        font-weight: 700;
    }
    .judgment-success {
        background-color: #dff3e4;
        color: #145a32;
        border: 1px solid #b7dfc3;
    }
    .judgment-warning {
        background-color: #fff2cf;
        color: #7a5200;
        border: 1px solid #f5d889;
    }
    .judgment-danger {
        background-color: #fde2df;
        color: #8f1d14;
        border: 1px solid #efb4ae;
    }
    .history-band {
        background: #edf4fb;
        border-left: 6px solid #0f4c8a;
        height: 56px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 8px 0 18px 0;
        font-weight: 800;
        color: #0f4c8a;
        font-size: 20px;
        letter-spacing: 0;
    }
    .history-band .history-emoji {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 8px;
        font-size: 24px;
        line-height: 1;
    }
    .analysis-card {
        background: white;
        border: 1px solid #dde4eb;
        border-radius: 14px;
        padding: 22px 22px 18px 22px;
        min-height: 520px;
        box-shadow: 0 1px 4px rgba(15, 61, 94, 0.04);
    }
    .analysis-card h3 {
        margin: 0 0 14px 0;
        color: #0f4c8a;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: 0;
    }
    .analysis-card p, .analysis-card li {
        color: #495057;
        font-size: 16px;
        line-height: 1.8;
    }
    .analysis-card ul {
        margin: 10px 0 0 20px;
    }
    .analysis-divider {
        border-top: 1px solid #e2e8ef;
        margin: 18px 0 16px 0;
    }
    .analysis-rating-title {
        color: #0f4c8a;
        font-size: 24px;
        font-weight: 800;
        margin: 0 0 12px 0;
    }
    .analysis-rating {
        font-size: 18px;
        line-height: 1.8;
    }
    .rating-tag {
        font-weight: 800;
    }
    .rating-low {
        color: #1f9d55;
    }
    .rating-mid {
        color: #d97706;
    }
    .rating-high {
        color: #dc2626;
    }
    div.stButton > button[kind="primary"] {
        background-color: #f07c22 !important;
        border-color: #f07c22 !important;
        color: white !important;
        width: 100%;
        height: 48px;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: white;
        padding: 12px;
        border: 1px solid #e1e7ed;
        border-radius: 8px;
    }
</style>
"""


PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    input_json TEXT NOT NULL,
    derived_json TEXT NOT NULL,
    prediction REAL NOT NULL,
    log_prediction REAL NOT NULL,
    avg_cost REAL,
    median_cost REAL,
    source TEXT NOT NULL
)
"""
FEEDBACK_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    predicted_cost REAL NOT NULL,
    actual_cost REAL NOT NULL,
    error REAL NOT NULL,
    error_rate REAL NOT NULL,
    reviewer TEXT,
    note TEXT,
    input_json TEXT NOT NULL,
    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
)
"""
FEEDBACK_SCHEMA_POSTGRES = FEEDBACK_SCHEMA_SQLITE.replace(
    "id INTEGER PRIMARY KEY AUTOINCREMENT",
    "id BIGSERIAL PRIMARY KEY",
)


def using_remote_database() -> bool:
    return bool(DATABASE_URL)


@st.cache_resource
def remote_database_engine():
    if not DATABASE_URL:
        return None
    if create_engine is None or text is None:
        raise RuntimeError(
            "已配置 DATABASE_URL，但 SQLAlchemy 导入失败。"
            "请确认已上传最新 requirements.txt 并重新部署。"
            f"原始错误：{SQLALCHEMY_IMPORT_ERROR}"
        ) from SQLALCHEMY_IMPORT_ERROR
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def initialize_database() -> None:
    if using_remote_database():
        engine = remote_database_engine()
        with engine.begin() as conn:
            conn.exec_driver_sql(PREDICTIONS_SCHEMA)
            conn.exec_driver_sql(FEEDBACK_SCHEMA_POSTGRES)
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(PREDICTIONS_SCHEMA)
        conn.execute(FEEDBACK_SCHEMA_SQLITE)
        conn.commit()


def execute_database(sql: str, params: dict[str, object]) -> None:
    initialize_database()
    if using_remote_database():
        with remote_database_engine().begin() as conn:
            conn.execute(text(sql), params)
        return

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(sql, params)
        conn.commit()


def read_database(sql: str, params: dict[str, object]) -> pd.DataFrame:
    initialize_database()
    if using_remote_database():
        with remote_database_engine().connect() as conn:
            result = conn.execute(text(sql), params)
            return pd.DataFrame(result.fetchall(), columns=result.keys())

    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def save_prediction(
    input_data: dict[str, object],
    derived_data: dict[str, object],
    prediction: float,
    log_prediction: float,
    avg_cost: float,
    median_cost: float,
    source: str,
) -> str:
    prediction_id = str(uuid.uuid4())
    execute_database(
        """
        INSERT INTO predictions
        (id, created_at, input_json, derived_json, prediction, log_prediction,
         avg_cost, median_cost, source)
        VALUES (:id, :created_at, :input_json, :derived_json, :prediction,
                :log_prediction, :avg_cost, :median_cost, :source)
        """,
        {
            "id": prediction_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_json": json.dumps(input_data, ensure_ascii=False),
            "derived_json": json.dumps(derived_data, ensure_ascii=False),
            "prediction": prediction,
            "log_prediction": log_prediction,
            "avg_cost": avg_cost,
            "median_cost": median_cost,
            "source": source,
        },
    )
    return prediction_id


def save_feedback(
    prediction_id: str,
    predicted_cost: float,
    actual_cost: float,
    reviewer: str,
    note: str,
    input_data: dict[str, object],
) -> None:
    error = actual_cost - predicted_cost
    error_rate = error / max(abs(actual_cost), 1e-12)
    execute_database(
        """
        INSERT INTO feedback
        (prediction_id, created_at, predicted_cost, actual_cost, error, error_rate,
         reviewer, note, input_json)
        VALUES (:prediction_id, :created_at, :predicted_cost, :actual_cost,
                :error, :error_rate, :reviewer, :note, :input_json)
        """,
        {
            "prediction_id": prediction_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "predicted_cost": predicted_cost,
            "actual_cost": actual_cost,
            "error": error,
            "error_rate": error_rate,
            "reviewer": reviewer,
            "note": note,
            "input_json": json.dumps(input_data, ensure_ascii=False),
        },
    )


def read_feedback(limit: int = 200) -> pd.DataFrame:
    return read_database(
        """
        SELECT created_at AS 录入时间,
               prediction_id AS 预测ID,
               predicted_cost AS 预测成本,
               actual_cost AS 人工确认成本,
               error AS 误差,
               error_rate AS 误差率,
               reviewer AS 确认人,
               note AS 备注
        FROM feedback
        ORDER BY id DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )


def parse_json_dict(value: object) -> dict[str, object]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def expand_prediction_inputs(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return records.drop(columns=["input_json"], errors="ignore")

    input_values = [parse_json_dict(value) for value in records["input_json"]]
    input_df = pd.DataFrame(input_values)
    for col in INPUT_PARAMETER_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = None

    extra_cols = [col for col in input_df.columns if col not in INPUT_PARAMETER_COLUMNS]
    input_df = input_df[INPUT_PARAMETER_COLUMNS + extra_cols].reset_index(drop=True)

    base_df = records.drop(columns=["input_json"]).reset_index(drop=True)
    prefix_cols = [col for col in ["预测时间", "预测ID"] if col in base_df.columns]
    suffix_cols = [col for col in base_df.columns if col not in prefix_cols]
    return pd.concat([base_df[prefix_cols], input_df, base_df[suffix_cols]], axis=1)


def read_predictions(limit: int = 200) -> pd.DataFrame:
    records = read_database(
        """
        SELECT created_at AS 预测时间,
               id AS 预测ID,
               input_json,
               prediction AS 预测成本,
               avg_cost AS 历史均值,
               median_cost AS 历史中位数,
               source AS 来源
        FROM predictions
        ORDER BY created_at DESC
        LIMIT :limit
        """,
        {"limit": limit},
    )
    return expand_prediction_inputs(records)


def validate_new_category_bundle(bundle: dict[str, object]) -> None:
    if bundle.get("bundle_format") != NEW_CATEGORY_BUNDLE_FORMAT:
        raise ValueError(f"模型包格式不是 {NEW_CATEGORY_BUNDLE_FORMAT}。")
    required = {
        "old_feature_columns",
        "original_feature_columns",
        "augmented_feature_columns",
        "fitted_model",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise ValueError(f"新类别模型包缺少字段：{', '.join(missing)}")
    original_columns = list(bundle["original_feature_columns"])
    if len(original_columns) != 37:
        raise ValueError(f"新类别模型应包含37个原始特征，实际为{len(original_columns)}个。")
    fitted_model = bundle["fitted_model"]
    if not isinstance(fitted_model, dict):
        raise ValueError("新类别模型缺少已拟合模型结构。")
    components = list(fitted_model.get("components", []))
    fitted = fitted_model.get("fitted", {})
    if not components or not isinstance(fitted, dict) or any(name not in fitted for name in components):
        raise ValueError("新类别模型的融合组件不完整。")


@st.cache_resource
def load_model_bundle() -> dict[str, object]:
    candidate_names = [
        os.environ.get("NEW_CATEGORY_MODEL_FILE"),
        MODEL_PATH.name,
    ]
    candidates: list[Path] = []
    for name in candidate_names:
        if not name:
            continue
        path = Path(name)
        if not path.is_absolute():
            path = APP_DIR / path
        candidates.append(path)
    seen = set()
    load_errors: list[str] = []
    for path in candidates:
        normalized = path.resolve() if path.exists() else path
        if normalized in seen:
            continue
        seen.add(normalized)
        if not path.exists():
            load_errors.append(f"{path.name}: 文件不存在")
            continue
        try:
            bundle = joblib.load(path)
        except Exception as exc:
            load_errors.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if isinstance(bundle, dict) and bundle.get("bundle_format") == NEW_CATEGORY_BUNDLE_FORMAT:
            try:
                validate_new_category_bundle(bundle)
            except ValueError as exc:
                load_errors.append(f"{path.name}: {exc}")
                continue
            bundle["_model_path"] = str(path)
            return bundle
        if isinstance(bundle, dict) and bundle.get("bundle_format") == STRICT_BUNDLE_FORMAT:
            try:
                validate_strict_bundle(bundle)
            except ValueError as exc:
                load_errors.append(f"{path.name}: {exc}")
                continue
            bundle["_model_path"] = str(path)
            return bundle
        if isinstance(bundle, dict) and (
            "model_pipeline" in bundle or "formula_pipeline" in bundle
        ):
            bundle["_model_path"] = str(path)
            return bundle
        load_errors.append(f"{path.name}: 文件已读取，但不是可用的新类别模型包")

    available = [str(p.relative_to(APP_DIR)) for p in APP_DIR.rglob("*.joblib")]
    st.error("未找到可用的新类别 joblib 模型文件。")
    st.write("云端当前能看到的 joblib 文件：", available or "没有找到任何 .joblib 文件")
    st.write("加载尝试记录：")
    st.code("\n".join(load_errors) if load_errors else "没有加载记录")
    st.stop()


def clean_excel_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
    return df


def quote_parameter_path() -> Path | None:
    direct = APP_DIR / QUOTE_PARAMETER_FILE
    return direct if direct.exists() else None


def read_quote_sheet(path: Path, sheet_name: str | int) -> pd.DataFrame:
    try:
        return clean_excel_columns(pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl"))
    except ValueError:
        return pd.DataFrame()


def value_is_empty(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan" or text == "/"


def to_float(value: object, default: float = np.nan) -> float:
    if value_is_empty(value):
        return default
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    if not match:
        return default
    return float(match.group())


def display_value(value: object) -> str:
    if value_is_empty(value):
        return "/"
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if np.isfinite(value) and value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def unique_values(values: object, fallback: list[object] | None = None) -> list[object]:
    if isinstance(values, pd.DataFrame):
        raise TypeError("unique_values expects a Series, list, tuple, or set")
    if isinstance(values, pd.Series):
        raw_values = values.dropna().tolist()
    elif isinstance(values, (list, tuple, set)):
        raw_values = list(values)
    else:
        raw_values = [values]
    if fallback:
        raw_values.extend(fallback)

    result: list[object] = []
    seen: set[str] = set()
    for value in raw_values:
        if value_is_empty(value):
            continue
        key = display_value(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)

    def sort_key(value: object) -> tuple[int, float | str]:
        numeric = to_float(value)
        if np.isfinite(numeric):
            return (0, numeric)
        return (1, str(value))

    return sorted(result, key=sort_key)


def select_single(label: str, options: list[object], default: object, key: str) -> object:
    choices = unique_values(options, [default])
    default_key = display_value(default)
    keys = [display_value(value) for value in choices]
    index = keys.index(default_key) if default_key in keys else 0
    return st.selectbox(label, choices, index=index, key=key, format_func=display_value)


def select_middle_material(default: object, key: str) -> object:
    """Render the fixed middle-package material list, including '/' as an option."""
    choices = list(MIDDLE_MATERIAL_OPTIONS)
    default_value = display_value(default)
    if default_value == "中盒-E瓦楞":
        default_value = "E瓦楞"
    elif default_value == "塑袋":
        default_value = "塑袋（贴标）"
    index = choices.index(default_value) if default_value in choices else 0
    return st.selectbox(
        "中包装 方式（材质）",
        choices,
        index=index,
        key=key,
        format_func=display_value,
    )


def select_config(
    label: str,
    options: pd.DataFrame,
    label_col: str,
    select_key: str,
    placeholder: str,
) -> int:
    indices = options.index.tolist()
    option_rows = [
        {"value": str(idx), "label": str(options.loc[idx, label_col])}
        for idx in indices
    ]
    value_to_index = {row["value"]: idx for row, idx in zip(option_rows, indices)}
    default_value = option_rows[0]["value"]
    current_value = str(st.session_state.get(select_key, default_value))
    if current_value not in value_to_index:
        current_value = default_value

    selected_value = SEARCHABLE_SELECT_COMPONENT(
        label=label,
        options=option_rows,
        value=current_value,
        placeholder=placeholder,
        key=select_key,
        default=current_value,
        height=76,
    )
    selected_value = str(selected_value if selected_value is not None else current_value)
    return value_to_index.get(selected_value, indices[0])


def normalize_sterilization(value: object) -> str:
    if value_is_empty(value):
        return "无灭菌"
    text = str(value).strip()
    if text in {"不灭菌", "无", "/"}:
        return "无灭菌"
    if text == "伽玛灭菌":
        return "伽马灭菌"
    return text if text in STERILIZATION_OPTIONS else text


@st.dialog("报价参考提示")
def show_low_sample_warning(reasons: list[str]) -> None:
    st.warning("案例数据过少，报价结果仅供参考。")
    st.write(f"本次选择涉及：{'、'.join(reasons)}")
    st.caption("建议结合人工成本核算结果进行复核，并持续补充同类案例数据。")
    if st.button("我知道了", key="low_sample_warning_confirm", type="primary"):
        st.rerun()


def parse_size(value: object) -> tuple[float, float]:
    text = str(value).lower().replace("×", "*").replace("x", "*")
    match = re.search(r"(\d+(?:\.\d+)?)\s*\*\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return (np.nan, np.nan)
    return (float(match.group(1)), float(match.group(2)))


def parse_composition(value: object) -> tuple[float, float]:
    text = str(value)
    viscose = re.search(r"(\d+(?:\.\d+)?)\s*%\s*粘胶", text)
    polyester = re.search(r"(\d+(?:\.\d+)?)\s*%\s*涤纶", text)
    if viscose and polyester:
        return (float(viscose.group(1)), float(polyester.group(1)))
    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) >= 2:
        return (float(numbers[0]), float(numbers[1]))
    return (np.nan, np.nan)


def percent_to_model_ratio(value: object) -> float:
    number = to_float(value)
    if not np.isfinite(number):
        return np.nan
    return number / 100 if number > 1 else number


def parse_color_count(value: object) -> float:
    if value_is_empty(value):
        return np.nan
    return to_float(value, default=np.nan)


def package_unit_count(value: object) -> float:
    if value_is_empty(value):
        return 1.0
    count = to_float(value, default=1.0)
    return count if np.isfinite(count) and count > 0 else 1.0


def package_box_qty(values: dict[str, object]) -> float:
    return (
        package_unit_count(values.get("内包装 装量"))
        * package_unit_count(values.get("中包装 装量"))
        * package_unit_count(values.get("外包装 装量"))
    )


def normalize_package_columns(values: dict[str, object]) -> dict[str, object]:
    normalized = dict(values)
    for old_name, new_name in [
        ("内包装 方式 （材质）", "内包装 方式（材质）"),
        ("中包装 方式 （材质）", "中包装 方式（材质）"),
        ("外包装 方式 （材质）", "外包装 方式（材质）"),
    ]:
        if old_name in normalized and new_name not in normalized:
            normalized[new_name] = normalized[old_name]
    return normalized


def build_history_from_latest_quote(path: Path, target: str) -> pd.DataFrame:
    final_df = read_quote_sheet(path, "最终数据")
    if final_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, source in final_df.iterrows():
        length, width = parse_size(source.get("长*宽（cm）"))
        viscose, polyester = parse_composition(source.get("配比%"))
        row = normalize_package_columns(
            {
                "产品标准配置": source.get("产品代码"),
                "包装标准配置": " / ".join(
                    display_value(source.get(col))
                    for col in [
                        "内包装 装量",
                        "内包装 方式 （材质）",
                        "中包装 装量",
                        "中包装 方式 （材质）",
                        "外包装 装量",
                        "外包装 方式 （材质）",
                    ]
                    if not value_is_empty(source.get(col))
                ),
                "长*宽（cm）": source.get("长*宽（cm）"),
                "长（cm）": length,
                "宽（cm）": width,
                "层数": source.get("层数"),
                "克重g/㎡": source.get("克重g/㎡"),
                "粘胶配比%": viscose,
                "涤纶配比%": polyester,
                "灭菌方式": normalize_sterilization(source.get("灭菌方式")),
                "内包装 装量": source.get("内包装 装量"),
                "内包装 方式 （材质）": source.get("内包装 方式 （材质）"),
                "内包装 印刷色数": source.get("内包装 印刷色数"),
                "中包装 装量": source.get("中包装 装量"),
                "中包装 方式 （材质）": source.get("中包装 方式 （材质）"),
                "中包装 印刷色数": source.get("中包装 印刷色数"),
                "外包装 装量": source.get("外包装 装量"),
                "外包装 方式 （材质）": source.get("外包装 方式 （材质）"),
                "外包装 印刷色数": source.get("外包装 印刷色数"),
                target: to_float(source.get("单价")),
            }
        )
        rows.append(row)

    history = pd.DataFrame(rows)
    history[target] = pd.to_numeric(history[target], errors="coerce")
    return history


def resolve_app_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else APP_DIR / path


def load_bottom_parameter_options(path: Path) -> dict[str, list[object]]:
    sheet = read_quote_sheet(path, 0)
    if sheet.empty or "底层参数" not in sheet.columns or "数据示例" not in sheet.columns:
        return {}

    marker = sheet.index[
        sheet["底层参数"].astype(str).str.strip().eq("底层参数")
        & sheet["数据示例"].astype(str).str.strip().eq("单选")
    ]
    option_rows = sheet.loc[marker[0] + 1 :] if len(marker) else sheet
    raw_options: dict[str, list[object]] = {}
    current_parameter = ""
    for _, row in option_rows.iterrows():
        parameter_value = row.get("底层参数")
        if not pd.isna(parameter_value) and str(parameter_value).strip():
            current_parameter = re.sub(r"\s+", " ", str(parameter_value)).strip()
        option_value = row.get("数据示例")
        if not current_parameter or pd.isna(option_value) or not str(option_value).strip():
            continue
        raw_options.setdefault(current_parameter, []).append(option_value)

    size_values = [parse_size(value) for value in raw_options.get("长*宽（cm）", [])]
    composition_values = [parse_composition(value) for value in raw_options.get("配比%", [])]

    def numeric_options(parameter: str) -> list[object]:
        return unique_values(
            [to_float(value) for value in raw_options.get(parameter, [])]
        )

    def text_options(parameter: str) -> list[object]:
        return unique_values(raw_options.get(parameter, []))

    def color_options(parameter: str) -> list[object]:
        return unique_values(
            [f"{display_value(to_float(value))}色印刷" for value in raw_options.get(parameter, [])]
        )

    sterilization_options = [
        "无灭菌" if str(value).strip() == "/" else str(value).strip()
        for value in raw_options.get("灭菌方式", [])
    ]
    return {
        "长（cm）": unique_values([length for length, _ in size_values]),
        "宽（cm）": unique_values([width for _, width in size_values]),
        "层数": numeric_options("层数"),
        "克重g/㎡": numeric_options("克重g/㎡"),
        "粘胶配比%": unique_values([viscose for viscose, _ in composition_values]),
        "涤纶配比%": unique_values([polyester for _, polyester in composition_values]),
        "灭菌方式": unique_values(sterilization_options),
        "内包装 装量": text_options("内包装 装量"),
        "内包装 方式 （材质）": text_options("内包装 方式 （材质）"),
        "内包装 印刷色数": color_options("内包装 印刷色数"),
        "中包装 装量": text_options("中包装 装量"),
        "中包装 方式 （材质）": list(MIDDLE_MATERIAL_OPTIONS),
        "中包装 印刷色数": color_options("中包装 印刷色数"),
        "外包装 装量": text_options("外包装 装量"),
        "外包装 方式 （材质）": text_options("外包装 方式 （材质）"),
        "外包装 印刷色数": color_options("外包装 印刷色数"),
    }


@st.cache_data
def load_quote_options() -> dict[str, object]:
    path = quote_parameter_path()
    if path is None:
        return {"products": pd.DataFrame(), "packages": pd.DataFrame(), "final": pd.DataFrame()}

    products = read_quote_sheet(path, "产品参数组合选项").dropna(how="all")
    packages = read_quote_sheet(path, "包装组合参数").dropna(how="all")
    final = read_quote_sheet(path, "最终数据").dropna(how="all")

    if "产品组合选项" in products.columns:
        products = products.dropna(subset=["产品组合选项"]).reset_index(drop=True)
    if "包装组合选项" in packages.columns:
        packages = packages.dropna(subset=["包装组合选项"]).reset_index(drop=True)
    middle_quantity_empty = packages["中包装 装量"].map(value_is_empty)
    middle_material_empty = packages["中包装 方式 （材质）"].map(value_is_empty)
    no_middle = middle_quantity_empty & middle_material_empty
    packages.loc[no_middle, "中包装 装量"] = "0袋"
    packages.loc[no_middle, "中包装 方式 （材质）"] = "无中包装"
    packages.loc[no_middle, "中包装 印刷色数"] = "0色印刷"

    return {
        "products": products,
        "packages": packages,
        "final": final,
        "bottom": load_bottom_parameter_options(path),
    }


@st.cache_data
def load_history(
    input_file: str,
    target: str,
    sheet_name: str = "",
    use_quote_fallback: bool = True,
) -> pd.DataFrame:
    quote_path = quote_parameter_path()
    if use_quote_fallback and quote_path is not None:
        latest_history = build_history_from_latest_quote(quote_path, target)
        if not latest_history.empty and target in latest_history.columns:
            return latest_history

    configured_path = resolve_app_path(input_file)
    candidates: list[tuple[Path, str]] = [
        (configured_path, sheet_name),
        (APP_DIR / RAW_HISTORY_FILE, ""),
    ]
    known_paths = {path for path, _ in candidates}
    for discovered_path in sorted(APP_DIR.rglob("*.xlsx")):
        if discovered_path.name.startswith("~$"):
            continue
        if discovered_path not in known_paths:
            candidates.append((discovered_path, sheet_name))
            known_paths.add(discovered_path)
    attempted: list[str] = []
    seen: set[Path] = set()

    for path, candidate_sheet in candidates:
        normalized = path.resolve() if path.exists() else path
        if normalized in seen:
            continue
        seen.add(normalized)
        if not path.is_file():
            attempted.append(f"{path}: 文件不存在")
            continue

        sheet_candidates: list[str | int] = [candidate_sheet] if candidate_sheet else [0]
        if candidate_sheet:
            sheet_candidates.append(0)
        for candidate in sheet_candidates:
            try:
                df = pd.read_excel(path, sheet_name=candidate, engine="openpyxl")
            except (OSError, ValueError) as exc:
                attempted.append(
                    f"{path}（工作表 {candidate}）: {type(exc).__name__}: {exc}"
                )
                continue

            df.columns = [str(c).replace("\n", "").strip() for c in df.columns]
            if target not in df.columns:
                attempted.append(f"{path}（工作表 {candidate}）: 找不到目标列“{target}”")
                continue

            df[target] = pd.to_numeric(df[target], errors="coerce")
            df = df.dropna(subset=[target])
            if df.empty:
                attempted.append(f"{path}（工作表 {candidate}）: 目标列“{target}”没有有效数值")
                continue
            return df

    details = "\n".join(attempted)
    raise FileNotFoundError(
        "无法加载历史成本数据。请上传模型指定的预处理文件，或在仓库根目录上传"
        f"“{RAW_HISTORY_FILE}”。\n{details}"
    )


def get_feature_options(history: pd.DataFrame, col: str, fallback: list[str]) -> list[str]:
    if col not in history.columns:
        return fallback
    values = [str(v) for v in history[col].dropna().unique().tolist()]
    return sorted(values) or fallback


def default_value(history: pd.DataFrame, col: str, fallback: float) -> float:
    if col not in history.columns:
        return fallback
    series = pd.to_numeric(history[col], errors="coerce").dropna()
    if series.empty:
        return fallback
    return float(series.median())


def numeric_input(
    label: str,
    history: pd.DataFrame,
    col: str,
    fallback: float,
    min_value: float | None = None,
    step: float = 0.01,
    fmt: str = "%.2f",
) -> float:
    value = default_value(history, col, fallback)
    kwargs = {"label": label, "value": value, "step": step, "format": fmt}
    if min_value is not None:
        kwargs["min_value"] = min_value
    return float(st.number_input(**kwargs))


def row_value(row: pd.Series, col: str, fallback: object = None) -> object:
    if col not in row or value_is_empty(row.get(col)):
        return fallback
    return row.get(col)


def column_options(df: pd.DataFrame, col: str, fallback: list[object] | None = None) -> list[object]:
    if col not in df.columns:
        return fallback or []
    return unique_values(df[col], fallback)


def split_quantity(value: object, default_number: int, default_unit: str) -> tuple[int, str]:
    if value_is_empty(value):
        return default_number, default_unit
    text = str(value).strip()
    number = to_float(text, default=float(default_number))
    unit = re.sub(r"^\s*\d+(?:\.\d+)?\s*", "", text).strip() or default_unit
    return int(round(number)), unit


def quantity_number_input(label: str, default_value: object, default_unit: str, key: str) -> str:
    number, unit = split_quantity(default_value, 10, default_unit)
    current = st.number_input(
        label,
        min_value=0,
        value=max(number, 0),
        step=1,
        format="%d",
        key=key,
    )
    return f"{int(current)}{unit}"


def parse_quantity_parts(value: object, default_unit: str) -> tuple[float, str]:
    if value_is_empty(value):
        return 0.0, default_unit
    text = str(value).strip()
    quantity = to_float(text, default=0.0)
    unit = re.sub(r"^\s*-?\d+(?:\.\d+)?\s*", "", text).strip() or default_unit
    return quantity, unit


def build_preprocessed_huber_row(
    input_data: dict[str, object],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    row = normalize_package_columns(dict(input_data))
    inner_quantity, inner_unit = parse_quantity_parts(row.get("内包装 装量"), "片")
    middle_quantity, middle_unit = parse_quantity_parts(row.get("中包装 装量"), "袋")
    outer_quantity, outer_unit = parse_quantity_parts(row.get("外包装 装量"), "盒")
    inner_material = display_value(row.get("内包装 方式（材质）"))
    middle_material = display_value(row.get("中包装 方式（材质）"))
    no_middle = middle_quantity <= 0 or middle_material in {"/", "无", "无中包装"}
    if no_middle:
        middle_quantity = 0.0
        middle_unit = "无中包装"
        middle_material = "无中包装"
    elif middle_material == "中盒-E瓦楞":
        middle_material = "E瓦楞"

    total_inner_packages = (middle_quantity if not no_middle else 1.0) * outer_quantity
    total_pieces = inner_quantity * total_inner_packages
    feature_values: dict[str, float] = {
        "克重g/㎡": to_float(row.get("克重g/㎡")),
        "长": to_float(row.get("长（cm）")),
        "宽（cm）": to_float(row.get("宽（cm）")),
        "层数": to_float(row.get("层数")),
        "粘胶配比": percent_to_model_ratio(row.get("粘胶配比%")),
        "涤纶配比": percent_to_model_ratio(row.get("涤纶配比%")),
        "是否EO灭菌": float(normalize_sterilization(row.get("灭菌方式")) == "EO灭菌"),
        "内包装数量": inner_quantity,
        "内包装印刷色数": parse_color_count(row.get("内包装 印刷色数")),
        "是否有中包装": float(not no_middle),
        "中包装数量": middle_quantity,
        "中包装印刷色数": (
            0.0 if no_middle else parse_color_count(row.get("中包装 印刷色数"))
        ),
        "中包装印刷色数_缺失标记": 0.0,
        "外包装数量": outer_quantity,
        "外包装印刷色数": parse_color_count(row.get("外包装 印刷色数")),
        "每外包装总装量": total_inner_packages,
        "每外包装总片数": total_pieces,
        "内包装成本分摊系数": 1.0 / inner_quantity if inner_quantity > 0 else 0.0,
        "中包装成本分摊系数": (
            1.0 / (inner_quantity * middle_quantity)
            if not no_middle and inner_quantity > 0 and middle_quantity > 0
            else 0.0
        ),
        "外包装成本分摊系数": 1.0 / total_pieces if total_pieces > 0 else 0.0,
    }
    categories = {
        "内包装材质": inner_material,
        "中包装材质": middle_material,
        "内包装单位": inner_unit,
        "中包装单位": middle_unit,
        "外包装单位": outer_unit,
    }
    for feature in feature_columns:
        for prefix, category in categories.items():
            marker = f"{prefix}_"
            if feature.startswith(marker):
                feature_values[feature] = float(feature[len(marker) :] == category)
                break

    model_row = pd.DataFrame(
        [[feature_values.get(feature, 0.0) for feature in feature_columns]],
        columns=feature_columns,
    )
    derived = {
        "每外包装总装量": total_inner_packages,
        "每外包装总片数": total_pieces,
        "内包装成本分摊系数": feature_values["内包装成本分摊系数"],
        "中包装成本分摊系数": feature_values["中包装成本分摊系数"],
        "外包装成本分摊系数": feature_values["外包装成本分摊系数"],
    }
    return model_row, derived


def canonical_middle_material(value: object) -> str:
    material = display_value(value)
    if material in {"/", "无", "无中包装"}:
        return "无中包装"
    if material == "中盒-E瓦楞":
        return "E瓦楞"
    if material == "塑袋":
        return "塑袋（贴标）"
    return material


def build_new_category_price_row(
    input_data: dict[str, object],
    bundle: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    old_columns = list(bundle["old_feature_columns"])
    feature_columns = list(bundle["original_feature_columns"])
    base_row, derived = build_strict_price_row(input_data, old_columns)

    sterilization = normalize_sterilization(input_data.get("灭菌方式"))
    middle_material = canonical_middle_material(input_data.get("中包装 方式（材质）"))
    has_middle = bool(base_row.iloc[0]["是否有中包装"])
    paper_weight = float(base_row.iloc[0]["中包装纸张克重"])
    paper_applicable = float(
        has_middle
        and ("白卡" in middle_material or "灰底白板" in middle_material)
    )
    extra_values = {
        "灭菌方式_EO预处理": float(sterilization == "EO预处理"),
        "灭菌方式_蒸汽灭菌": float(sterilization == "蒸汽灭菌"),
        "灭菌方式_伽马灭菌": float(sterilization == "伽马灭菌"),
        "中包装材质_E瓦楞": float(has_middle and middle_material == "E瓦楞"),
        "中包装材质_塑袋（贴标）": float(
            has_middle and middle_material == "塑袋（贴标）"
        ),
        "中包装材质_中箱": float(has_middle and middle_material == "中箱"),
        "中包装纸张克重_适用": paper_applicable,
        "中包装纸张克重_缺失": float(paper_applicable == 1.0 and paper_weight <= 0),
    }
    values = base_row.iloc[0].to_dict()
    values.update(extra_values)
    missing = [column for column in feature_columns if column not in values]
    if missing:
        raise ValueError(f"网页未生成新类别模型所需特征：{', '.join(missing)}")
    model_row = pd.DataFrame(
        [[values[column] for column in feature_columns]],
        columns=feature_columns,
        dtype=float,
    )
    if not np.isfinite(model_row.to_numpy(dtype=float)).all():
        raise ValueError("新类别模型输入包含无效数值，请检查包装装量和产品规格。")
    derived.update(
        {
            "灭菌方式_清洗": sterilization,
            "中包装材质_清洗": middle_material,
            "中包装纸张克重_适用": paper_applicable,
            "中包装纸张克重_缺失": extra_values["中包装纸张克重_缺失"],
        }
    )
    return model_row, derived


def build_legacy_input_row(input_data: dict[str, object]) -> tuple[pd.DataFrame, dict[str, object]]:
    row = normalize_package_columns(dict(input_data))
    row["粘胶配比%"] = percent_to_model_ratio(row.get("粘胶配比%"))
    row["涤纶配比%"] = percent_to_model_ratio(row.get("涤纶配比%"))

    row["内包装材质"] = row.get("内包装 方式（材质）")
    row["材质色数（内）"] = parse_color_count(row.get("内包装 印刷色数"))
    row["外箱材质"] = row.get("外包装 方式（材质）")
    row["材质色数（外）"] = parse_color_count(row.get("外包装 印刷色数"))
    row["每箱数量（PCS)"] = package_box_qty(row)

    length = to_float(row.get("长（cm）"))
    width = to_float(row.get("宽（cm）"))
    layers = to_float(row.get("层数"))
    gram = to_float(row.get("克重g/㎡"))
    row["产品面积cm2"] = length * width if np.isfinite(length) and np.isfinite(width) else np.nan
    row["用料指数"] = (
        length * width * layers * gram
        if all(np.isfinite(v) for v in [length, width, layers, gram])
        else np.nan
    )

    moq = to_float(row.get("MOQ"))
    row["MOQ_log"] = float(np.log1p(moq)) if np.isfinite(moq) else np.nan

    carton_l = to_float(row.get("纸箱（长）"))
    carton_w = to_float(row.get("纸箱（宽）"))
    carton_h = to_float(row.get("纸箱（高）"))
    row["纸箱面积"] = (
        carton_l * carton_w if np.isfinite(carton_l) and np.isfinite(carton_w) else np.nan
    )
    row["纸箱体积"] = (
        row["纸箱面积"] * carton_h
        if np.isfinite(row["纸箱面积"]) and np.isfinite(carton_h)
        else np.nan
    )
    row["是否有纸箱"] = (
        int(carton_l + carton_w + carton_h > 0)
        if all(np.isfinite(v) for v in [carton_l, carton_w, carton_h])
        else np.nan
    )

    capacity = to_float(row.get("折叠产能（PCS)"))
    people = to_float(row.get("折叠岗位人数"))
    row["折叠人均产能"] = (
        capacity / people
        if np.isfinite(capacity) and np.isfinite(people) and people not in (0.0, -0.0)
        else np.nan
    )
    derived = {
        "产品面积cm2": row["产品面积cm2"],
        "用料指数": row["用料指数"],
        "MOQ_log": row["MOQ_log"],
        "纸箱面积": row["纸箱面积"],
        "纸箱体积": row["纸箱体积"],
        "是否有纸箱": row["是否有纸箱"],
        "折叠人均产能": row["折叠人均产能"],
    }
    return pd.DataFrame([row]), derived


def build_input_row(
    input_data: dict[str, object],
    bundle: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    if bundle.get("bundle_format") == NEW_CATEGORY_BUNDLE_FORMAT:
        return build_new_category_price_row(input_data, bundle)
    if bundle.get("bundle_format") == STRICT_BUNDLE_FORMAT:
        return build_strict_price_row(
            input_data,
            list(bundle["original_feature_columns"]),
        )
    if bundle.get("bundle_format") == "preprocessed_huber_v1":
        return build_preprocessed_huber_row(
            input_data,
            list(bundle["feature_columns"]),
        )
    return build_legacy_input_row(input_data)


def augment_new_category_features(frame: pd.DataFrame) -> pd.DataFrame:
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
    for feature, label in [
        ("灭菌方式_EO预处理", "EO预处理"),
        ("灭菌方式_蒸汽灭菌", "蒸汽灭菌"),
        ("灭菌方式_伽马灭菌", "伽马灭菌"),
    ]:
        result[f"{label}材料交互"] = result[feature] * result["理论材料重量g"]
        result[f"{label}面积交互"] = result[feature] * result["产品面积cm2"]
    for feature, label in [
        ("中包装材质_E瓦楞", "E瓦楞"),
        ("中包装材质_塑袋（贴标）", "塑袋（贴标）"),
        ("中包装材质_中箱", "中箱"),
    ]:
        result[f"中包装{label}摊销交互"] = result[feature] * result["中包装成本分摊系数"]
    result["中包装白卡摊销交互"] = result["中包装材质_白卡"] * result["中包装成本分摊系数"]
    return result


def _predict_new_category_model_raw(
    bundle: dict[str, object],
    input_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    validate_new_category_bundle(bundle)
    original_columns = list(bundle["original_feature_columns"])
    original = input_df[original_columns].astype(float)
    augmented = augment_new_category_features(original)
    expected_augmented = list(bundle["augmented_feature_columns"])
    if list(augmented.columns) != expected_augmented:
        raise ValueError("新类别增强特征顺序与训练模型不一致。")
    frames = {"new_base": original, "new_aug": augmented}

    fitted_model = bundle["fitted_model"]
    component_predictions: dict[str, np.ndarray] = {}
    for name in fitted_model["components"]:
        artifact = fitted_model["fitted"][name]
        feature_set = str(artifact["spec"]["feature_set"])
        if feature_set not in frames:
            raise ValueError(f"新类别模型引用了未知特征集：{feature_set}")
        component_predictions[name] = np.asarray(
            artifact["estimator"].predict(frames[feature_set]),
            dtype=float,
        )

    weights = np.asarray(fitted_model["weights"], dtype=float)
    component_names = list(fitted_model["components"])
    if len(component_names) != len(weights) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("新类别融合模型的组件权重无效。")
    matrix = np.column_stack([component_predictions[name] for name in component_names])
    prediction = matrix @ weights * float(fitted_model["calibration_factor"])
    return np.asarray(prediction, dtype=float), component_predictions


def _apply_middle_color_increment(
    bundle: dict[str, object],
    input_df: pd.DataFrame,
    raw_prediction: np.ndarray,
) -> np.ndarray:
    """Use the 0-color model output as the base and add a strict color increment."""
    original_columns = list(bundle["original_feature_columns"])
    if (
        len(input_df) == 0
        or MIDDLE_PRINT_COLOR_FEATURE not in original_columns
        or "是否有中包装" not in original_columns
    ):
        return np.asarray(raw_prediction, dtype=float)

    original = input_df[original_columns].astype(float).reset_index(drop=True)
    constrained = np.asarray(raw_prediction, dtype=float).copy()
    for row_index in range(len(original)):
        if not np.isclose(float(original.loc[row_index, "是否有中包装"]), 1.0):
            continue

        current_color = float(original.loc[row_index, MIDDLE_PRINT_COLOR_FEATURE])
        level_index = int(np.argmin(np.abs(MIDDLE_PRINT_COLOR_LEVELS - current_color)))
        if not np.isclose(current_color, MIDDLE_PRINT_COLOR_LEVELS[level_index]):
            continue

        allocation_factor = float(original.loc[row_index, "中包装成本分摊系数"])
        if not np.isfinite(allocation_factor) or allocation_factor <= 0:
            continue

        base_scenario = original.iloc[[row_index]].copy().reset_index(drop=True)
        base_scenario.loc[0, MIDDLE_PRINT_COLOR_FEATURE] = 0.0
        base_prediction, _ = _predict_new_category_model_raw(bundle, base_scenario)
        constrained[row_index] = (
            float(base_prediction[0])
            + allocation_factor * MIDDLE_PRINT_INCREMENT_PER_ALLOCATION_UNIT[level_index]
        )

    return constrained


def _apply_outer_color_increment(
    bundle: dict[str, object],
    input_df: pd.DataFrame,
    raw_prediction: np.ndarray,
) -> np.ndarray:
    """Use the 0-color model output as the base and add a strict outer-color increment."""
    original_columns = list(bundle["original_feature_columns"])
    if (
        len(input_df) == 0
        or OUTER_PRINT_COLOR_FEATURE not in original_columns
        or "外包装成本分摊系数" not in original_columns
    ):
        return np.asarray(raw_prediction, dtype=float)

    original = input_df[original_columns].astype(float).reset_index(drop=True)
    constrained = np.asarray(raw_prediction, dtype=float).copy()
    for row_index in range(len(original)):
        current_color = float(original.loc[row_index, OUTER_PRINT_COLOR_FEATURE])
        level_index = int(np.argmin(np.abs(OUTER_PRINT_COLOR_LEVELS - current_color)))
        if not np.isclose(current_color, OUTER_PRINT_COLOR_LEVELS[level_index]):
            continue

        allocation_factor = float(original.loc[row_index, "外包装成本分摊系数"])
        if not np.isfinite(allocation_factor) or allocation_factor <= 0:
            continue

        base_scenario = original.iloc[[row_index]].copy().reset_index(drop=True)
        base_scenario.loc[0, OUTER_PRINT_COLOR_FEATURE] = 0.0
        base_prediction, _ = _predict_new_category_model_raw(bundle, base_scenario)
        base_prediction = _apply_middle_color_increment(
            bundle,
            base_scenario,
            base_prediction,
        )
        constrained[row_index] = (
            float(base_prediction[0])
            + allocation_factor * OUTER_PRINT_INCREMENT_PER_ALLOCATION_UNIT[level_index]
        )

    return constrained


def predict_new_category_model(
    bundle: dict[str, object],
    input_df: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    raw_prediction, component_predictions = _predict_new_category_model_raw(
        bundle,
        input_df,
    )
    prediction = _apply_middle_color_increment(bundle, input_df, raw_prediction)
    prediction = _apply_outer_color_increment(bundle, input_df, prediction)
    return prediction, component_predictions


def new_category_local_sensitivity(
    bundle: dict[str, object],
    input_df: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    if len(input_df) != 1:
        raise ValueError("局部敏感度分析只支持单条预测。")
    original_columns = list(bundle["original_feature_columns"])
    base_frame = input_df[original_columns].astype(float)
    base_prediction = float(predict_new_category_model(bundle, base_frame)[0][0])

    scale_by_feature: dict[str, float] = {}
    fitted_model = bundle["fitted_model"]
    for artifact in fitted_model["fitted"].values():
        if artifact["spec"].get("feature_set") != "new_aug":
            continue
        fitted_pipeline = getattr(artifact["estimator"], "regressor_", None)
        if fitted_pipeline is None or "scale" not in fitted_pipeline.named_steps:
            continue
        scales = np.asarray(fitted_pipeline.named_steps["scale"].scale_, dtype=float)
        scale_by_feature.update(dict(zip(bundle["augmented_feature_columns"], scales)))
        break

    records: list[dict[str, object]] = []
    for feature in original_columns:
        value = float(base_frame.iloc[0][feature])
        if feature in {MIDDLE_PRINT_COLOR_FEATURE, OUTER_PRINT_COLOR_FEATURE}:
            current_level = int(np.clip(round(value), 0, 4))
            candidates = [
                float(level)
                for level in (current_level - 1, current_level + 1)
                if 0 <= level <= 4
            ]
        elif feature in NEW_CATEGORY_BINARY_FEATURES:
            candidates = [1.0 - value]
        else:
            reference_scale = abs(scale_by_feature.get(feature, value))
            step = max(abs(value) * 0.05, reference_scale * 0.05, 1e-6)
            lower = max(0.0, value - step)
            upper = value + step
            if feature in NEW_CATEGORY_RATIO_FEATURES:
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
            prediction = float(predict_new_category_model(bundle, changed)[0][0])
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
    sensitivity = pd.DataFrame(records)
    feature_to_input = {
        feature: input_name
        for input_name, features in NEW_CATEGORY_INPUT_FEATURE_GROUPS.items()
        for feature in features
    }
    sensitivity = sensitivity[sensitivity["特征"].isin(feature_to_input)].copy()
    sensitivity["特征"] = sensitivity["特征"].map(feature_to_input)
    # A categorical input can map to several one-hot columns; retain its largest change.
    sensitivity = (
        sensitivity.sort_values("绝对敏感度", ascending=False)
        .drop_duplicates(subset=["特征"], keep="first")
        .head(top_n)
        .reset_index(drop=True)
    )
    total_sensitivity = float(sensitivity["绝对敏感度"].sum())
    sensitivity["影响权重"] = (
        sensitivity["绝对敏感度"] / total_sensitivity * 100.0
        if total_sensitivity > 0
        else 0.0
    )
    return sensitivity


def predict_huber(bundle: dict[str, object], input_df: pd.DataFrame) -> tuple[float, float]:
    if bundle.get("bundle_format") == NEW_CATEGORY_BUNDLE_FORMAT:
        prediction, _ = predict_new_category_model(bundle, input_df)
        predicted_cost = float(prediction[0])
        if predicted_cost <= 0:
            raise ValueError("新类别融合模型返回了非正价格，请检查输入参数。")
        return predicted_cost, float(np.log(predicted_cost))
    if bundle.get("bundle_format") == STRICT_BUNDLE_FORMAT:
        prediction, _ = predict_strict_price_model(bundle, input_df)
        predicted_cost = float(prediction[0])
        if predicted_cost <= 0:
            raise ValueError("融合模型返回了非正价格，请检查输入参数。")
        return predicted_cost, float(np.log(predicted_cost))
    if bundle.get("bundle_format") == "preprocessed_huber_v1":
        model = bundle["model_pipeline"]
        expected_cols = list(bundle["feature_columns"])
        predicted_cost = float(model.predict(input_df[expected_cols])[0])
        return predicted_cost, float(np.log(predicted_cost))

    model = bundle["formula_pipeline"]
    expected_numeric = bundle["numeric_cols"]
    expected_categorical = bundle["categorical_cols"]
    expected_cols = list(expected_numeric) + list(expected_categorical)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df = input_df[expected_cols]
    log_prediction = float(model.predict(input_df)[0])
    return float(np.exp(log_prediction)), log_prediction


def coefficient_analysis(bundle: dict[str, object], input_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if bundle.get("bundle_format") == NEW_CATEGORY_BUNDLE_FORMAT:
        return new_category_local_sensitivity(bundle, input_df, top_n=top_n)
    if bundle.get("bundle_format") == STRICT_BUNDLE_FORMAT:
        return local_sensitivity(bundle, input_df, top_n=top_n)
    if bundle.get("bundle_format") == "preprocessed_huber_v1":
        model = bundle["model_pipeline"]
        feature_names = list(bundle["feature_columns"])
        fitted_pipeline = model.regressor_
        scaler = fitted_pipeline.named_steps["scale"]
        regressor = fitted_pipeline.named_steps["model"]
        transformed = scaler.transform(input_df[feature_names])
        coefs = np.asarray(regressor.coef_, dtype=float)
        contributions = np.asarray(transformed[0], dtype=float) * coefs
        result = pd.DataFrame(
            {
                "特征": feature_names,
                "模型输入值": np.asarray(transformed[0], dtype=float),
                "系数": coefs,
                "成本贡献": contributions,
                "影响方向": np.where(contributions >= 0, "推高成本", "降低成本"),
            }
        )
        result["绝对影响"] = result["成本贡献"].abs()
        return result.sort_values("绝对影响", ascending=False).head(top_n)

    model = bundle["formula_pipeline"]
    preprocessor = model.named_steps["preprocess"]
    regressor = model.named_steps["model"]
    numeric_cols = list(bundle["numeric_cols"])
    categorical_cols = list(bundle["categorical_cols"])

    transformed = preprocessor.transform(input_df)
    coefs = np.asarray(regressor.coef_, dtype=float)
    feature_names = list(numeric_cols)
    if categorical_cols:
        onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
        feature_names.extend(list(onehot.get_feature_names_out(categorical_cols)))

    contributions = np.asarray(transformed[0], dtype=float) * coefs
    result = pd.DataFrame(
        {
            "特征": feature_names,
            "模型输入值": np.asarray(transformed[0], dtype=float),
            "系数": coefs,
            "成本贡献": contributions,
            "影响方向": np.where(contributions >= 0, "推高成本", "降低成本"),
        }
    )
    result["绝对影响"] = result["成本贡献"].abs()
    return result.sort_values("绝对影响", ascending=False).head(top_n)


def find_history_match(
    history: pd.DataFrame,
    input_data: dict[str, object] | pd.DataFrame,
    target: str,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if feature_columns is not None and isinstance(input_data, pd.DataFrame):
        if input_data.empty or any(col not in history.columns for col in feature_columns):
            return history.iloc[0:0]
        input_row = input_data.iloc[0]
        mask = pd.Series(True, index=history.index)
        for col in feature_columns:
            history_number = pd.to_numeric(history[col], errors="coerce")
            input_number = to_float(input_row[col])
            mask &= np.isclose(
                history_number,
                input_number,
                rtol=0,
                atol=1e-9,
                equal_nan=False,
            )
        return history.loc[mask].dropna(subset=[target])

    if not isinstance(input_data, dict):
        return history.iloc[0:0]
    label_cols = {"产品标准配置", "包装标准配置"}
    match_cols = [
        col
        for col in INPUT_PARAMETER_COLUMNS
        if col not in label_cols
        and col in history.columns
        and col in input_data
        and not value_is_empty(input_data[col])
    ]
    mask = pd.Series(True, index=history.index)
    for col in match_cols:
        input_number = to_float(input_data[col])
        history_number = pd.to_numeric(history[col], errors="coerce")
        if np.isfinite(input_number) and history_number.notna().any():
            mask &= np.isclose(history_number, input_number, rtol=0, atol=1e-9, equal_nan=False)
        else:
            left = history[col].fillna("").astype(str).str.strip()
            right = str(input_data[col]).strip()
            mask &= left == right
    return history.loc[mask].dropna(subset=[target])


def render_distribution(history: pd.DataFrame, target: str, predicted_cost: float) -> go.Figure:
    data = history[target].dropna()
    data = data[data > 0]
    fig = go.Figure()
    if len(data) > 1:
        try:
            from scipy.stats import gaussian_kde

            x_min = float(data.min())
            x_max = float(data.max())
            padding = max((x_max - x_min) * 0.1, x_max * 0.05, 1e-6)
            x_range = np.linspace(x_min - padding, x_max + padding, 280)
            kde = gaussian_kde(data)
            y_kde = kde(x_range)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y_kde,
                    fill="tozeroy",
                    line={"color": "#9ebbd1", "width": 4},
                    fillcolor="rgba(170, 192, 214, 0.48)",
                    name="历史成本密度",
                    hovertemplate="成本：%{x:.6f}<br>密度：%{y:.3f}<extra></extra>",
                )
            )
        except Exception:
            fig.add_trace(
                go.Histogram(
                    x=data,
                    nbinsx=40,
                    histnorm="probability density",
                    marker_color="#9fb8cc",
                    opacity=0.72,
                    name="历史成本分布",
                )
            )
    else:
        fig.add_trace(go.Scatter(x=data, y=np.ones(len(data)), mode="lines", line={"color": "#9ebbd1", "width": 4}))

    fig.add_vline(x=predicted_cost, line_width=3, line_dash="dash", line_color="#e5554f")
    peak_y = float(np.nanmax(fig.data[0].y)) if len(fig.data) and getattr(fig.data[0], "y", None) is not None else 1.0
    fig.add_annotation(
        x=predicted_cost,
        y=peak_y * 0.95 if peak_y > 0 else 0.95,
        text="当前核价位置",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#e5554f",
        font={"color": "#e5554f", "size": 14},
        bgcolor="white",
        bordercolor="#e5554f",
        borderpad=4,
    )
    fig.update_layout(
        height=520,
        title={"text": "内部成本分布密度图", "x": 0.42, "y": 0.94},
        xaxis_title="成本单价（元/PCS）",
        yaxis_title="出现频率（密度）",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"color": "#667085", "size": 14},
        margin={"l": 35, "r": 20, "t": 75, "b": 50},
        showlegend=False,
    )
    fig.update_xaxes(
        gridcolor="#dfe5ec",
        zeroline=False,
        tickformat=".3f",
        title_font={"size": 18, "color": "#667085"},
    )
    fig.update_yaxes(
        gridcolor="#dfe5ec",
        zeroline=False,
        title_font={"size": 18, "color": "#667085"},
    )
    return fig


def distribution_rating(predicted_cost: float, history: pd.Series) -> tuple[str, str, str]:
    if history.empty:
        return "中位区", "#d97706", "历史数据不足，当前结果仅供参考。"
    data = history.dropna()
    data = data[data > 0]
    if data.empty:
        return "中位区", "#d97706", "历史数据不足，当前结果仅供参考。"
    pct = float((data <= predicted_cost).mean() * 100)
    if pct <= 25:
        return "低价位", "#1f9d55", "极具市场竞争力。"
    if pct <= 75:
        return "中价位", "#d97706", "处于常规区间，适合正常报价。"
    if pct <= 90:
        return "偏高位", "#f59e0b", "建议核对工艺、包装或用料参数。"
    return "高价位", "#dc2626", "明显高于常规水平，建议重点复核成本构成。"


def history_percentile(predicted_cost: float, history: pd.Series) -> float | None:
    data = pd.to_numeric(history, errors="coerce").dropna()
    data = data[data > 0]
    if data.empty:
        return None
    return float((data <= predicted_cost).mean() * 100)


def prediction_judgment(predicted_cost: float, history: pd.Series) -> tuple[str, str]:
    pct = history_percentile(predicted_cost, history)
    if pct is None:
        return "judgment-warning", "历史数据不足，当前结果仅供参考。"
    if pct < 5:
        return "judgment-warning", f"当前预测低于历史5%分位（约{pct:.1f}%分位），建议复核是否存在漏填或异常低值。"
    if pct > 95:
        return "judgment-danger", f"当前预测高于历史95%分位（约{pct:.1f}%分位），建议重点复核成本构成。"
    return "judgment-success", f"当前预测位于历史常规区间（约{pct:.1f}%分位），可作为常规报价参考。"


def render_history_analysis_section(history: pd.DataFrame, target: str, predicted_cost: float) -> None:
    data = history[target].dropna()
    data = data[data > 0]
    if data.empty:
        st.info("历史数据不足，无法绘制分布分析图。")
        return

    label, color, text = distribution_rating(predicted_cost, data)
    pct = float((data <= predicted_cost).mean() * 100)

    st.markdown(
        """
        <div class="history-band">
            <span class="history-emoji">📊</span>
            历史基准对照分析
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([2.45, 1.0], gap="large")
    with left:
        st.plotly_chart(render_distribution(history, target, predicted_cost), width="stretch")
    with right:
        right_html = f"""
        <div class="analysis-card">
            <h3>📈 图像说明</h3>
            <p>该图展示了当前报价在公司历史报价库中的相对位置。</p>
            <ul>
                <li><b>蓝色阴影：</b>代表历史订单的成本分布。</li>
                <li><b>红色虚线：</b>代表您当前的核价结果。</li>
            </ul>
            <div class="analysis-divider"></div>
            <div class="analysis-rating-title">💡 评价：</div>
            <div class="analysis-rating">
                <span class="rating-tag" style="color:{color};">[{label}]</span>
                {text}
            </div>
            <div style="margin-top: 16px; color: #667085; font-size: 15px; line-height: 1.8;">
                当前成本位于历史数据的约 <b>{pct:.1f}%</b> 分位。
            </div>
        </div>
        """
        st.markdown(right_html, unsafe_allow_html=True)


st.set_page_config(page_title="无纺布新增类别成本预测系统", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

bundle = load_model_bundle()
bundle_format = str(bundle.get("bundle_format", ""))
if bundle_format != NEW_CATEGORY_BUNDLE_FORMAT:
    st.error("当前网页必须使用新类别模型，请检查 new_category_cost_model.joblib。")
    st.stop()
target_col = str(bundle.get("target", STRICT_TARGET))
is_preprocessed_model = bundle_format in {
    "preprocessed_huber_v1",
    STRICT_BUNDLE_FORMAT,
    NEW_CATEGORY_BUNDLE_FORMAT,
}
model_input_file = (
    "0730数据源.xlsx"
    if bundle_format == NEW_CATEGORY_BUNDLE_FORMAT
    else str(bundle.get("input_file", RAW_HISTORY_FILE))
)
model_input_sheet = str(
    bundle.get(
        "input_sheet",
        "最终数据"
        if bundle_format == NEW_CATEGORY_BUNDLE_FORMAT
        else "预处理数据"
        if bundle_format == STRICT_BUNDLE_FORMAT
        else "",
    )
)
history_df = load_history(
    model_input_file,
    target_col,
    model_input_sheet,
    use_quote_fallback=not is_preprocessed_model,
)
avg_cost = float(history_df[target_col].mean())
median_cost = float(history_df[target_col].median())
quote_options = load_quote_options()
product_options = quote_options["products"]
package_options = quote_options["packages"]
bottom_parameter_options = quote_options.get("bottom", {})
if not isinstance(bottom_parameter_options, dict):
    bottom_parameter_options = {}
sterilization_options = list(
    unique_values(
        bottom_parameter_options.get("灭菌方式", []),
        STERILIZATION_OPTIONS,
    )
)

st.title("无纺布产品成本智能预测系统")
st.info("本系统对生产工艺参数进行预测，可以自动识别非线性关系并剔除无关干扰因素。")

if product_options.empty or package_options.empty:
    st.error(f"未能从 {QUOTE_PARAMETER_FILE} 读取产品或包装组合选项，请确认文件和工作表名称。")
    st.stop()

st.markdown('<div class="section-header">报价参数配置</div>', unsafe_allow_html=True)

choice_col1, choice_col2, choice_col3 = st.columns([1.35, 0.65, 1.6], gap="large")
with choice_col1:
    product_index = select_config(
        "产品标准配置",
        product_options,
        "产品组合选项",
        "product_standard_config",
        "搜索规格、层数、克重或配比",
    )
with choice_col2:
    sterilization = st.selectbox(
        "灭菌方式",
        sterilization_options,
        index=sterilization_options.index("EO灭菌"),
        key="sterilization_method",
    )
with choice_col3:
    package_index = select_config(
        "包装标准配置",
        package_options,
        "包装组合选项",
        "package_standard_config",
        "搜索装量、包装方式或材质",
    )

selected_product = product_options.loc[product_index]
selected_package = package_options.loc[package_index]
product_config_label = str(selected_product["产品组合选项"])
package_config_label = str(selected_package["包装组合选项"])

with st.container(border=True):
    st.markdown("#### 产品底层参数")
    product_cols = st.columns(6)
    with product_cols[0]:
        length = select_single(
            "长（cm）",
            list(bottom_parameter_options.get("长（cm）", column_options(product_options, "长（cm）"))),
            row_value(selected_product, "长（cm）", 7.5),
            f"product_length_{product_index}",
        )
    with product_cols[1]:
        width = select_single(
            "宽（cm）",
            list(bottom_parameter_options.get("宽（cm）", column_options(product_options, "宽（cm）"))),
            row_value(selected_product, "宽（cm）", 7.5),
            f"product_width_{product_index}",
        )
    with product_cols[2]:
        layers = select_single(
            "层数",
            list(bottom_parameter_options.get("层数", column_options(product_options, "层数"))),
            row_value(selected_product, "层数", 4),
            f"product_layers_{product_index}",
        )
    with product_cols[3]:
        gram = select_single(
            "克重g/㎡",
            list(bottom_parameter_options.get("克重g/㎡", column_options(product_options, "克重g/㎡"))),
            row_value(selected_product, "克重g/㎡", 30),
            f"product_gram_{product_index}",
        )
    with product_cols[4]:
        viscose = select_single(
            "粘胶配比%",
            list(bottom_parameter_options.get("粘胶配比%", column_options(product_options, "粘胶配比%"))),
            row_value(selected_product, "粘胶配比%", 50),
            f"product_viscose_{product_index}",
        )
    with product_cols[5]:
        polyester = select_single(
            "涤纶配比%",
            list(bottom_parameter_options.get("涤纶配比%", column_options(product_options, "涤纶配比%"))),
            row_value(selected_product, "涤纶配比%", 50),
            f"product_polyester_{product_index}",
        )

with st.container(border=True):
    st.markdown("#### 包装底层参数")
    inner_cols = st.columns(3)
    with inner_cols[0]:
        inner_qty = select_single(
            "内包装 装量",
            list(bottom_parameter_options.get("内包装 装量", column_options(package_options, "内包装 装量"))),
            row_value(selected_package, "内包装 装量", "1片"),
            f"package_inner_qty_{package_index}",
        )
    with inner_cols[1]:
        inner_material = select_single(
            "内包装 方式（材质）",
            list(bottom_parameter_options.get("内包装 方式 （材质）", column_options(package_options, "内包装 方式 （材质）"))),
            row_value(selected_package, "内包装 方式 （材质）", "纸+纸"),
            f"package_inner_material_{package_index}",
        )
    with inner_cols[2]:
        inner_colors = select_single(
            "内包装 印刷色数",
            list(bottom_parameter_options.get("内包装 印刷色数", column_options(package_options, "内包装 印刷色数"))),
            row_value(selected_package, "内包装 印刷色数", "1色印刷"),
            f"package_inner_colors_{package_index}",
        )

    middle_cols = st.columns(3)
    with middle_cols[0]:
        middle_qty = quantity_number_input(
            "中包装 数量",
            row_value(selected_package, "中包装 装量", "10袋"),
            "袋",
            f"package_middle_qty_{package_index}",
        )
    with middle_cols[1]:
        middle_material_default = selected_package.get("中包装 方式 （材质）")
        if (
            middle_material_default is None
            or (
                isinstance(middle_material_default, (float, np.floating))
                and np.isnan(middle_material_default)
            )
            or str(middle_material_default).strip().lower() in {"", "nan"}
        ):
            middle_material_default = MIDDLE_MATERIAL_OPTIONS[0]
        elif str(middle_material_default).strip() in {"/", "无中包装"}:
            middle_material_default = "/"
        middle_material = select_middle_material(
            middle_material_default,
            f"package_middle_material_{package_index}",
        )
    with middle_cols[2]:
        middle_colors = select_single(
            "中包装 印刷色数",
            list(bottom_parameter_options.get("中包装 印刷色数", column_options(package_options, "中包装 印刷色数", ["0色印刷"]))),
            row_value(selected_package, "中包装 印刷色数", "1色印刷"),
            f"package_middle_colors_{package_index}",
        )

    outer_cols = st.columns(3)
    with outer_cols[0]:
        outer_qty = quantity_number_input(
            "外包装 数量",
            row_value(selected_package, "外包装 装量", "10盒"),
            "盒",
            f"package_outer_qty_{package_index}",
        )
    with outer_cols[1]:
        outer_material = select_single(
            "外包装 方式（材质）",
            list(bottom_parameter_options.get("外包装 方式 （材质）", column_options(package_options, "外包装 方式 （材质）"))),
            row_value(selected_package, "外包装 方式 （材质）", "纸箱"),
            f"package_outer_material_{package_index}",
        )
    with outer_cols[2]:
        outer_colors = select_single(
            "外包装 印刷色数",
            list(bottom_parameter_options.get("外包装 印刷色数", column_options(package_options, "外包装 印刷色数", ["0色印刷"]))),
            row_value(selected_package, "外包装 印刷色数", "1色印刷"),
            f"package_outer_colors_{package_index}",
        )

predict_clicked = st.button(
    "开始预测",
    type="primary",
)

low_sample_reasons: list[str] = []
if normalize_sterilization(sterilization) in LOW_SAMPLE_STERILIZATION:
    low_sample_reasons.append(f"灭菌方式“{normalize_sterilization(sterilization)}”")
canonical_selected_middle_material = canonical_middle_material(middle_material)
if canonical_selected_middle_material in LOW_SAMPLE_MIDDLE_MATERIALS:
    low_sample_reasons.append(f"中包装材质“{canonical_selected_middle_material}”")

input_payload = {
    "产品标准配置": product_config_label,
    "包装标准配置": package_config_label,
    "长*宽（cm）": f"{display_value(length)}*{display_value(width)}",
    "长（cm）": to_float(length),
    "宽（cm）": to_float(width),
    "层数": to_float(layers),
    "克重g/㎡": to_float(gram),
    "粘胶配比%": to_float(viscose),
    "涤纶配比%": to_float(polyester),
    "灭菌方式": sterilization,
    "内包装 装量": inner_qty,
    "内包装 方式（材质）": inner_material,
    "内包装 印刷色数": inner_colors,
    "中包装 装量": middle_qty,
    "中包装 方式（材质）": middle_material,
    "中包装 印刷色数": middle_colors,
    "外包装 装量": outer_qty,
    "外包装 方式（材质）": outer_material,
    "外包装 印刷色数": outer_colors,
}

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if predict_clicked:
    input_df, derived_payload = build_input_row(input_payload, bundle)
    model_feature_columns = (
        list(bundle["original_feature_columns"])
        if bundle_format in {STRICT_BUNDLE_FORMAT, NEW_CATEGORY_BUNDLE_FORMAT}
        else list(bundle["feature_columns"])
        if bundle_format == "preprocessed_huber_v1"
        else None
    )
    history_match = find_history_match(
        history_df,
        input_df if is_preprocessed_model else input_payload,
        target_col,
        model_feature_columns,
    )
    if not history_match.empty:
        predicted_cost = float(history_match[target_col].iloc[0])
        predicted_cost = float(
            _apply_outer_color_increment(
                bundle,
                input_df,
                np.asarray([predicted_cost], dtype=float),
            )[0]
        )
        log_prediction = float(np.log(predicted_cost))
        source = "历史完全匹配（已做外箱印刷单调校正）"
        st.success("匹配到历史完全一致记录，本次结果已按外箱印刷色数递增规则校正。")
    else:
        predicted_cost, log_prediction = predict_huber(bundle, input_df)
        source = "AI模型预测"

    prediction_id = save_prediction(
        input_payload,
        derived_payload,
        predicted_cost,
        log_prediction,
        avg_cost,
        median_cost,
        source,
    )
    analysis = coefficient_analysis(bundle, input_df)
    st.session_state.last_prediction = {
        "prediction_id": prediction_id,
        "predicted_cost": predicted_cost,
        "log_prediction": log_prediction,
        "source": source,
        "input_payload": input_payload,
        "derived_payload": derived_payload,
        "analysis": analysis,
    }
    if low_sample_reasons:
        show_low_sample_warning(low_sample_reasons)

last = st.session_state.last_prediction
if last:
    predicted_cost = float(last["predicted_cost"])

    st.markdown('<div class="section-header">成本预测结果</div>', unsafe_allow_html=True)
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.markdown(
            f"""
            <div class="price-box">
                <p style="color:#596b7a; margin:0; font-size:14px;">预测成本单价（PCS）</p>
                <h2>{predicted_cost:.6f}</h2>
                <p style="color:#596b7a; margin:6px 0 0 0;">{last["source"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        klass, text = prediction_judgment(predicted_cost, history_df[target_col])
        st.markdown(f'<div class="judgment-box {klass}">{text}</div>', unsafe_allow_html=True)

        st.markdown("#### 人工确认成本")
        with st.form("feedback_form", clear_on_submit=False):
            actual_cost = st.number_input(
                "人工确认后的准确成本",
                min_value=0.0,
                value=predicted_cost,
                step=0.0001,
                format="%.6f",
            )
            reviewer = st.text_input("确认人", value="")
            note = st.text_area("备注", value="", height=80)
            submitted = st.form_submit_button("保存人工确认结果")
            if submitted:
                save_feedback(
                    str(last["prediction_id"]),
                    predicted_cost,
                    float(actual_cost),
                    reviewer.strip(),
                    note.strip(),
                    dict(last["input_payload"]),
                )
                st.success("人工确认成本已保存")

    with res_col2:
        analysis_df = last["analysis"]
        if bundle_format in {STRICT_BUNDLE_FORMAT, NEW_CATEGORY_BUNDLE_FORMAT}:
            display_analysis_df = analysis_df.copy()
            if "影响权重" not in display_analysis_df.columns:
                total_sensitivity = float(display_analysis_df["绝对敏感度"].sum())
                display_analysis_df["影响权重"] = (
                    display_analysis_df["绝对敏感度"] / total_sensitivity * 100.0
                    if total_sensitivity > 0
                    else 0.0
                )
            fig = px.bar(
                display_analysis_df.sort_values("影响权重"),
                x="影响权重",
                y="特征",
                orientation="h",
                color="影响方向",
                color_discrete_map={"推高预测": "#c1121f", "降低预测": "#087f5b"},
                text="影响权重",
                title="各工艺参数对成本的影响权重",
            )
            fig.update_traces(
                texttemplate="%{text:.2f}%",
                textposition="outside",
                cliponaxis=False,
            )
            analysis_columns = [
                "特征",
                "影响权重",
                "影响方向",
            ]
            display_table_df = display_analysis_df[analysis_columns].copy()
            display_table_df["影响权重"] = display_table_df["影响权重"].map(
                lambda value: f"{float(value):.2f}%"
            )
        else:
            fig = px.bar(
                analysis_df.sort_values("成本贡献"),
                x="成本贡献",
                y="特征",
                orientation="h",
                color="成本贡献",
                color_continuous_scale="RdYlGn_r",
                title="各工艺参数对成本的影响权重",
            )
            analysis_columns = ["特征", "模型输入值", "系数", "成本贡献", "影响方向"]
            display_table_df = analysis_df[analysis_columns]
        fig.update_layout(
            height=430,
            margin={"l": 20, "r": 20, "t": 55, "b": 20},
            title={
                "text": "各工艺参数对成本的影响权重",
                "x": 0.5,
                "xanchor": "center",
            },
            coloraxis_showscale=False,
            plot_bgcolor="white",
        )
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            display_table_df,
            width="stretch",
            hide_index=True,
        )

    render_history_analysis_section(history_df, target_col, predicted_cost)

with st.expander("数据库记录", expanded=False):
    tab1, tab2 = st.tabs(["人工确认记录", "预测记录"])
    with tab1:
        feedback_df = read_feedback()
        if feedback_df.empty:
            st.info("还没有人工确认记录。")
        else:
            feedback_df["误差率"] = feedback_df["误差率"].map(lambda x: f"{x * 100:+.2f}%")
            st.dataframe(feedback_df, width="stretch", hide_index=True)
            st.download_button(
                "下载人工确认记录 CSV",
                data=feedback_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="cost_feedback_records.csv",
                mime="text/csv",
            )
    with tab2:
        predictions_df = read_predictions()
        if predictions_df.empty:
            st.info("还没有预测记录。")
        else:
            st.dataframe(predictions_df, width="stretch", hide_index=True)
            st.download_button(
                "下载预测记录 CSV",
                data=predictions_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="cost_prediction_records.csv",
                mime="text/csv",
            )
