"""Normalized cost storage and transactional migration for SQLite/PostgreSQL."""
from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import uuid
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path


PRODUCT_FIELDS = {
    "length_cm": "长（cm）", "width_cm": "宽（cm）", "layers": "层数",
    "grammage": "克重g/㎡", "viscose_pct": "粘胶配比%", "polyester_pct": "涤纶配比%",
}
SCHEMA = [
    "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)",
    """CREATE TABLE IF NOT EXISTS product_config (
        id TEXT PRIMARY KEY, length_cm REAL, width_cm REAL, layers REAL,
        grammage REAL, viscose_pct REAL, polyester_pct REAL)""",
    """CREATE TABLE IF NOT EXISTS packaging_material (
        id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, category TEXT NOT NULL,
        paper_grammage REAL)""",
    """CREATE TABLE IF NOT EXISTS packaging_config (
        id TEXT PRIMARY KEY, version INTEGER NOT NULL DEFAULT 1)""",
    """CREATE TABLE IF NOT EXISTS packaging_layer (
        id TEXT PRIMARY KEY, packaging_config_id TEXT NOT NULL REFERENCES packaging_config(id),
        layer TEXT NOT NULL CHECK(layer IN ('inner','middle','outer')),
        material_id TEXT NOT NULL REFERENCES packaging_material(id),
        quantity REAL, quantity_unit TEXT, print_colors INTEGER,
        CHECK(quantity IS NULL OR quantity >= 0),
        CHECK(print_colors IS NULL OR print_colors >= 0),
        UNIQUE(packaging_config_id, layer))""",
    """CREATE TABLE IF NOT EXISTS model_version (
        id TEXT PRIMARY KEY, name TEXT NOT NULL, file_sha256 TEXT,
        feature_version TEXT, registered_at TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
        product_config_id TEXT NOT NULL REFERENCES product_config(id),
        packaging_config_id TEXT NOT NULL REFERENCES packaging_config(id),
        model_version_id TEXT NOT NULL REFERENCES model_version(id),
        sterilization TEXT, product_label TEXT, packaging_label TEXT,
        input_snapshot TEXT NOT NULL, derived_json TEXT NOT NULL,
        prediction REAL NOT NULL, log_prediction REAL NOT NULL,
        avg_cost REAL, median_cost REAL, source TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY, prediction_id TEXT NOT NULL REFERENCES predictions(id),
        created_at TEXT NOT NULL, actual_cost REAL NOT NULL CHECK(actual_cost >= 0),
        reviewer TEXT, note TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_feedback_prediction ON feedback(prediction_id, created_at)",
    """CREATE VIEW IF NOT EXISTS prediction_details AS
        SELECT p.*, c.length_cm, c.width_cm, c.layers, c.grammage,
        c.viscose_pct, c.polyester_pct FROM predictions p
        JOIN product_config c ON c.id = p.product_config_id""",
    """CREATE VIEW IF NOT EXISTS feedback_details AS
        SELECT f.*, p.prediction AS predicted_cost,
        f.actual_cost-p.prediction AS error,
        CASE WHEN f.actual_cost=0 THEN NULL
             ELSE (f.actual_cost-p.prediction)/ABS(f.actual_cost) END AS error_rate
        FROM feedback f JOIN predictions p ON p.id=f.prediction_id""",
]


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     allow_nan=False).encode()).hexdigest()


def number(value):
    if value is None or str(value).strip() in ('', '/', '无', 'None'):
        return None
    match = re.search(r'[-+]?\d+(?:\.\d+)?', str(value))
    if not match:
        raise ValueError(f'Cannot parse numeric value: {value!r}')
    result = float(match.group())
    if not math.isfinite(result):
        raise ValueError('Non-finite numeric value')
    return result


class CostDatabase:
    def __init__(self, path, engine=None):
        self.path = Path(path)
        self.engine = engine
        self.ready = False

    @contextmanager
    def transaction(self):
        if self.engine is not None:
            with self.engine.begin() as conn:
                yield conn
        else:
            with closing(sqlite3.connect(self.path, timeout=30)) as conn:
                with conn:
                    conn.execute('PRAGMA foreign_keys=ON')
                    conn.execute('BEGIN IMMEDIATE')
                    yield conn

    def execute(self, conn, sql, params=None):
        if self.engine is not None:
            from sqlalchemy import text
            return conn.execute(text(sql), params or {})
        return conn.execute(sql, params or {})

    def rows(self, conn, sql, params=None):
        result = self.execute(conn, sql, params)
        keys = list(result.keys()) if self.engine is not None else [x[0] for x in result.description]
        return [dict(zip(keys, row)) for row in result.fetchall()]

    def put(self, conn, table, values):
        columns = ', '.join(values)
        params = ', '.join(':'+key for key in values)
        self.execute(conn, f'INSERT INTO {table} ({columns}) VALUES ({params}) ON CONFLICT DO NOTHING', values)

    def initialize(self):
        if self.ready:
            return
        if self.engine is None and self.path.exists():
            with closing(sqlite3.connect(self.path)) as conn:
                old = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'").fetchone()
                legacy = old and 'input_json' in [r[1] for r in conn.execute('PRAGMA table_info(predictions)')]
                if legacy:
                    backup = self.path.with_name(self.path.name + '.before_v2.' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.bak')
                    with closing(sqlite3.connect(backup)) as dest:
                        conn.backup(dest)
        with self.transaction() as conn:
            if self.engine is not None:
                self.execute(conn, 'SELECT pg_advisory_xact_lock(7302026)')
                columns = self.rows(conn, "SELECT column_name FROM information_schema.columns WHERE table_schema=current_schema() AND table_name='predictions'")
                legacy = any(r['column_name']=='input_json' for r in columns)
            else:
                legacy = any(r['name']=='input_json' for r in self.rows(conn, 'PRAGMA table_info(predictions)'))
            if legacy:
                self.execute(conn, 'ALTER TABLE feedback RENAME TO feedback_legacy_v1')
                self.execute(conn, 'ALTER TABLE predictions RENAME TO predictions_legacy_v1')
            for sql in SCHEMA:
                self.execute(conn, sql)
            if legacy:
                for row in self.rows(conn, 'SELECT * FROM predictions_legacy_v1'):
                    self.insert_prediction(conn, json.loads(row['input_json']),
                        json.loads(row['derived_json']), row, {'id':'legacy-unknown', 'name':'Legacy model (unknown)'})
                for row in self.rows(conn, 'SELECT * FROM feedback_legacy_v1'):
                    self.put(conn, 'feedback', {k: str(row[k]) if k=='id' else row[k]
                        for k in ('id','prediction_id','created_at','actual_cost','reviewer','note')})
                for table in ('predictions','feedback'):
                    counts = self.rows(conn, f'SELECT COUNT(*) AS n FROM {table}')[0]['n']
                    old_counts = self.rows(conn, f'SELECT COUNT(*) AS n FROM {table}_legacy_v1')[0]['n']
                    if counts != old_counts:
                        raise RuntimeError('Migration record count mismatch')
            self.put(conn, 'schema_migrations', {'version':2, 'applied_at':datetime.now().isoformat()})
        self.ready = True

    def insert_prediction(self, conn, inputs, derived, values, model):
        product = {key:number(inputs.get(label)) for key,label in PRODUCT_FIELDS.items()}
        product_id = identity(product)
        self.put(conn, 'product_config', {'id':product_id, **product})
        layers = []
        for layer, prefix in [('inner','内包装'),('middle','中包装'),('outer','外包装')]:
            material = str(inputs.get(prefix+' 方式（材质）') or '/')
            qty = inputs.get(prefix+' 装量')
            unit = re.sub(r'[-+]?\d+(?:\.\d+)?', '', str(qty or '')).strip()
            colors = number(inputs.get(prefix+' 印刷色数'))
            if colors is not None and not colors.is_integer():
                raise ValueError('Print colors must be an integer')
            layers.append({'layer':layer,'material':material,'quantity':number(qty),
                           'quantity_unit':unit,'print_colors':int(colors) if colors is not None else None})
        packaging_id = identity(layers)
        self.put(conn, 'packaging_config', {'id':packaging_id,'version':1})
        for item in layers:
            item = dict(item)
            material = item.pop('material')
            material_id = identity(material)
            paper = re.search(r'(\d+(?:\.\d+)?)\s*g', material, re.I)
            category = next((x for x in ('灰底白板','白卡','瓦楞','纸箱','塑袋','纸+纸','纸+塑') if x in material), material)
            self.put(conn, 'packaging_material', {'id':material_id,'name':material,
                     'category':category,'paper_grammage':float(paper.group(1)) if paper else None})
            self.put(conn, 'packaging_layer', {'id':identity([packaging_id,item['layer']]),
                     'packaging_config_id':packaging_id,'material_id':material_id,**item})
        self.put(conn, 'model_version', {'file_sha256':None,'feature_version':None,
                 'registered_at':datetime.now().isoformat(),**model})
        row = {key:values[key] for key in ('id','created_at','prediction','log_prediction','avg_cost','median_cost','source')}
        self.put(conn, 'predictions', {**row,'product_config_id':product_id,
                 'packaging_config_id':packaging_id,'model_version_id':model['id'],
                 'sterilization':inputs.get('灭菌方式'),'product_label':inputs.get('产品标准配置'),
                 'packaging_label':inputs.get('包装标准配置'),
                 'input_snapshot':json.dumps(inputs,ensure_ascii=False,allow_nan=False),
                 'derived_json':json.dumps(derived,ensure_ascii=False,allow_nan=False)})

    def save_prediction(self, inputs, derived, prediction, log_prediction, avg_cost, median_cost, source, model):
        self.initialize()
        key = str(uuid.uuid4())
        with self.transaction() as conn:
            self.insert_prediction(conn, inputs, derived, dict(id=key,
                created_at=datetime.now().isoformat(timespec='microseconds'),prediction=prediction,
                log_prediction=log_prediction,avg_cost=avg_cost,median_cost=median_cost,source=source),model)
        return key

    def save_feedback(self, prediction_id, actual_cost, reviewer, note):
        self.initialize()
        if not math.isfinite(actual_cost) or actual_cost < 0:
            raise ValueError('Actual cost must be finite and nonnegative')
        with self.transaction() as conn:
            self.put(conn,'feedback',dict(id=str(uuid.uuid4()),prediction_id=prediction_id,
                created_at=datetime.now().isoformat(timespec='microseconds'),actual_cost=actual_cost,
                reviewer=reviewer,note=note))

    def read(self, sql, params=None):
        import pandas as pd
        self.initialize()
        with self.transaction() as conn:
            result = self.execute(conn,sql,params)
            columns = list(result.keys()) if self.engine is not None else [x[0] for x in result.description]
            return pd.DataFrame(result.fetchall(),columns=columns)

    def prediction_records(self, limit=200):
        records = self.read('''SELECT created_at, id, product_config_id,
            packaging_config_id, product_label, packaging_label, sterilization,
            length_cm, width_cm, layers, grammage, viscose_pct, polyester_pct,
            prediction, avg_cost, median_cost, source
            FROM prediction_details ORDER BY created_at DESC, id DESC LIMIT :limit''', {'limit':limit})
        packaging = self.read('''SELECT l.*, m.name AS material_name
            FROM packaging_layer l JOIN packaging_material m ON m.id=l.material_id
            WHERE l.packaging_config_id IN
            (SELECT packaging_config_id FROM predictions ORDER BY created_at DESC, id DESC LIMIT :limit)''', {'limit':limit})
        packages = {}
        def display(value):
            import pandas as pd
            return '' if pd.isna(value) else format(float(value), 'g')
        for row in packaging.to_dict('records'):
            prefix = {'inner':'内包装','middle':'中包装','outer':'外包装'}[row['layer']]
            package = packages.setdefault(row['packaging_config_id'], {})
            package[prefix+' 装量'] = display(row['quantity']) + (row['quantity_unit'] or '')
            package[prefix+' 方式（材质）'] = row['material_name']
            color_text = display(row['print_colors'])
            package[prefix+' 印刷色数'] = color_text+'色印刷' if color_text else None
        result = []
        for row in records.to_dict('records'):
            inputs = {label:row[key] for key,label in PRODUCT_FIELDS.items()}
            inputs.update({'产品标准配置':row['product_label'],'包装标准配置':row['packaging_label'],
                '灭菌方式':row['sterilization'],
                '长*宽（cm）':display(row['length_cm'])+'*'+display(row['width_cm'])})
            inputs.update(packages.get(row['packaging_config_id'], {}))
            result.append({'预测时间':row['created_at'],'预测ID':row['id'],
                'input_json':json.dumps(inputs,ensure_ascii=False),'预测成本':row['prediction'],
                '历史均值':row['avg_cost'],'历史中位数':row['median_cost'],'来源':row['source']})
        import pandas as pd
        return pd.DataFrame(result, columns=['预测时间','预测ID','input_json','预测成本','历史均值','历史中位数','来源'])
