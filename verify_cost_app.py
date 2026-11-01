"""Exercise the actual Streamlit workflow against a disposable database copy."""
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from streamlit.testing.v1 import AppTest

from cost_database import CostDatabase, PRODUCT_FIELDS, number


def main():
    root = Path(__file__).resolve().parent
    database = CostDatabase(root / 'new_category_cost_feedback.db')
    current = database.prediction_records(10000).set_index('预测ID')
    with closing(sqlite3.connect(database.path)) as conn:
        conn.row_factory = sqlite3.Row
        for old in conn.execute('SELECT * FROM predictions_legacy_v1'):
            row = current.loc[old['id']]
            assert row['预测成本'] == old['prediction']
            inputs = json.loads(row['input_json'])
            original = json.loads(old['input_json'])
            for label in PRODUCT_FIELDS.values():
                assert inputs[label] == number(original.get(label))
            for prefix in ('内包装','中包装','外包装'):
                assert inputs[prefix+' 方式（材质）'] == original[prefix+' 方式（材质）']
                for suffix in (' 装量',' 印刷色数'):
                    assert number(inputs[prefix+suffix]) == number(original[prefix+suffix])
        assert not conn.execute('PRAGMA foreign_key_check').fetchall()
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'verification.db'
        with closing(sqlite3.connect(database.path)) as src, closing(sqlite3.connect(path)) as dest:
            src.backup(dest)
        source = (root / 'new_category_cost_app.py').read_text(encoding='utf-8')
        source = source.replace('APP_DIR = Path(__file__).resolve().parent', f'APP_DIR = Path({str(root)!r})')
        source = source.replace('DB_PATH = APP_DIR / "new_category_cost_feedback.db"', f'DB_PATH = Path({str(path)!r})')
        # The test must never connect to a deployment database through environment/secrets.
        source = source.replace('DATABASE_URL = normalize_database_url(DATABASE_URL)', 'DATABASE_URL = ""')
        app = AppTest.from_string(source, default_timeout=90).run()
        assert not app.exception, [e.message for e in app.exception]
        app.button[0].click().run()
        assert not app.exception, [e.message for e in app.exception]
        submit = next(b for b in app.button if b.label == '保存人工确认结果')
        submit.click().run()
        assert not app.exception, [e.message for e in app.exception]
        with closing(sqlite3.connect(path)) as conn:
            assert conn.execute('SELECT COUNT(*) FROM predictions').fetchone()[0] == len(current)+1
            assert conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0] == 1
            assert not conn.execute('PRAGMA foreign_key_check').fetchall()
    print(f'PASS: {len(current)} migrated records reconciled; app prediction and feedback workflow passed.')


if __name__ == '__main__':
    main()
