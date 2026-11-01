import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from cost_database import CostDatabase


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'cost.db'
        self.db = CostDatabase(self.path)
        self.inputs = {'长（cm）':5,'宽（cm）':5,'层数':4,'克重g/㎡':30,
            '粘胶配比%':40,'涤纶配比%':60,'灭菌方式':'EO灭菌',
            '内包装 装量':'1片','内包装 方式（材质）':'纸+纸','内包装 印刷色数':'1色印刷',
            '中包装 装量':'10袋','中包装 方式（材质）':'中盒-250g/㎡灰底白板','中包装 印刷色数':'0色印刷',
            '外包装 装量':'10盒','外包装 方式（材质）':'纸箱','外包装 印刷色数':'0色印刷'}

    def tearDown(self):
        self.temp.cleanup()

    def save(self, inputs=None):
        return self.db.save_prediction(inputs or self.inputs, {}, 0.2, -1.6, 0.3, 0.25,
                                       'AI', {'id':'test-model','name':'test'})

    def test_reuse_and_versioned_configuration(self):
        first = self.save()
        self.save()
        changed = dict(self.inputs, **{'长（cm）':6})
        self.save(changed)
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM product_config').fetchone()[0],2)
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM packaging_config').fetchone()[0],1)
        record = self.db.prediction_records(200).set_index('预测ID').loc[first]
        self.assertEqual(json.loads(record['input_json'])['长（cm）'],5)

    def test_feedback_join_zero_and_foreign_key(self):
        key = self.save()
        self.db.save_feedback(key, 0.25, 'reviewer', '')
        self.db.save_feedback(key, 0, 'reviewer', '')
        rows = self.db.read('SELECT * FROM feedback_details ORDER BY actual_cost')
        self.assertTrue(rows.iloc[0]['error_rate'] != rows.iloc[0]['error_rate'])
        self.assertAlmostEqual(rows.iloc[1]['error'],0.05)
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.save_feedback('missing', 1, '', '')
        self.assertEqual(len(self.db.read('SELECT * FROM feedback')),2)

    def legacy_database(self, invalid=False):
        with closing(sqlite3.connect(self.path)) as conn:
            conn.execute('''CREATE TABLE predictions(id TEXT PRIMARY KEY, created_at TEXT,
                input_json TEXT, derived_json TEXT, prediction REAL, log_prediction REAL,
                avg_cost REAL, median_cost REAL, source TEXT)''')
            conn.execute('''CREATE TABLE feedback(id INTEGER PRIMARY KEY, prediction_id TEXT,
                created_at TEXT, predicted_cost REAL, actual_cost REAL, error REAL,
                error_rate REAL, reviewer TEXT, note TEXT, input_json TEXT,
                FOREIGN KEY(prediction_id) REFERENCES predictions(id))''')
            conn.execute('INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?,?)',
                ('original','2026-01-01','broken' if invalid else json.dumps(self.inputs),'{}',.2,-1.6,.3,.25,'AI'))
            conn.execute('INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?,?)',
                (1,'original','2026-01-02',.2,.25,.05,.2,'a','',json.dumps(self.inputs)))
            conn.commit()

    def test_migration_preserves_ids_results_and_is_idempotent(self):
        self.legacy_database()
        self.db.initialize()
        CostDatabase(self.path).initialize()
        self.assertEqual(self.db.read('SELECT id FROM predictions').iloc[0,0],'original')
        self.assertEqual(self.db.read('SELECT actual_cost FROM feedback').iloc[0,0],.25)
        self.assertEqual(len(list(self.path.parent.glob('*.bak'))),1)
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertEqual(conn.execute('PRAGMA foreign_key_check').fetchall(),[])

    def test_bad_migration_rolls_back(self):
        self.legacy_database(invalid=True)
        with self.assertRaises(json.JSONDecodeError):
            self.db.initialize()
        with closing(sqlite3.connect(self.path)) as conn:
            self.assertIn('input_json',[r[1] for r in conn.execute('PRAGMA table_info(predictions)')])
            self.assertEqual(conn.execute('SELECT COUNT(*) FROM feedback').fetchone()[0],1)


if __name__ == '__main__':
    unittest.main()
