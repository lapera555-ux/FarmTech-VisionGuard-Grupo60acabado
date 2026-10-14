"""Persistência local SQLite com snapshots, sem serviços externos."""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import io
import sqlite3
from contextlib import contextmanager, closing
from pathlib import Path

SCHEMA = '''
CREATE TABLE IF NOT EXISTS events (
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 occurred_utc TEXT NOT NULL,
 camera TEXT NOT NULL,
 object_class TEXT NOT NULL CHECK(object_class IN ('person','car')),
 confidence REAL NOT NULL CHECK(confidence>=0 AND confidence<=1),
 zone_match INTEGER NOT NULL CHECK(zone_match IN (0,1)),
 snapshot TEXT NOT NULL,
 model_mode TEXT NOT NULL,
 source_kind TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_date ON events(occurred_utc DESC);
'''

class EventStore:
    def __init__(self, root: Path):
        self.root=Path(root).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.db=self.root/'events.sqlite3'
        self.snaps=self.root/'snapshots'
        self.snaps.mkdir(exist_ok=True)
        with self._connect() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(str(self.db), timeout=8)) as connection:
            with connection:
                yield connection

    def record(self,camera: str,label: str,confidence: float,snapshot: str,model_mode: str,source_kind: str) -> int:
        if label not in ('person','car') or not 0<=confidence<=1:
            raise ValueError('Evento inválido')
        when=datetime.now(timezone.utc).isoformat(timespec='seconds')
        with self._connect() as c:
            cur=c.execute('INSERT INTO events (occurred_utc,camera,object_class,confidence,zone_match,snapshot,model_mode,source_kind) VALUES (?,?,?,?,?,?,?,?)',
                (when,str(camera)[:80],label,float(confidence),1,str(snapshot),str(model_mode),str(source_kind)))
            return int(cur.lastrowid)

    def recent(self,limit=200):
        limit=max(1,min(int(limit),5000))
        with self._connect() as c:
            c.row_factory=sqlite3.Row
            return [dict(r) for r in c.execute('SELECT * FROM events ORDER BY id DESC LIMIT ?', (limit,))]

    def counts(self):
        with self._connect() as c:
            d=dict(c.execute('SELECT object_class,COUNT(*) FROM events GROUP BY object_class'))
        return {k:int(d.get(k,0)) for k in ('person','car')}

    def csv_bytes(self, limit=5000):
        data=self.recent(limit)
        buf=io.StringIO()
        cols=['id','occurred_utc','camera','object_class','confidence','zone_match','snapshot','model_mode','source_kind']
        writer=csv.DictWriter(buf, fieldnames=cols)
        writer.writeheader();writer.writerows(data)
        return buf.getvalue().encode('utf-8-sig')
