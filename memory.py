"""Historian: persistent research memory (SQLite). Everything is recorded; nothing is deleted."""
from __future__ import annotations
import json
import sqlite3
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS strategies(
  id INTEGER PRIMARY KEY AUTOINCREMENT, parents TEXT NOT NULL, generation INTEGER NOT NULL,
  origin TEXT NOT NULL, family TEXT NOT NULL, fingerprint TEXT NOT NULL, genome TEXT NOT NULL,
  stage TEXT NOT NULL DEFAULT 'UNKNOWN', status TEXT NOT NULL DEFAULT 'alive',
  status_reason TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS hypotheses(
  id INTEGER PRIMARY KEY AUTOINCREMENT, generation INTEGER, text TEXT NOT NULL, strategy_id INTEGER, created REAL);
CREATE TABLE IF NOT EXISTS states(
  id INTEGER PRIMARY KEY AUTOINCREMENT, generation INTEGER, vector TEXT NOT NULL,
  novelty REAL, reason TEXT, regime TEXT, created REAL);
CREATE TABLE IF NOT EXISTS experiments(
  id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id INTEGER NOT NULL, generation INTEGER NOT NULL,
  state_id INTEGER, family TEXT, hypothesis TEXT, metrics TEXT NOT NULL, win INTEGER,
  learning_efficiency REAL, lesson TEXT, created REAL);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, generation INTEGER, kind TEXT, detail TEXT, created REAL);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS holdout_peeks(
  id INTEGER PRIMARY KEY AUTOINCREMENT, strategy_id INTEGER, created REAL);
"""


class Memory:
    def __init__(self, path: str = ":memory:"):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    # ---- writes
    def add_strategy(self, genome, parents=(), generation=0, origin="random") -> int:
        cur = self.db.execute(
            "INSERT INTO strategies(parents,generation,origin,family,fingerprint,genome,created) VALUES(?,?,?,?,?,?,?)",
            (json.dumps(list(parents)), generation, origin, genome.family(), genome.fingerprint(),
             json.dumps(genome.to_dict()), time.time()))
        self.db.commit()
        return cur.lastrowid

    def add_hypothesis(self, text, strategy_id, generation) -> int:
        cur = self.db.execute("INSERT INTO hypotheses(generation,text,strategy_id,created) VALUES(?,?,?,?)",
                              (generation, text, strategy_id, time.time()))
        self.db.commit()
        return cur.lastrowid

    def add_state(self, generation, vector, novelty, reason, regime="") -> int:
        cur = self.db.execute("INSERT INTO states(generation,vector,novelty,reason,regime,created) VALUES(?,?,?,?,?,?)",
                              (generation, json.dumps([float(x) for x in vector]), novelty, reason, regime, time.time()))
        self.db.commit()
        return cur.lastrowid

    def add_experiment(self, strategy_id, generation, state_id, family, hypothesis, metrics, win, le, lesson) -> int:
        cur = self.db.execute(
            "INSERT INTO experiments(strategy_id,generation,state_id,family,hypothesis,metrics,win,learning_efficiency,lesson,created)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (strategy_id, generation, state_id, family, hypothesis, json.dumps(metrics), int(win), le, lesson, time.time()))
        self.db.commit()
        return cur.lastrowid

    def set_status(self, strategy_id, status=None, stage=None, reason=None):
        if status is not None:
            self.db.execute("UPDATE strategies SET status=?, status_reason=? WHERE id=?", (status, reason, strategy_id))
        if stage is not None:
            self.db.execute("UPDATE strategies SET stage=? WHERE id=?", (stage, strategy_id))
        self.db.commit()

    def log(self, generation, kind, detail):
        self.db.execute("INSERT INTO events(generation,kind,detail,created) VALUES(?,?,?,?)",
                        (generation, kind, detail, time.time()))
        self.db.commit()

    def log_holdout_peek(self, strategy_id):
        self.db.execute("INSERT INTO holdout_peeks(strategy_id,created) VALUES(?,?)", (strategy_id, time.time()))
        self.db.commit()

    def set_meta(self, key, value):
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, str(value)))
        self.db.commit()

    # ---- reads
    def get_meta(self, key, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    def get_strategy(self, sid):
        return self.db.execute("SELECT * FROM strategies WHERE id=?", (sid,)).fetchone()

    def lineage(self, sid) -> list:
        """Ancestor chain from oldest root to this strategy (first-parent path)."""
        chain = []
        cur = sid
        while cur is not None:
            row = self.get_strategy(cur)
            if row is None:
                break
            chain.append(cur)
            parents = json.loads(row["parents"])
            cur = parents[0] if parents else None
        return list(reversed(chain))

    def state_archive(self) -> list:
        return [json.loads(r["vector"]) for r in self.db.execute("SELECT vector FROM states ORDER BY id")]

    def family_record(self, family) -> tuple:
        r = self.db.execute("SELECT COALESCE(SUM(win),0) w, COUNT(*) n FROM experiments WHERE family=?", (family,)).fetchone()
        return int(r["w"]), int(r["n"]) - int(r["w"])

    def similar_state_report(self, vector, k=5, family=None) -> dict:
        """'I've seen a similar state N times; strategies of this family won W, lost L.'"""
        import numpy as np
        rows = list(self.db.execute("SELECT id, vector FROM states"))
        if not rows:
            return {"similar_states": 0, "wins": 0, "losses": 0}
        V = np.array([json.loads(r["vector"]) for r in rows])
        sd = V.std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        d = np.linalg.norm((V - np.array(vector)) / sd, axis=1)
        ids = [rows[i]["id"] for i in np.argsort(d)[:k]]
        q = f"SELECT win FROM experiments WHERE state_id IN ({','.join('?' * len(ids))})"
        args = list(ids)
        if family:
            q += " AND family=?"
            args.append(family)
        wins = [r["win"] for r in self.db.execute(q, args)]
        return {"similar_states": len(ids), "wins": int(sum(wins)), "losses": len(wins) - int(sum(wins))}

    def counts(self) -> dict:
        one = lambda q: self.db.execute(q).fetchone()[0]
        return {"strategies": one("SELECT COUNT(*) FROM strategies"),
                "experiments": one("SELECT COUNT(*) FROM experiments"),
                "hypotheses": one("SELECT COUNT(*) FROM hypotheses"),
                "discarded": one("SELECT COUNT(*) FROM strategies WHERE status='discarded'"),
                "holdout_peeks": one("SELECT COUNT(*) FROM holdout_peeks")}
