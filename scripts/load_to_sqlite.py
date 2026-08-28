import csv
import os
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_DIR = ROOT / "db"
DB_PATH = DB_DIR / "store_sales.db"


def create_tables(conn):
    conn.executescript(
        """
        DROP TABLE IF EXISTS train;
        DROP TABLE IF EXISTS test;
        DROP TABLE IF EXISTS stores;
        DROP TABLE IF EXISTS oil;
        DROP TABLE IF EXISTS holidays_events;
        DROP TABLE IF EXISTS transactions;

        CREATE TABLE IF NOT EXISTS train (
            id INTEGER PRIMARY KEY,
            date TEXT,
            store_nbr INTEGER,
            family TEXT,
            sales REAL,
            onpromotion INTEGER
        );

        CREATE TABLE IF NOT EXISTS test (
            id INTEGER PRIMARY KEY,
            date TEXT,
            store_nbr INTEGER,
            family TEXT,
            onpromotion INTEGER
        );

        CREATE TABLE IF NOT EXISTS stores (
            store_nbr INTEGER PRIMARY KEY,
            city TEXT,
            state TEXT,
            type TEXT,
            cluster INTEGER
        );

        CREATE TABLE IF NOT EXISTS oil (
            date TEXT PRIMARY KEY,
            dcoilwtico REAL
        );

        CREATE TABLE IF NOT EXISTS holidays_events (
            id INTEGER PRIMARY KEY,
            date TEXT,
            type TEXT,
            locale TEXT,
            locale_name TEXT,
            description TEXT,
            transferred INTEGER
        );

        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY,
            date TEXT,
            store_nbr INTEGER,
            transactions INTEGER
        );
        """
    )


def load_csv(conn, table, csv_path, columns):
    if not csv_path.exists():
        print(f"跳过 {table}：未找到 {csv_path.name}")
        return 0

    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            values = []
            for col in columns:
                raw = row.get(col)
                if raw is None or raw == "":
                    values.append(None)
                elif table == "holidays_events" and col == "transferred":
                    values.append(1 if str(raw).lower() in ("1", "true", "1.0") else 0)
                else:
                    values.append(raw)
            rows.append(tuple(values))

    placeholders = ",".join(["?"] * len(columns))
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    conn.executemany(sql, rows)
    conn.commit()
    print(f"{table}: {len(rows)} 行")
    return len(rows)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    create_tables(conn)

    load_csv(conn, "train", DATA_DIR / "train.csv", ["date", "store_nbr", "family", "sales", "onpromotion"])
    load_csv(conn, "test", DATA_DIR / "test.csv", ["date", "store_nbr", "family", "onpromotion"])
    load_csv(conn, "stores", DATA_DIR / "stores.csv", ["store_nbr", "city", "state", "type", "cluster"])
    load_csv(conn, "oil", DATA_DIR / "oil.csv", ["date", "dcoilwtico"])
    load_csv(
        conn,
        "holidays_events",
        DATA_DIR / "holidays_events.csv",
        ["date", "type", "locale", "locale_name", "description", "transferred"],
    )
    load_csv(conn, "transactions", DATA_DIR / "transactions.csv", ["date", "store_nbr", "transactions"])

    conn.close()
    print("数据库已生成：", DB_PATH)


if __name__ == "__main__":
    main()
