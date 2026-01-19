import sqlite3
import os
from datetime import datetime
import numpy as np

DB_PATH = "CV/storage.db" # set the path to the database

class EdgeDatabase:
    def __init__(self, db_path: str = DB_PATH):

        self.db_path = db_path # create the db_path instance
        # the folder where the db_path exist
        folder = os.path.dirname(db_path)
        # if the folder exist
        if folder:
            # makes folder if not existing, but if it exist then no error
            os.makedirs(folder, exist_ok=True)
        # create tables
        self._create_tables()

    def _connect(self):
        # returns the connection from sql to the db_path
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        # opens connection to SQL file
        conn = self._connect()
        # create cursor object tied to the open connection
        cur = conn.cursor()
        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                gx BLOB NOT NULL,
                gy BLOB NOT NULL
            );
        """)
        # sends table to sql and closes connection
        conn.commit()
        conn.close()

    def insert_gradients(self, gx: np.ndarray, gy: np.ndarray) -> int:
        """
        Stores gx/gy as BLOBs + metadata so we can reconstruct later.
        Returns the inserted row id.
        """
        if gx.shape != gy.shape:
            raise ValueError(f"gx and gy shapes differ: {gx.shape} vs {gy.shape}")
        if gx.dtype != gy.dtype:
            raise ValueError(f"gx and gy dtypes differ: {gx.dtype} vs {gy.dtype}")

        h, w = gx.shape
        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

        gx_blob = sqlite3.Binary(gx.tobytes())
        gy_blob = sqlite3.Binary(gy.tobytes())
        dtype_str = str(gx.dtype)  # e.g. 'int16'

        conn = self._connect()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO edges (ts, width, height, dtype, gx, gy) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, w, h, dtype_str, gx_blob, gy_blob)
        )
        row_id = cur.lastrowid
        conn.commit()
        conn.close()
        return row_id