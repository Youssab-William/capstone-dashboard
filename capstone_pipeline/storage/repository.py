import os
import json
from typing import List, Dict, Optional
from ..interfaces import StorageRepository
import logging


class JsonlStorage(StorageRepository):
    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.logger = logging.getLogger(__name__)

    def _path(self, table: str, partition: Optional[str]) -> str:
        if partition:
            return os.path.join(self.root_dir, table, partition + ".jsonl")
        return os.path.join(self.root_dir, table + ".jsonl")

    def save_rows(self, rows: List[Dict[str, any]], table: str, partition: Optional[str] = None) -> None:
        path = self._path(table, partition)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.logger.info(f"save_rows table={table} partition={partition} path={path} rows={len(rows)}")
        with open(path, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def load_rows(self, table: str, partition: Optional[str] = None) -> List[Dict[str, any]]:
        path = self._path(table, partition)
        if not os.path.exists(path):
            self.logger.info(f"load_rows_missing table={table} partition={partition}")
            return []
        self.logger.info(f"load_rows table={table} partition={partition} path={path}")
        out: List[Dict[str, any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                out.append(json.loads(line))
        return out

    def exists(self, table: str, partition: Optional[str] = None) -> bool:
        path = self._path(table, partition)
        ok = os.path.exists(path)
        self.logger.debug(f"exists table={table} partition={partition} path={path} exists={ok}")
        return ok
