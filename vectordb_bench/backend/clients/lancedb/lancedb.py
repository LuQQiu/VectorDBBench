import logging
import uuid
from contextlib import contextmanager

import lancedb
import pyarrow as pa
from lancedb.pydantic import LanceModel

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import IndexType, VectorDB
from .config import LanceDBConfig, LanceDBIndexConfig

log = logging.getLogger(__name__)


class VectorModel(LanceModel):
    id: int
    vector: list[float]


class LanceDB(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: LanceDBConfig,
        db_case_config: LanceDBIndexConfig,
        collection_name: str | None = None,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "LanceDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.with_scalar_labels = with_scalar_labels
        # Use provided table name, or generate default with random suffix
        if collection_name:
            self.table_name = collection_name
        else:
            suffix = uuid.uuid4().hex[:8]
            self.table_name = f"lancedb_bench_test_{suffix}"
        self.dim = dim
        self.uri = db_config["uri"]
        self.api_key = db_config.get("api_key")
        self.host_override = db_config.get("host_override")
        # avoid the search_param being called every time during the search process
        self.search_config = db_case_config.search_param()

        # Field names
        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"

        # Filter expression (set by prepare_filter)
        self.filter_expr = None

        log.info(f"Table name: {self.table_name}")
        log.info(f"Search config: {self.search_config}")
        log.info(f"With scalar labels: {self.with_scalar_labels}")

        connect_args = {"uri": self.uri}
        if self.api_key:
            connect_args["api_key"] = self.api_key
        if self.host_override:
            connect_args["host_override"] = self.host_override

        db = lancedb.connect(**connect_args)

        self._existing_row_count = 0
        try:
            table = db.open_table(self.table_name)
            self._existing_row_count = table.count_rows()
            log.info(f"Existing table found with {self._existing_row_count} rows")
        except Exception:
            fields = [
                pa.field(self._scalar_id_field, pa.int64()),
                pa.field(self._vector_field, pa.list_(pa.float32(), list_size=self.dim)),
            ]
            if self.with_scalar_labels:
                fields.append(pa.field(self._scalar_label_field, pa.string()))
            schema = pa.schema(fields)
            db.create_table(self.table_name, schema=schema, mode="overwrite")
            log.info(f"Created new table: {self.table_name}")

    def get_existing_row_count(self) -> int:
        """Return the number of rows already in the table (for resume support)"""
        return self._existing_row_count

    @contextmanager
    def init(self):
        connect_args = {"uri": self.uri}
        if self.api_key:
            connect_args["api_key"] = self.api_key
        if self.host_override:
            connect_args["host_override"] = self.host_override

        self.db = lancedb.connect(**connect_args)
        self.table = self.db.open_table(self.table_name)
        yield
        self.db = None
        self.table = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            if self.with_scalar_labels and labels_data:
                data = [
                    {
                        self._scalar_id_field: meta,
                        self._vector_field: emb,
                        self._scalar_label_field: label,
                    }
                    for meta, emb, label in zip(metadata, embeddings, labels_data, strict=False)
                ]
            else:
                data = [
                    {self._scalar_id_field: meta, self._vector_field: emb}
                    for meta, emb in zip(metadata, embeddings, strict=False)
                ]
            self.table.add(data)
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into LanceDB table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        """Prepare filter expression before search."""
        if filters.type == FilterOp.NonFilter:
            self.filter_expr = None
        elif filters.type == FilterOp.NumGE:
            self.filter_expr = f"{self._scalar_id_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.filter_expr = f"{self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Not supported Filter for LanceDB - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        results = self.table.search(query).select([self._scalar_id_field]).limit(k)

        # Apply filter if set
        if self.filter_expr:
            results = results.where(self.filter_expr, prefilter=True)

        # Apply search parameters
        if "nprobes" in self.search_config:
            results = results.nprobes(self.search_config["nprobes"])
        if "ef" in self.search_config:
            results = results.ef(self.search_config["ef"])
        if "refine_factor" in self.search_config:
            results = results.refine_factor(self.search_config["refine_factor"])

        results = results.to_list()
        return [int(result[self._scalar_id_field]) for result in results]

    def optimize(self, data_size: int | None = None):
        if self.table:
            # Create BTREE index on id column for filter performance
            log.info(f"Creating BTREE index on id column for table ({self.table_name})")
            self.table.create_scalar_index(self._scalar_id_field)  # BTree is default

            # Create BITMAP index on label column for string equality filter
            if self.with_scalar_labels:
                log.info(f"Creating BITMAP index on label column for table ({self.table_name})")
                self.table.create_scalar_index(self._scalar_label_field, index_type="BITMAP")

        if self.table and hasattr(self, "case_config") and self.case_config.index != IndexType.NONE:
            log.info(f"Creating index for LanceDB table ({self.table_name})")
            log.info(f"Index parameters: {self.case_config.index_param()}")
            self.table.create_index(**self.case_config.index_param())
            # Better recall with IVF_PQ (though still bad) but breaks HNSW: https://github.com/lancedb/lancedb/issues/2369
            # Note: Do NOT call optimize() for AUTOINDEX since it now uses IVF_HNSW_SQ
            if self.case_config.index == IndexType.IVFPQ:
                self.table.optimize()
