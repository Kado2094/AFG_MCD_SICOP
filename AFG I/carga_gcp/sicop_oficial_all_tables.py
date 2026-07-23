#!/usr/bin/env python3
"""
SICOP OFICIAL Bronze Loader: GCS -> BigQuery
- Loads every CSV under a GCS prefix into BigQuery tables.
- Forces ALL columns to STRING (stable raw/bronze layer).
- Reads header from GCS to define schema.
- Detects delimiter per file (supports ';' and ',').
- Optional: logs also to a file with timestamp name.
- Optional: tolerant mode for malformed CSV rows.
- Logs job_id and output_rows per table for quantitative traceability.
"""

import os
os.environ.pop("SSLKEYLOGFILE", None)

import logging
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from google.cloud import bigquery, storage
from google.api_core.exceptions import BadRequest, Forbidden, NotFound
from google.api_core.retry import Retry


# ============================================================
# CONFIG
# ============================================================

PROJECT_ID = "sicop-anomalias"
BQ_LOCATION = "US"
DATASET_ID = "sicop_oficial"

BUCKET_NAME = "sicop-data"

# New source path
PREFIXES = [
    "sicop_oficial/Datos/"
]

# Not needed for this structure, but kept for compatibility
SUFFIX_TABLE_WITH_MONTH = False

DEFAULT_DELIMITER = "|"
CSV_ENCODING = "utf-8"

# WRITE_TRUNCATE: replaces table each time
# WRITE_APPEND: appends new data to existing table
# WRITE_DISPOSITION = bigquery.WriteDisposition.WRITE_APPEND
WRITE_DISPOSITION = bigquery.WriteDisposition.WRITE_TRUNCATE

ALLOW_QUOTED_NEWLINES = True
ALLOW_JAGGED_ROWS = True

# Tolerant mode for dirty CSVs
ENABLE_TOLERANT_MODE = True
MAX_BAD_RECORDS_TOLERANT = 500

# Optional fallback if CSV quote parsing fails
ENABLE_QUOTE_FALLBACK = True

# Logging
ENABLE_FILE_LOGGING = True
LOG_DIR = "logs"
LOG_LEVEL = "INFO"

HEADER_MAX_BYTES = 256 * 1024


# ============================================================
# HELPERS
# ============================================================

VALID_BQ_NAME_REGEX = re.compile(r"[^A-Z0-9_]")


def init_logger() -> logging.Logger:
    logger = logging.getLogger("sicop_oficial_bronze_loader")
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.handlers = [ch]

    if ENABLE_FILE_LOGGING:
        import os
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(LOG_DIR, f"sicop_oficial_bq_load_{ts}.txt")

        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.info("File logging enabled: %s", log_path)

    return logger


def sanitize_bq_field_name(name: str) -> str:
    s = (name or "").strip().upper().replace(" ", "_")
    s = VALID_BQ_NAME_REGEX.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    if not s:
        s = "COL"
    if s[0].isdigit():
        s = f"F_{s}"

    return s


def sanitize_bq_table_name(filename_no_ext: str) -> str:
    s = (filename_no_ext or "").strip().upper().replace(" ", "_")
    s = VALID_BQ_NAME_REGEX.sub("_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    if not s:
        s = "TABLE"
    if s[0].isdigit():
        s = f"T_{s}"

    return s


def list_csv_blobs(storage_client: storage.Client, bucket_name: str, prefix: str) -> List[str]:
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    return sorted([b.name for b in blobs if b.name.lower().endswith(".csv")])


def read_first_nonempty_line(
    storage_client: storage.Client,
    bucket_name: str,
    blob_name: str,
    encoding: str,
    max_bytes: int
) -> str:
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    raw = blob.download_as_bytes(start=0, end=max_bytes - 1)
    text = raw.decode(encoding, errors="replace")

    for line in text.splitlines():
        if line.strip():
            return line.strip()

    raise ValueError(f"Could not read header for gs://{bucket_name}/{blob_name}")


def detect_delimiter_from_header_line(header_line: str, default_delim: str = ";") -> str:
    candidates = [default_delim, "|", ";", ","]
    best = default_delim
    best_count = 1

    for d in candidates:
        parts = header_line.split(d)
        if len(parts) > best_count:
            best = d
            best_count = len(parts)

    return best


def read_header_from_gcs(
    storage_client: storage.Client,
    bucket_name: str,
    blob_name: str,
    encoding: str,
    max_bytes: int
) -> Tuple[List[str], str]:
    header_line = read_first_nonempty_line(
        storage_client=storage_client,
        bucket_name=bucket_name,
        blob_name=blob_name,
        encoding=encoding,
        max_bytes=max_bytes
    )

    delim = detect_delimiter_from_header_line(header_line, DEFAULT_DELIMITER)

    cols = [c.strip().strip('"').strip("'") for c in header_line.split(delim)]

    if len(cols) <= 1:
        raise ValueError(
            f"Header split produced {len(cols)} column(s). "
            f"Detected delimiter='{delim}'. Header was: {header_line[:200]}"
        )

    return cols, delim


def build_string_schema(raw_cols: List[str]) -> Tuple[List[bigquery.SchemaField], List[Tuple[str, str]]]:
    used = set()
    schema: List[bigquery.SchemaField] = []
    mapping: List[Tuple[str, str]] = []

    for raw in raw_cols:
        clean = sanitize_bq_field_name(raw)
        base = clean
        i = 2

        while clean in used:
            clean = f"{base}_{i}"
            i += 1

        used.add(clean)
        mapping.append((raw, clean))
        schema.append(bigquery.SchemaField(clean, "STRING", mode="NULLABLE"))

    return schema, mapping


def ensure_dataset_exists(bq_client: bigquery.Client, logger: logging.Logger) -> None:
    dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
    dataset_ref.location = BQ_LOCATION

    try:
        bq_client.get_dataset(dataset_ref)
        logger.info("Dataset exists: %s.%s", PROJECT_ID, DATASET_ID)
    except NotFound:
        logger.warning("Dataset not found. Creating dataset: %s.%s", PROJECT_ID, DATASET_ID)
        bq_client.create_dataset(dataset_ref)
        logger.info("Dataset created: %s.%s", PROJECT_ID, DATASET_ID)


@dataclass
class LoadOutcome:
    uri: str
    table_id: str
    ok: bool
    job_id: Optional[str] = None
    output_rows: Optional[int] = None
    error: Optional[str] = None


def run_bq_load_job(
    bq_client: bigquery.Client,
    uri: str,
    table_id: str,
    schema: List[bigquery.SchemaField],
    delimiter: str,
    max_bad_records: int,
    quote_character: Optional[str]
) -> bigquery.LoadJob:
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        field_delimiter=delimiter,
        quote_character=quote_character,
        allow_quoted_newlines=ALLOW_QUOTED_NEWLINES,
        allow_jagged_rows=ALLOW_JAGGED_ROWS,
        encoding="UTF-8",
        write_disposition=WRITE_DISPOSITION,
        max_bad_records=max_bad_records,
    )

    retry = Retry(deadline=180.0)
    job = bq_client.load_table_from_uri(
        uri,
        table_id,
        location=BQ_LOCATION,
        job_config=job_config,
        retry=retry
    )
    job.result()
    return job


def load_csv_to_bq(
    bq_client: bigquery.Client,
    storage_client: storage.Client,
    logger: logging.Logger,
    blob_name: str
) -> LoadOutcome:
    filename = blob_name.split("/")[-1]
    base_table_name = sanitize_bq_table_name(filename.replace(".csv", ""))

    table_name = base_table_name
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
    uri = f"gs://{BUCKET_NAME}/{blob_name}"

    try:
        raw_cols, detected_delim = read_header_from_gcs(
            storage_client=storage_client,
            bucket_name=BUCKET_NAME,
            blob_name=blob_name,
            encoding=CSV_ENCODING,
            max_bytes=HEADER_MAX_BYTES
        )

        schema, mapping = build_string_schema(raw_cols)

        logger.info("Loading %s -> %s", uri, table_id)
        logger.info("Detected delimiter: '%s' | Columns sample mapping: %s", detected_delim, mapping[:8])

        max_bad = MAX_BAD_RECORDS_TOLERANT if ENABLE_TOLERANT_MODE else 0

        try:
            job = run_bq_load_job(
                bq_client=bq_client,
                uri=uri,
                table_id=table_id,
                schema=schema,
                delimiter=detected_delim,
                max_bad_records=max_bad,
                quote_character='"'
            )

            logger.info("OK: %s | job_id=%s | output_rows=%s", table_id, job.job_id, job.output_rows)
            return LoadOutcome(
                uri=uri,
                table_id=table_id,
                ok=True,
                job_id=job.job_id,
                output_rows=job.output_rows
            )

        except BadRequest as e1:
            if ENABLE_QUOTE_FALLBACK:
                logger.warning("Retrying with quote_character=None due to CSV quote errors: %s", table_id)

                job = run_bq_load_job(
                    bq_client=bq_client,
                    uri=uri,
                    table_id=table_id,
                    schema=schema,
                    delimiter=detected_delim,
                    max_bad_records=max_bad,
                    quote_character=None
                )

                logger.info("OK (fallback): %s | job_id=%s | output_rows=%s", table_id, job.job_id, job.output_rows)
                return LoadOutcome(
                    uri=uri,
                    table_id=table_id,
                    ok=True,
                    job_id=job.job_id,
                    output_rows=job.output_rows
                )

            raise e1

    except (BadRequest, Forbidden, NotFound, ValueError) as e:
        logger.exception("FAILED: %s -> %s", uri, table_id)
        return LoadOutcome(uri=uri, table_id=table_id, ok=False, error=str(e))

    except Exception as e:
        logger.exception("UNEXPECTED ERROR: %s -> %s", uri, table_id)
        return LoadOutcome(uri=uri, table_id=table_id, ok=False, error=f"Unexpected: {e}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logger = init_logger()

    logger.info("Project=%s | Dataset=%s | Location=%s", PROJECT_ID, DATASET_ID, BQ_LOCATION)
    logger.info("Bucket=%s | Prefixes=%s", BUCKET_NAME, PREFIXES)

    bq_client = bigquery.Client(project=PROJECT_ID, location=BQ_LOCATION)
    storage_client = storage.Client(project=PROJECT_ID)

    ensure_dataset_exists(bq_client, logger)

    outcomes: List[LoadOutcome] = []

    for prefix in PREFIXES:
        logger.info("=== Processing prefix: %s ===", prefix)

        csv_blobs = list_csv_blobs(storage_client, BUCKET_NAME, prefix)
        logger.info("Found %d CSV files in %s", len(csv_blobs), prefix)

        if not csv_blobs:
            logger.warning("No CSV files found under prefix: %s", prefix)
            continue

        for blob_name in csv_blobs:
            outcomes.append(load_csv_to_bq(bq_client, storage_client, logger, blob_name))

    ok = [o for o in outcomes if o.ok]
    bad = [o for o in outcomes if not o.ok]

    logger.info("Summary: %d succeeded, %d failed.", len(ok), len(bad))

    if ok:
        logger.info("Loaded tables (table_id | output_rows | job_id):")
        for o in ok:
            logger.info("- %s | %s | %s", o.table_id, o.output_rows, o.job_id)

    if bad:
        logger.error("Failures:")
        for o in bad:
            logger.error("- %s -> %s | %s", o.uri, o.table_id, (o.error or "")[:350])

    sys.exit(0 if not bad else 2)


if __name__ == "__main__":
    main()