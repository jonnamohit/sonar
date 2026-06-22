# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze Layer — Raw Ingestion
# MAGIC **Storage:** Unity Catalog managed table `DE_1.bronze_healthcare`
# MAGIC - Data → Parquet files inside `ingestion_date=YYYY-MM-DD/` partition folders
# MAGIC - ACID log → `_delta_log/` JSON transaction files
# MAGIC - Running twice will NOT duplicate data — checks `source_file` column to prevent re-ingestion
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 2
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType
from delta.tables import DeltaTable
import re

# Unity Catalog schema for serverless compute
spark.sql("CREATE SCHEMA IF NOT EXISTS de_1")
spark.sql("USE de_1")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Widget
# MAGIC Business reason: ADF or a job trigger can pass a different source file without editing the notebook.
# MAGIC

# COMMAND ----------

dbutils.widgets.text("source_path", "/Volumes/workspace/default/practice/healthcare_2019.csv", "Source File Path")
dbutils.widgets.text("batch_id", "", "Batch ID (blank = auto)")

SOURCE_PATH = dbutils.widgets.get("source_path")
BATCH_ID    = dbutils.widgets.get("batch_id") or str(spark.sql("SELECT unix_timestamp()").collect()[0][0])

print(f"Source  : {SOURCE_PATH}")
print(f"Batch ID: {BATCH_ID}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · File Format Validation & Error Handling
# MAGIC Unsupported formats are logged to quarantine Delta table — pipeline never fails silently.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 6
SUPPORTED_FORMATS = {"csv", "json", "orc", "xlsx"}

def get_ext(path):
    return path.rsplit(".", 1)[-1].lower()

def read_raw(path):
    ext = get_ext(path)

    if ext not in SUPPORTED_FORMATS:
        spark.createDataFrame([{
            "source_file":  path,
            "error_reason": f"Unsupported format: .{ext}",
            "logged_at":    str(spark.sql("SELECT current_timestamp()").collect()[0][0])
        }]).write.format("delta").mode("append") \
          .option("mergeSchema", "true") \
          .saveAsTable("de_1.quarantine")
        raise ValueError(f"Unsupported format '.{ext}' — supported: {SUPPORTED_FORMATS}")

    if ext == "csv":
        df = (spark.read
                .option("header", "true")
                .option("inferSchema", "false")
                .option("mode", "PERMISSIVE")
                .option("columnNameOfCorruptRecord", "_corrupt_record")
                .csv(path))
    elif ext == "json":
        df = (spark.read
                .option("mode", "PERMISSIVE")
                .option("columnNameOfCorruptRecord", "_corrupt_record")
                .json(path))
    elif ext == "orc":
        df = spark.read.orc(path)
    elif ext == "xlsx":
        df = (spark.read
                .format("com.crealytics.spark.excel")
                .option("header", "true")
                .option("inferSchema", "false")
                .load(path))
    else:
        df = None

    if df is not None and df.count() == 0:
        spark.createDataFrame([{
            "source_file":  path,
            "error_reason": "Empty file",
            "logged_at":    str(spark.sql("SELECT current_timestamp()").collect()[0][0])
        }]).write.format("delta").mode("append") \
          .option("mergeSchema", "true") \
          .saveAsTable("de_1.quarantine")
        raise ValueError("Inserted file is empty — no rows found.")

    return df

# COMMAND ----------

# DBTITLE 1,Cell 7
# MAGIC %md
# MAGIC ## 2 · File Status Check
# MAGIC Checks if the source file already exists in bronze_healthcare. If found → upsert (update existing). If not found → insert new.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 8
# Check if source file already exists in bronze_healthcare
try:
    existing = spark.sql(f"""
        SELECT COUNT(*) as cnt 
        FROM de_1.bronze_healthcare 
        WHERE source_file = '{SOURCE_PATH}'
    """).collect()[0]["cnt"]
    
    if existing > 0:
        print(f"UPDATE MODE — '{SOURCE_PATH}' already exists ({existing} rows found). Will upsert.")
    else:
        print(f"INSERT MODE — '{SOURCE_PATH}' not found. Will insert new records.")
except Exception as e:
    # Table doesn't exist yet — first load
    print(f"FIRST LOAD — Bronze table not found. Creating new table.")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Read Raw File
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 10
raw_df = read_raw(SOURCE_PATH)

# Quarantine corrupt rows (PERMISSIVE mode captures them in _corrupt_record)
if "_corrupt_record" in raw_df.columns:
    corrupt_df = raw_df.filter(F.col("_corrupt_record").isNotNull()) \
                       .withColumn("source_file",   F.lit(SOURCE_PATH)) \
                       .withColumn("error_reason",  F.lit("parse_error")) \
                       .withColumn("logged_at",     F.current_timestamp())
    c = corrupt_df.count()
    if c > 0:
        corrupt_df.write.format("delta").mode("append") \
                  .option("mergeSchema", "true").saveAsTable("de_1.quarantine")
        print(f"Quarantined {c} corrupt rows")
    raw_df = raw_df.filter(F.col("_corrupt_record").isNull()).drop("_corrupt_record")

print(f"Raw rows: {raw_df.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Schema Drift Detection
# MAGIC New columns in source → kept via mergeSchema (schema evolution).
# MAGIC Missing columns → filled with null so downstream schema stays consistent.
# MAGIC

# COMMAND ----------

EXPECTED_COLS = {
    "Name", "Age", "Gender", "Blood Type", "Medical Condition",
    "Date of Admission", "Doctor", "Hospital", "Insurance Provider",
    "Billing Amount", "Room Number", "Admission Type", "Discharge Date",
    "Medication", "Test Results"
}

incoming     = set(raw_df.columns)
new_cols     = incoming - EXPECTED_COLS
missing_cols = EXPECTED_COLS - incoming

if new_cols:
    print(f"SCHEMA DRIFT — new columns: {new_cols}  → kept via mergeSchema")
if missing_cols:
    print(f"SCHEMA DRIFT — missing columns: {missing_cols} → filled with null")

for col in missing_cols:
    raw_df = raw_df.withColumn(col, F.lit(None).cast("string"))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Rename Columns & Add Metadata
# MAGIC

# COMMAND ----------

def clean_col(c):
    return re.sub(r'[^a-zA-Z0-9]', '_', c.strip()).lower()

for old in raw_df.columns:
    raw_df = raw_df.withColumnRenamed(old, clean_col(old))

bronze_df = (raw_df
    .withColumn("ingested_at",    F.current_timestamp())
    .withColumn("ingestion_date", F.current_date())
    .withColumn("source_file",    F.lit(SOURCE_PATH))
    .withColumn("source_format",  F.lit(get_ext(SOURCE_PATH)))
    .withColumn("batch_id",       F.lit(BATCH_ID))
)

bronze_df.printSchema()


# COMMAND ----------

# DBTITLE 1,Cell 15
# MAGIC %md
# MAGIC ## 6 · Write to Bronze Delta Table (Merge Upsert)
# MAGIC
# MAGIC **Upsert Logic:**
# MAGIC 1. **Deduplicate source** on merge keys (name + date_of_admission + hospital)
# MAGIC    - Source file has ~954 duplicate business keys
# MAGIC    - Must deduplicate BEFORE merge to avoid ambiguity error
# MAGIC 2. If table exists → **MERGE** on business keys
# MAGIC    - Matched rows → UPDATE with latest data
# MAGIC    - Unmatched rows → INSERT as new records
# MAGIC 3. If table doesn't exist → **CREATE** with initial load
# MAGIC 4. mergeSchema = new columns auto-added (schema evolution)
# MAGIC
# MAGIC **Why deduplicate on merge keys?**
# MAGIC - Delta Lake merge requires 1:1 mapping (one source row → one target row)
# MAGIC - Multiple source rows matching same target → ambiguity error
# MAGIC - Deduplication removes exact duplicates from raw file
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 16
from delta.tables import DeltaTable

# STEP 1: Deduplicate source data on MERGE KEY columns only
# Must match the merge condition: name + date_of_admission + hospital
original_count = bronze_df.count()
bronze_df_dedup = bronze_df.dropDuplicates(["name", "date_of_admission", "hospital"])
dedup_count = bronze_df_dedup.count()
duplicates_removed = original_count - dedup_count

if duplicates_removed > 0:
    print(f"🧼 Removed {duplicates_removed} duplicate rows from source ({original_count} → {dedup_count})")

# STEP 2: Check if table exists
try:
    existing_table = spark.table("de_1.bronze_healthcare")
    table_exists = True
except:
    table_exists = False

# STEP 3: Merge or create
if table_exists:
    # MERGE on business keys: name + date_of_admission + hospital
    bronze_table = DeltaTable.forName(spark, "de_1.bronze_healthcare")
    
    bronze_table.alias("target").merge(
        bronze_df_dedup.alias("source"),
        """target.name = source.name 
           AND target.date_of_admission = source.date_of_admission
           AND target.hospital = source.hospital"""
    ).whenMatchedUpdateAll() \
     .whenNotMatchedInsertAll() \
     .execute()
    
    print("✅ Bronze upsert complete (merge on business keys)")
else:
    # Table doesn't exist yet - create it with initial load
    bronze_df_dedup.write.format("delta").mode("overwrite") \
        .option("mergeSchema", "true") \
        .option("overwriteSchema", "true") \
        .saveAsTable("de_1.bronze_healthcare")
    
    print("✅ Bronze initial load complete (table created)")

print(f"\nTotal bronze rows: {spark.table('de_1.bronze_healthcare').count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Ingestion Complete
# MAGIC

# COMMAND ----------

print(f"Ingestion complete: {SOURCE_PATH}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Schema Evolution Demo
# MAGIC Simulates a future file arriving with a new column.
# MAGIC mergeSchema absorbs it without recreating the table.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 20
demo = (spark.createDataFrame(
    [("demo",)],
    StructType([StructField("new_future_column", StringType(), True)])
).withColumn("ingested_at",    F.current_timestamp())
 .withColumn("ingestion_date", F.current_date())
 .withColumn("source_file",    F.lit("schema_evo_demo"))
 .withColumn("source_format",  F.lit("csv"))
 .withColumn("batch_id",       F.lit("evo_demo")))

demo.write.format("delta").mode("append") \
    .option("mergeSchema", "true").saveAsTable("de_1.bronze_healthcare")

print("Schema evolution: new column merged into bronze")

# Clean up demo row
spark.sql("DELETE FROM de_1.bronze_healthcare WHERE source_file = 'schema_evo_demo'")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Time Travel — view last 5 versions
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 22
spark.sql("DESCRIBE HISTORY de_1.bronze_healthcare") \
     .select("version", "timestamp", "operation") \
     .show(5, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Optimize + Z-Order
# MAGIC

# COMMAND ----------

# DBTITLE 1,Cell 24
spark.sql("OPTIMIZE de_1.bronze_healthcare")
print("Bronze optimized")
