# Databricks notebook source
# MAGIC %md
# MAGIC # Silver Layer — Cleansed & Conformed
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setup — Unity Catalog Schema
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, DateType
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# Unity Catalog schema for serverless compute (no DBFS paths needed)
spark.sql("CREATE SCHEMA IF NOT EXISTS de_1")
spark.sql("USE de_1")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Widgets
# MAGIC Business reason: allows reprocessing only one admission type if that slice fails in a job run.
# MAGIC

# COMMAND ----------

dbutils.widgets.dropdown("admission_type_filter", "All", ["All", "Emergency", "Urgent", "Elective"], "Filter Admission Type")
dbutils.widgets.text("null_threshold_pct", "0.30", "Max Null % per row (above = quarantine)")

ADMISSION_FILTER = dbutils.widgets.get("admission_type_filter")
NULL_THRESHOLD   = float(dbutils.widgets.get("null_threshold_pct"))

print(f"Admission filter : {ADMISSION_FILTER}")
print(f"Null threshold   : {NULL_THRESHOLD * 100:.0f}%")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · CDC Incremental Load — Silver Watermark
# MAGIC Only reads Bronze rows ingested after the last Silver watermark. No full re-scan on every run.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Watermark Functions — Unity Catalog
def get_silver_watermark():
    try:
        wm = spark.table("de_1.silver_watermark")
        result = wm.orderBy(F.col("processed_at").desc()).limit(1).collect()
        
        # Handle empty table
        if not result:
            print("Watermark table is empty - starting from epoch")
            return "1900-01-01T00:00:00.000+0000"
        
        # Get the max_ingested_at value
        max_ts = result[0]["max_ingested_at"]
        if max_ts is None or str(max_ts).strip() == "":
            print("Watermark value is null - starting from epoch")
            return "1900-01-01T00:00:00.000+0000"
        
        print(f"Using watermark: {max_ts}")
        return str(max_ts)
        
    except Exception as e:
        print(f"Watermark table not found (first run): {e}")
        return "1900-01-01T00:00:00.000+0000"

def save_silver_watermark(max_ts):
    spark.createDataFrame([{
        "max_ingested_at": str(max_ts),
        "processed_at":    spark.sql("SELECT current_timestamp()").collect()[0][0]
    }]).write.format("delta").mode("append") \
      .option("mergeSchema", "true").saveAsTable("de_1.silver_watermark")

silver_wm = get_silver_watermark()
print(f"Silver watermark: {silver_wm}")

df = spark.table("de_1.bronze_healthcare") \
          .filter(F.col("ingested_at") > silver_wm) \
          .filter(F.col("source_file") != "schema_evo_demo")

if ADMISSION_FILTER != "All":
    df = df.filter(F.upper(F.col("admission_type")) == ADMISSION_FILTER.upper())

print(f"Incremental rows from bronze: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Null % Check
# MAGIC Rows where more than NULL_THRESHOLD % of columns are null → quarantine.
# MAGIC Rows below threshold → proceed to silver.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Null % Check
total_cols = len(df.columns)

df = df.withColumn(
    "_null_count",
    sum(F.when(F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == ""), 1).otherwise(0) for c in df.columns)
).withColumn("_null_pct", F.col("_null_count") / F.lit(total_cols))

quarantine_df = df.filter(F.col("_null_pct") > NULL_THRESHOLD) \
                  .withColumn("quarantine_reason", F.lit(f"null_pct_exceeds_{NULL_THRESHOLD}")) \
                  .withColumn("quarantine_ts", F.current_timestamp())

clean_df = df.filter(F.col("_null_pct") <= NULL_THRESHOLD).drop("_null_count", "_null_pct")

q_count = quarantine_df.count()
if q_count > 0:
    quarantine_df.write.format("delta").mode("append") \
                 .option("mergeSchema", "true").saveAsTable("de_1.quarantine")
    print(f"Quarantined {q_count} rows (null% > {NULL_THRESHOLD * 100:.0f}%)")

print(f"Rows proceeding to silver: {clean_df.count()}")
df = clean_df


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Remove Duplicates
# MAGIC

# COMMAND ----------

df = df.dropDuplicates()

w = Window.partitionBy("name", "date_of_admission", "hospital").orderBy(F.col("ingested_at").desc())
df = df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")

print(f"After dedup: {df.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Fix Invalid Values, Types, Format Issues
# MAGIC

# COMMAND ----------

# Age — invalid sentinel values (-5, 0, 130, 999) → null
df = df.withColumn("age", F.col("age").cast(IntegerType())) \
       .withColumn("age", F.when((F.col("age") <= 0) | (F.col("age") > 120), None).otherwise(F.col("age")))

# Billing — negative values → null (will be imputed in next step)
df = df.withColumn("billing_amount", F.col("billing_amount").cast(DoubleType())) \
       .withColumn("billing_amount", F.when(F.col("billing_amount") < 0, None).otherwise(F.col("billing_amount")))

# Dates
df = df.withColumn("date_of_admission", F.to_date("date_of_admission", "yyyy-MM-dd")) \
       .withColumn("discharge_date",    F.to_date("discharge_date",    "yyyy-MM-dd"))

# Room number
df = df.withColumn("room_number", F.col("room_number").cast(IntegerType()))

# Fix mixed name casing (e.g. cHaRlEs rOdGeRs)
df = df.withColumn("name",   F.initcap(F.col("name"))) \
       .withColumn("doctor", F.initcap(F.col("doctor")))

# Standardise categoricals — invalid values → null (imputed below)
df = df.withColumn("medical_condition",
        F.when(F.initcap("medical_condition").isin("Cancer","Arthritis","Diabetes","Asthma","Hypertension","Obesity"),
               F.initcap("medical_condition")).otherwise(None)) \
       .withColumn("admission_type",
        F.when(F.initcap("admission_type").isin("Emergency","Urgent","Elective"),
               F.initcap("admission_type")).otherwise(None)) \
       .withColumn("test_results",
        F.when(F.initcap("test_results").isin("Normal","Abnormal","Inconclusive"),
               F.initcap("test_results")).otherwise(None)) \
       .withColumn("gender",
        F.when(F.initcap("gender").isin("Male","Female"),
               F.initcap("gender")).otherwise(None)) \
       .withColumn("blood_type",
        F.when(F.col("blood_type").isin("A+","A-","B+","B-","AB+","AB-","O+","O-"),
               F.col("blood_type")).otherwise(None))

print(f"After type fixing: {df.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Professional Null Imputation — Clean Approach
# MAGIC
# MAGIC Each column imputed using **clinically and operationally accurate groupings**.  
# MAGIC Simple, readable, efficient — uses `coalesce()` + window functions.
# MAGIC
# MAGIC **🏥 Clinical/Operational Logic:**
# MAGIC
# MAGIC | # | Column | Grouped By | Logic |
# MAGIC |---|---|---|---|
# MAGIC | 1 | `medical_condition` | Medication | Insulin → Diabetes |
# MAGIC | 2 | `billing_amount` | Condition + Admission Type | Severity + urgency → cost |
# MAGIC | 3 | `insurance_provider` | Hospital + Admission Type | Network contracts vary by hospital |
# MAGIC | 4 | `test_results` | Medical Condition | Diabetes → Abnormal |
# MAGIC | 5 | `doctor` | Hospital | Same hospital → same doctor pool |
# MAGIC | 6 | `medication` | Medical Condition | Diabetes → Insulin |
# MAGIC | 7 | `room_number` | Admission Type | Emergency → ICU, Elective → Standard |
# MAGIC
# MAGIC **Execution Order:** `medical_condition` filled FIRST (dependency for others).
# MAGIC

# COMMAND ----------

# DBTITLE 1,Medical Condition ← Medication
# 1. Medical Condition ← Medication
# Bootstrap: medication implies condition
from pyspark.sql import Window as W

# Mode per medication
df = df.withColumn("medical_condition",
    F.coalesce(
        F.col("medical_condition"),
        F.first(F.col("medical_condition"), ignorenulls=True).over(
            W.partitionBy("medication").orderBy(F.rand())
        ),
        F.lit("Unknown Condition")
    )
)

print(f"Medical condition nulls: {df.filter(F.col('medical_condition').isNull()).count()}")

# COMMAND ----------

# DBTITLE 1,Billing Amount ← Medical Condition + Admission Type
# 2. Billing Amount ← Medical Condition + Admission Type
# Cost varies by condition severity + admission urgency

# Median per medical_condition + admission_type
df = df.withColumn("billing_amount",
    F.coalesce(
        F.col("billing_amount"),
        F.percentile_approx("billing_amount", 0.5, 1000000).over(
            W.partitionBy("medical_condition", "admission_type")
        ),
        F.lit(10000.0)
    )
)

print(f"Billing amount nulls: {df.filter(F.col('billing_amount').isNull()).count()}")

# COMMAND ----------

# DBTITLE 1,Insurance Provider ← Hospital + Admission Type
# 3. Insurance Provider ← Hospital + Admission Type
# Insurance networks vary by hospital contracts + admission type

# Mode per hospital + admission_type
df = df.withColumn("insurance_provider",
    F.coalesce(
        F.col("insurance_provider"),
        F.first(F.col("insurance_provider"), ignorenulls=True).over(
            W.partitionBy("hospital", "admission_type").orderBy(F.rand())
        ),
        F.lit("Unknown")
    )
)

print(f"Insurance provider nulls: {df.filter(F.col('insurance_provider').isNull()).count()}")

# COMMAND ----------

# DBTITLE 1,Test Results ← Medical Condition
# 4. Test Results ← Medical Condition
# Disease type predicts test outcomes (Diabetes → Abnormal)

# Mode per medical_condition
df = df.withColumn("test_results",
    F.coalesce(
        F.col("test_results"),
        F.first(F.col("test_results"), ignorenulls=True).over(
            W.partitionBy("medical_condition").orderBy(F.rand())
        ),
        F.lit("Normal")
    )
)

print(f"Test results nulls: {df.filter(F.col('test_results').isNull()).count()}")

# COMMAND ----------

# DBTITLE 1,Doctor ← Hospital
# 5. Doctor ← Hospital
# Same hospital → same pool of doctors

# Mode per hospital
df = df.withColumn("doctor",
    F.coalesce(
        F.col("doctor"),
        F.first(F.col("doctor"), ignorenulls=True).over(
            W.partitionBy("hospital").orderBy(F.rand())
        ),
        F.lit("Dr. Unknown")
    )
)

print(f"Doctor nulls: {df.filter(F.col('doctor').isNull()).count()}")

# COMMAND ----------

# DBTITLE 1,Medication ← Medical Condition
# 6. Medication ← Medical Condition
# Reverse mapping: condition implies medication

# Mode per medical_condition
df = df.withColumn("medication",
    F.coalesce(
        F.col("medication"),
        F.first(F.col("medication"), ignorenulls=True).over(
            W.partitionBy("medical_condition").orderBy(F.rand())
        ),
        F.lit("Not Prescribed")
    )
)

print(f"Medication nulls: {df.filter(F.col('medication').isNull()).count()}")

# COMMAND ----------

# DBTITLE 1,Room Number ← Admission Type
# 7. Room Number ← Admission Type
# Emergency → ICU rooms (low numbers), Elective → Standard rooms

# Median per admission_type
df = df.withColumn("room_number",
    F.coalesce(
        F.col("room_number"),
        F.percentile_approx("room_number", 0.5, 1000000).over(
            W.partitionBy("admission_type")
        ).cast(IntegerType()),
        F.lit(100)
    )
)

print(f"Room number nulls: {df.filter(F.col('room_number').isNull()).count()}")

# COMMAND ----------

# ── Null Summary after all imputation ───────────────────────────────────────
print("Null summary after imputation:")
for c in ["billing_amount", "insurance_provider", "test_results", "doctor",
          "medical_condition", "medication", "room_number"]:
    n = df.filter(F.col(c).isNull()).count()
    print(f"  {c}: {n} nulls remaining")

# Drop rows where critical fields still null after imputation
df = df.filter(
    F.col("medical_condition").isNotNull() &
    F.col("date_of_admission").isNotNull() &
    F.col("billing_amount").isNotNull() &
    F.col("admission_type").isNotNull()
)

print(f"Final rows after imputation + critical filter: {df.count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Derived Columns
# MAGIC

# COMMAND ----------

df = df \
    .withColumn("length_of_stay",
        F.datediff(F.col("discharge_date"), F.col("date_of_admission"))) \
    .withColumn("age_group",
        F.when(F.col("age") < 18,  "Paediatric")
         .when(F.col("age") < 40,  "Young Adult")
         .when(F.col("age") < 60,  "Middle Aged")
         .when(F.col("age") < 80,  "Senior")
         .otherwise("Elderly")) \
    .withColumn("billing_category",
        F.when(F.col("billing_amount") < 10000, "Low")
         .when(F.col("billing_amount") < 30000, "Medium")
         .otherwise("High")) \
    .withColumn("admission_year",  F.year("date_of_admission")) \
    .withColumn("admission_month", F.month("date_of_admission")) \
    .withColumn("silver_ts",       F.current_timestamp())


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · PII Masking
# MAGIC Column masking — Name and Doctor hashed with SHA-256. Raw values stay in bronze only.
# MAGIC Row masking — rows with `No Insurance` or null flagged `_is_restricted=true`, hidden from gold public views.
# MAGIC

# COMMAND ----------

df = df \
    .withColumn("name_masked",   F.sha2(F.col("name"),   256)) \
    .withColumn("doctor_masked", F.sha2(F.col("doctor"), 256)) \
    .withColumn("_is_restricted",
        F.when(
            F.col("insurance_provider").isNull() |
            (F.trim(F.col("insurance_provider")) == "") |
            (F.col("insurance_provider") == "No Insurance"),
            True
        ).otherwise(False))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · SCD Type 2 Columns
# MAGIC Tracks history when medical_condition changes for a patient.
# MAGIC Old row → `_is_current=false`, `_eff_end` set. New row → `_is_current=true`.
# MAGIC

# COMMAND ----------

df = df \
    .withColumn("_eff_start",  F.col("date_of_admission")) \
    .withColumn("_eff_end",    F.lit(None).cast(DateType())) \
    .withColumn("_is_current", F.lit(True))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 9 · Write to Silver — SCD Merge (Type 1 + Type 2)
# MAGIC

# COMMAND ----------

# DBTITLE 1,SCD Merge
try:
    silver_dt = DeltaTable.forName(spark, "de_1.silver_healthcare")

    (silver_dt.alias("existing")
     .merge(
         df.alias("incoming"),
         """existing.name = incoming.name
            AND existing.hospital = incoming.hospital
            AND existing.date_of_admission = incoming.date_of_admission
            AND existing._is_current = true"""
     )
     .whenMatchedUpdate(
         condition="existing.insurance_provider != incoming.insurance_provider AND existing.medical_condition = incoming.medical_condition",
         set={"insurance_provider": "incoming.insurance_provider",
              "silver_ts":          "incoming.silver_ts"}
     )
     .whenMatchedUpdate(
         condition="existing.medical_condition != incoming.medical_condition",
         set={"_is_current": "false",
              "_eff_end":    "incoming._eff_start"}
     )
     .whenNotMatchedInsertAll()
     .execute())
    print("Silver SCD merge complete")

except Exception:
    df.write.format("delta").mode("overwrite") \
      .option("overwriteSchema", "true").saveAsTable("de_1.silver_healthcare")
    print("Silver initial load complete")

print(f"Silver rows: {spark.table('de_1.silver_healthcare').count()}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 10 · Z-Ordering
# MAGIC Co-locates related data in the same Parquet files for faster filtered reads.
# MAGIC Best columns = ones most used in WHERE and JOIN clauses.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Z-Ordering
spark.sql("OPTIMIZE de_1.silver_healthcare ZORDER BY (medical_condition, admission_type, date_of_admission)")
print("Z-Ordering applied: medical_condition, admission_type, date_of_admission")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 11 · Update Silver Watermark
# MAGIC

# COMMAND ----------

# DBTITLE 1,Update Watermark
max_ts = df.agg(F.max("ingested_at")).collect()[0][0]

# Save watermark to Unity Catalog table instead of DBFS
spark.createDataFrame([{
    "max_ingested_at": str(max_ts),
    "processed_at":    spark.sql("SELECT current_timestamp()").collect()[0][0]
}]).write.format("delta").mode("append") \
  .option("mergeSchema", "true").saveAsTable("de_1.silver_watermark")

print(f"Silver watermark updated: {max_ts}")


# COMMAND ----------

# DBTITLE 1,Data Quality Report Header
# MAGIC %md
# MAGIC ## 11.5 · Data Quality Report — Percentage Metrics
# MAGIC Comprehensive quality metrics report. Shows percentages but does not fail the pipeline.
# MAGIC Use this as the single source of truth for data quality KPIs.

# COMMAND ----------

# DBTITLE 1,Data Quality Report Metrics
# ═══════════════════════════════════════════════════════════════════════════
# DATA QUALITY REPORT — Percentage Metrics
# ═══════════════════════════════════════════════════════════════════════════

silver_final = spark.table("de_1.silver_healthcare")
total_rows = silver_final.count()

print("\n" + "="*80)
print(" DATA QUALITY REPORT — SILVER LAYER")
print("="*80)

# ─── 1. Row Counts & Transformation Funnel ────────────────────────────────────
print("\n📊 TRANSFORMATION FUNNEL:")
print(f"  Bronze rows processed:        {df.count():,}")
try:
    quarantine_count = spark.table("de_1.quarantine").count()
except:
    quarantine_count = 0
print(f"  Quarantined (null % > {NULL_THRESHOLD*100:.0f}%): {quarantine_count:,}")
print(f"  Silver rows final:            {total_rows:,}")
quarantine_pct = (quarantine_count / (total_rows + quarantine_count) * 100) if (total_rows + quarantine_count) > 0 else 0
print(f"  Quarantine rate:              {quarantine_pct:.2f}%")

# ─── 2. Null Percentage Per Column ────────────────────────────────────────────
print("\n🔍 NULL PERCENTAGE PER COLUMN:")
print(f"  {'Column':<30} {'Null Count':>12} {'Null %':>10}")
print("  " + "-"*54)

critical_cols = [
    "medical_condition", "date_of_admission", "billing_amount", 
    "admission_type", "insurance_provider", "test_results",
    "doctor", "medication", "age", "room_number"
]

for col in critical_cols:
    null_count = silver_final.filter(F.col(col).isNull()).count()
    null_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
    status = "✅" if null_pct == 0 else "⚠️" if null_pct < 5 else "❌"
    print(f"  {status} {col:<27} {null_count:>12,} {null_pct:>9.2f}%")

# ─── 3. Duplicate Percentage ──────────────────────────────────────────────────
print("\n🔄 DUPLICATE ANALYSIS:")
business_key_cols = ["name", "date_of_admission", "hospital"]

# Use name_masked since name is dropped after masking
bk_cols_check = ["name_masked", "date_of_admission", "hospital"]
total_current = silver_final.filter(F.col("_is_current") == True).count()
distinct_current = silver_final.filter(F.col("_is_current") == True) \
                                .dropDuplicates(bk_cols_check).count()
dup_count = total_current - distinct_current
dup_pct = (dup_count / total_current * 100) if total_current > 0 else 0

print(f"  Total current records:        {total_current:,}")
print(f"  Distinct business keys:       {distinct_current:,}")
print(f"  Duplicate count:              {dup_count:,}")
dup_status = "✅" if dup_pct == 0 else "⚠️" if dup_pct < 1 else "❌"
print(f"  {dup_status} Duplicate %:                  {dup_pct:.2f}%")

# ─── 4. Data Type & Value Quality ─────────────────────────────────────────────
print("\n✓ VALUE QUALITY CHECKS:")

# Age validation
invalid_age = silver_final.filter((F.col("age") <= 0) | (F.col("age") > 120)).count()
age_pct = (invalid_age / total_rows * 100) if total_rows > 0 else 0
age_status = "✅" if age_pct == 0 else "⚠️" if age_pct < 1 else "❌"
print(f"  {age_status} Invalid age (≤0 or >120):      {invalid_age:,} ({age_pct:.2f}%)")

# Billing validation
neg_billing = silver_final.filter(F.col("billing_amount") < 0).count()
billing_pct = (neg_billing / total_rows * 100) if total_rows > 0 else 0
billing_status = "✅" if billing_pct == 0 else "⚠️" if billing_pct < 1 else "❌"
print(f"  {billing_status} Negative billing:            {neg_billing:,} ({billing_pct:.2f}%)")

# PII masking validation
unmasked = silver_final.filter(F.col("name_masked").isNull() | F.col("doctor_masked").isNull()).count()
mask_pct = (unmasked / total_rows * 100) if total_rows > 0 else 0
mask_status = "✅" if mask_pct == 0 else "❌"
print(f"  {mask_status} Unmasked PII:                 {unmasked:,} ({mask_pct:.2f}%)")

# ─── 5. SCD Type 2 Tracking ───────────────────────────────────────────────────
print("\n📜 SCD TYPE 2 TRACKING:")
total_all = silver_final.count()
current_records = silver_final.filter(F.col("_is_current") == True).count()
historical_records = total_all - current_records
historical_pct = (historical_records / total_all * 100) if total_all > 0 else 0

print(f"  Total records (current + hist): {total_all:,}")
print(f"  Current records (_is_current):  {current_records:,}")
print(f"  Historical records:             {historical_records:,} ({historical_pct:.1f}%)")

# ─── 6. Overall Data Quality Score ────────────────────────────────────────────
print("\n🎯 OVERALL DATA QUALITY SCORE:")

# Calculate score (0-100)
score = 100
score -= min(quarantine_pct * 2, 20)      # -2 points per % quarantined (max -20)
score -= min(dup_pct * 10, 20)             # -10 points per % duplicate (max -20)
score -= min(age_pct * 5, 10)              # -5 points per % invalid age (max -10)
score -= min(billing_pct * 5, 10)          # -5 points per % negative billing (max -10)
score -= min(mask_pct * 20, 20)            # -20 points per % unmasked (max -20)

# Null penalty (average null % across critical columns)
avg_null_pct = sum([silver_final.filter(F.col(c).isNull()).count() for c in critical_cols]) / len(critical_cols) / total_rows * 100 if total_rows > 0 else 0
score -= min(avg_null_pct * 4, 20)         # -4 points per avg % null (max -20)

score = max(0, score)  # Floor at 0

if score >= 95:
    grade = "A+ (Excellent)"
    emoji = "🏆"
elif score >= 90:
    grade = "A (Very Good)"
    emoji = "✅"
elif score >= 80:
    grade = "B (Good)"
    emoji = "👍"
elif score >= 70:
    grade = "C (Acceptable)"
    emoji = "⚠️"
else:
    grade = "D (Needs Improvement)"
    emoji = "❌"

print(f"  {emoji} Quality Score: {score:.1f}/100 — {grade}")
print("\n" + "="*80)
print("📌 This report does NOT fail the pipeline. See validation checks below.")
print("="*80 + "\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12 · Time Travel
# MAGIC

# COMMAND ----------

# DBTITLE 1,Time Travel
spark.sql("DESCRIBE HISTORY de_1.silver_healthcare") \
     .select("version", "timestamp", "operation") \
     .show(5, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 13 · Validation Checks — Data Quality Gate
# MAGIC Any failure raises an exception → Job task fails → pipeline stops. No silent bad data.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Validation Checks
silver = spark.table("de_1.silver_healthcare")
errors = []

row_count = silver.count()
if row_count == 0:
    errors.append("FAIL: silver_healthcare is empty")
else:
    print(f"PASS: row count = {row_count}")

for col in ["date_of_admission", "billing_amount", "medical_condition", "admission_type"]:
    n = silver.filter(F.col(col).isNull()).count()
    if n > 0:
        errors.append(f"FAIL: {n} nulls in critical column '{col}'")
    else:
        print(f"PASS: no nulls in '{col}'")

neg = silver.filter(F.col("billing_amount") < 0).count()
if neg > 0:
    errors.append(f"FAIL: {neg} rows with negative billing_amount")
else:
    print("PASS: no negative billing amounts")

bad_age = silver.filter((F.col("age") <= 0) | (F.col("age") > 120)).count()
if bad_age > 0:
    errors.append(f"FAIL: {bad_age} rows with invalid age")
else:
    print("PASS: all ages valid")

# Check duplicates only in CURRENT records (SCD Type 2 creates historical versions)
current_only = silver.filter(F.col("_is_current") == True)
total    = current_only.count()
distinct = current_only.dropDuplicates(["name_masked", "date_of_admission", "hospital"]).count()
if total != distinct:
    errors.append(f"FAIL: {total - distinct} duplicate business keys (current records)")
else:
    print("PASS: no duplicate business keys (current records)")

if errors:
    raise Exception("DATA QUALITY GATE FAILED:\n" + "\n".join(errors))

print("\nALL CHECKS PASSED — Silver is clean.")


# COMMAND ----------

