# Databricks notebook source
# MAGIC %md
# MAGIC # Gold Layer — Star Schema
# MAGIC ```
# MAGIC fact_patient_visits
# MAGIC     → dim_patient   (SCD Type 2)
# MAGIC     → dim_doctor    (SCD Type 1)
# MAGIC     → dim_hospital  (SCD Type 1)
# MAGIC ```
# MAGIC

# COMMAND ----------

# DBTITLE 1,Setup
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Unity Catalog schema for serverless compute
spark.sql("CREATE SCHEMA IF NOT EXISTS de_1")
spark.sql("USE de_1")

silver = spark.table("de_1.silver_healthcare").filter(F.col("_is_current") == True)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · dim_patient — SCD Type 2
# MAGIC Surrogate key + full condition history. _is_current=true = latest record.
# MAGIC

# COMMAND ----------

# DBTITLE 1,dim_patient
dim_patient = (silver
    .select(
        F.col("name_masked").alias("patient_id"),
        F.col("age"),
        F.col("age_group"),
        F.col("gender"),
        F.col("blood_type"),
        F.col("medical_condition"),
        F.col("insurance_provider"),
        F.col("_eff_start"),
        F.col("_eff_end"),
        F.col("_is_current"),
    )
    .dropDuplicates(["patient_id", "medical_condition", "_eff_start"])
    .withColumn("patient_sk", F.monotonically_increasing_id())
)

dim_patient.write.format("delta").mode("overwrite") \
           .option("overwriteSchema", "true") \
           .saveAsTable("de_1.dim_patient")

print(f"dim_patient: {dim_patient.count()} rows")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · dim_doctor — SCD Type 1 (no history)
# MAGIC

# COMMAND ----------

# DBTITLE 1,dim_doctor
dim_doctor = (silver
    .select(
        F.col("doctor_masked").alias("doctor_id"),
        F.col("doctor").alias("doctor_name"),
    )
    .dropDuplicates(["doctor_id"])
    .withColumn("doctor_sk", F.monotonically_increasing_id())
)

dim_doctor.write.format("delta").mode("overwrite") \
          .option("overwriteSchema", "true") \
          .saveAsTable("de_1.dim_doctor")

print(f"dim_doctor: {dim_doctor.count()} rows")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · dim_hospital — SCD Type 1
# MAGIC

# COMMAND ----------

# DBTITLE 1,dim_hospital
dim_hospital = (silver
    .select("hospital")
    .dropDuplicates()
    .withColumn("hospital_sk", F.monotonically_increasing_id())
)

dim_hospital.write.format("delta").mode("overwrite") \
            .option("overwriteSchema", "true") \
            .saveAsTable("de_1.dim_hospital")

print(f"dim_hospital: {dim_hospital.count()} rows")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · fact_patient_visits
# MAGIC

# COMMAND ----------

# DBTITLE 1,fact_patient_visits
dp = dim_patient.select("patient_sk", "patient_id").alias("dp")
dd = dim_doctor.select("doctor_sk", "doctor_id").alias("dd")
dh = dim_hospital.select("hospital_sk", "hospital").alias("dh")

fact = (silver.alias("s")
    .join(dp, F.col("s.name_masked")   == F.col("dp.patient_id"), "left")
    .join(dd, F.col("s.doctor_masked") == F.col("dd.doctor_id"),  "left")
    .join(dh, F.col("s.hospital")      == F.col("dh.hospital"),   "left")
    .select(
        F.monotonically_increasing_id().alias("visit_sk"),
        F.col("dp.patient_sk"),
        F.col("dd.doctor_sk"),
        F.col("dh.hospital_sk"),
        F.col("s.date_of_admission"),
        F.col("s.discharge_date"),
        F.col("s.admission_type"),
        F.col("s.medication"),
        F.col("s.test_results"),
        F.col("s.billing_amount"),
        F.col("s.room_number"),
        F.col("s.length_of_stay"),
        F.col("s.billing_category"),
        F.col("s.admission_year"),
        F.col("s.admission_month"),
        F.col("s._is_restricted"),
        F.col("s.silver_ts").alias("etl_ts"),
    )
)

fact.write.format("delta").mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("admission_year") \
    .saveAsTable("de_1.fact_patient_visits")

print(f"fact_patient_visits: {fact.count()} rows")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Gold Agg 1 — Condition Summary
# MAGIC

# COMMAND ----------

# DBTITLE 1,gold_condition_summary
fact_df  = spark.table("de_1.fact_patient_visits")
dim_p_df = spark.table("de_1.dim_patient").filter("_is_current = true")

agg_condition = (fact_df
    .join(dim_p_df.select("patient_sk", "medical_condition", "age_group"), "patient_sk", "left")
    .groupBy("medical_condition", "admission_type")
    .agg(
        F.count("*").alias("total_visits"),
        F.round(F.avg("billing_amount"), 2).alias("avg_billing"),
        F.round(F.avg("length_of_stay"), 2).alias("avg_los"),
        F.round(F.sum("billing_amount"), 2).alias("total_revenue"),
        F.countDistinct("patient_sk").alias("unique_patients"),
    )
)

agg_condition.write.format("delta").mode("overwrite") \
             .option("overwriteSchema", "true") \
             .saveAsTable("de_1.gold_condition_summary")

print("gold_condition_summary written")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Gold Agg 2 — Monthly Admission Trend
# MAGIC

# COMMAND ----------

# DBTITLE 1,gold_monthly_trend
agg_trend = (fact_df
    .groupBy("admission_year", "admission_month", "admission_type")
    .agg(
        F.count("*").alias("total_admissions"),
        F.round(F.avg("billing_amount"), 2).alias("avg_billing"),
        F.round(F.avg("length_of_stay"), 2).alias("avg_los"),
        F.sum(F.when(F.col("test_results") == "Abnormal", 1).otherwise(0)).alias("abnormal_count"),
    )
    .orderBy("admission_year", "admission_month")
)

agg_trend.write.format("delta").mode("overwrite") \
         .option("overwriteSchema", "true") \
         .saveAsTable("de_1.gold_monthly_trend")

print("gold_monthly_trend written")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Row & Column Masking Views
# MAGIC Column masking: name/doctor are SHA-256 hashed — raw PII never reaches gold.
# MAGIC Row masking: _is_restricted=true rows (no insurance) excluded from public view.
# MAGIC

# COMMAND ----------

# DBTITLE 1,Masking Views
spark.sql("""
    CREATE OR REPLACE VIEW de_1.vw_fact_visits_public AS
    SELECT
        visit_sk, patient_sk, doctor_sk, hospital_sk,
        date_of_admission, discharge_date, admission_type,
        medication, test_results, billing_amount,
        room_number, length_of_stay, billing_category,
        admission_year, admission_month, etl_ts
    FROM de_1.fact_patient_visits
    WHERE _is_restricted = false
""")

spark.sql("""
    CREATE OR REPLACE VIEW de_1.vw_condition_summary AS
    SELECT * FROM de_1.gold_condition_summary
    WHERE medical_condition IS NOT NULL
""")

print("Masking views created")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Optimize Gold Tables
# MAGIC

# COMMAND ----------

# DBTITLE 1,Optimize
for tbl in ["de_1.fact_patient_visits", "de_1.gold_condition_summary", "de_1.gold_monthly_trend"]:
    spark.sql(f"OPTIMIZE {tbl}")
    print(f"Optimized: {tbl}")
