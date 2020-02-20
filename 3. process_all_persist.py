# Databricks notebook source
# Import All Libraries.
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql.types import StructField, StringType, IntegerType, DecimalType, StructType, ArrayType
from pyspark.sql.window import Window
from pyspark.sql.functions import lag, isnull, udf, col, lit, row_number, unix_timestamp, from_unixtime, months_between, count, when, isnull
from pyspark import StorageLevel

# COMMAND ----------

if __name__ == '__main__':

  spark = SparkSession.builder.appName('data_prep_persist').getOrCreate()

  # COMMAND ----------

  # Define Acquisition Schema.
  orig_fields = [StructField('fico', StringType(), True),
                 StructField('dt_first_pi', StringType(), True),
                 StructField('flag_fthb', StringType(), True),
                 StructField('dt_matr', StringType(), True),
                 StructField('cd_msa', StringType(), True),
                 StructField('mi_pct', StringType(), True),
                 StructField('cnt_units', StringType(), True),
                 StructField('occpy_sts', StringType(), True),
                 StructField('cltv', StringType(), True),
                 StructField('dti', StringType(), True),
                 StructField('orig_upb', StringType(), True),
                 StructField('ltv', StringType(), True),
                 StructField('int_rt', StringType(), True),
                 StructField('channel', StringType(), True),
                 StructField('ppmt_pnlty', StringType(), True),
                 StructField('prod_type', StringType(), True),
                 StructField('st', StringType(), True),
                 StructField('prop_type', StringType(), True),
                 StructField('zipcode', StringType(), True),
                 StructField('id_loan', StringType(), True),
                 StructField('loan_purpose', StringType(), True),
                 StructField('orig_loan_term', StringType(), True),
                 StructField('cnt_borr', StringType(), True),
                 StructField('seller_name', StringType(), True),
                 StructField('servicer_name', StringType(), True),
                 StructField('flag_sc', StringType(), True),
                 StructField('pre_harp', StringType(), True)]

  orig_schema = StructType(fields=orig_fields)

  # COMMAND ----------

  # Define Transaction Schema.
  svcg_fields = [StructField('id_loan', StringType(), True),
                 StructField('period', StringType(), True),
                 StructField('act_endg_upb', StringType(), True),
                 StructField('delq_sts', StringType(), True),
                 StructField('loan_age', StringType(), True),
                 StructField('mths_remng', StringType(), True),
                 StructField('repch_flag', StringType(), True),
                 StructField('flag_mod', StringType(), True),
                 StructField('cd_zero_bal', StringType(), True),
                 StructField('dt_zero_bal', StringType(), True),
                 StructField('new_int_rt', StringType(), True),
                 StructField('amt_non_int_brng_upb', StringType(), True),
                 StructField('dt_lst_pi', StringType(), True),
                 StructField('mi_recoveries', StringType(), True),
                 StructField('net_sale_proceeds', StringType(), True),
                 StructField('non_mi_recoveries', StringType(), True),
                 StructField('expenses', StringType(), True),
                 StructField('legal_costs', StringType(), True),
                 StructField('maint_pres_costs', StringType(), True),
                 StructField('taxes_ins_costs', StringType(), True),
                 StructField('misc_costs', StringType(), True),
                 StructField('actual_loss', StringType(), True),
                 StructField('modcost', StringType(), True),
                 StructField('stepmod_ind', StringType(), True),
                 StructField('dpm_ind', StringType(), True),
                 StructField('eltv', StringType(), True)]

  svcg_schema = StructType(fields=svcg_fields)

  # COMMAND ----------

  # UDF to calculate prior_variables.
  def calulate_prior_variables(id_loan, New_Int_rt, Period, delq_sts, lag_id, lag2_id, lag_act_endg_upb, lag_delq_sts, \
                               lag2_delq_sts, lag_period, lag_new_int_rt, lag_non_int_brng_upb):
    if lag_id is None:
      prior_upb=0
      prior_int_rt=New_Int_rt
      prior_delq_sts='00'
      prior_delq_sts_2='00'
      prior_period = 0
      prior_frb_upb = 0
    else:
      prior_delq_sts=lag_delq_sts
      if id_loan==lag2_id:
        prior_delq_sts_2=lag2_delq_sts
      else:
        prior_delq_sts_2=None
      prior_period=lag_period
      prior_upb=lag_act_endg_upb
      prior_int_rt=lag_new_int_rt 
      prior_frb_upb = lag_non_int_brng_upb

    if Period is None:
      Period=0
    if prior_period is None:
      prior_period=0
      
    period_diff = Period - prior_period
    if delq_sts != 'R':
      delq_sts_new = delq_sts
    elif delq_sts == 'R' and period_diff == 1 and prior_delq_sts == '5':
      delq_sts_new = '6'
    elif delq_sts == 'R' and period_diff == 1 and prior_delq_sts == '3':
      delq_sts_new = '4'
    elif delq_sts == 'R' and period_diff == 1 and prior_delq_sts == '2':
      delq_sts_new = '3'
    else:
      delq_sts_new = None  
      
    return prior_delq_sts, prior_delq_sts_2, prior_period, prior_upb, prior_int_rt, prior_frb_upb, delq_sts_new

  schema = StructType([
    StructField("prior_delq_sts", StringType(), True),
    StructField("prior_delq_sts_2", StringType(), True),
    StructField("prior_period", IntegerType(), True),
    StructField("prior_upb", DecimalType(), True),
    StructField("prior_int_rt", DecimalType(), True),
    StructField("prior_frb_upb", IntegerType(), True),
    StructField("delq_sts_new", StringType(), True)
  ])

  calulate_prior_variables_udf = udf(calulate_prior_variables,schema)

  # COMMAND ----------

  # UDF to calculate UPB.
  def calc_upb(act_endg_upb_, prior_upb_, orig_upb_):
    if (act_endg_upb_ != 0) & (act_endg_upb_ is not None):
      upb = act_endg_upb_
    elif (prior_upb_ != 0) & (prior_upb_ is not None):
      upb = prior_upb_
    else:
      upb = orig_upb_
      
    return upb

  calc_upb_udf = udf(calc_upb,DecimalType())

  # COMMAND ----------

  # UDF to calculate current_int_rt.
  def calc_current_int_rt(new_int_rt, prior_int_rt):
    if (new_int_rt != 0) & (new_int_rt is not None):
      current_int_rt = new_int_rt
    else:
      current_int_rt = prior_int_rt
      
    return current_int_rt

  calc_current_int_rt_udf = udf(calc_current_int_rt,DecimalType())

  # COMMAND ----------

  # Read input files.
  orig_path = 's3a://freddie-mac-all/acquisition/'
  orig_df = spark.read.option("delimiter", "|").option("encoding", "UTF-8").schema(orig_schema).csv(orig_path)

  svcg_path= 's3a://freddie-mac-all/performance/'
  svcg_df = spark.read.option("delimiter", "|").option("encoding", "UTF-8").schema(svcg_schema).csv(svcg_path)

  # COMMAND ----------

  # Typecasting.
  orig_df = orig_df.withColumn("fico", orig_df['fico'].cast(IntegerType()))
  orig_df = orig_df.withColumn("dt_first_pi", orig_df['dt_first_pi'].cast(IntegerType()))
  orig_df = orig_df.withColumn("dt_matr", orig_df['dt_matr'].cast(IntegerType()))
  orig_df = orig_df.withColumn("cd_msa", orig_df['cd_msa'].cast(IntegerType()))
  orig_df = orig_df.withColumn("mi_pct", orig_df['mi_pct'].cast(IntegerType()))
  orig_df = orig_df.withColumn("cnt_units", orig_df['cnt_units'].cast(IntegerType()))
  orig_df = orig_df.withColumn("cltv", orig_df['cltv'].cast(DecimalType()))
  orig_df = orig_df.withColumn("dti", orig_df['dti'].cast(IntegerType()))
  orig_df = orig_df.withColumn("orig_upb", orig_df['orig_upb'].cast(IntegerType()))
  orig_df = orig_df.withColumn("ltv", orig_df['ltv'].cast(IntegerType()))
  orig_df = orig_df.withColumn("int_rt", orig_df['int_rt'].cast(DecimalType()))
  orig_df = orig_df.withColumn("zipcode", orig_df['zipcode'].cast(IntegerType()))
  orig_df = orig_df.withColumn("orig_loan_term", orig_df['orig_loan_term'].cast(IntegerType()))
  orig_df = orig_df.withColumn("cnt_borr", orig_df['cnt_borr'].cast(IntegerType()))

  svcg_df = svcg_df.withColumn("period", svcg_df['period'].cast(IntegerType()))
  svcg_df = svcg_df.withColumn("act_endg_upb", svcg_df['act_endg_upb'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("loan_age", svcg_df['loan_age'].cast(IntegerType()))
  svcg_df = svcg_df.withColumn("mths_remng", svcg_df['mths_remng'].cast(IntegerType()))
  svcg_df = svcg_df.withColumn("cd_zero_bal", svcg_df['cd_zero_bal'].cast(IntegerType()))
  svcg_df = svcg_df.withColumn("dt_zero_bal", svcg_df['dt_zero_bal'].cast(IntegerType()))
  svcg_df = svcg_df.withColumn("new_int_rt", svcg_df['new_int_rt'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("amt_non_int_brng_upb", svcg_df['amt_non_int_brng_upb'].cast(IntegerType()))
  svcg_df = svcg_df.withColumn("dt_lst_pi", svcg_df['dt_lst_pi'].cast(IntegerType()))
  svcg_df = svcg_df.withColumn("mi_recoveries", svcg_df['mi_recoveries'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("non_mi_recoveries", svcg_df['non_mi_recoveries'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("expenses", svcg_df['expenses'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("legal_costs", svcg_df['legal_costs'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("maint_pres_costs", svcg_df['maint_pres_costs'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("taxes_ins_costs", svcg_df['taxes_ins_costs'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("misc_costs", svcg_df['misc_costs'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("actual_loss", svcg_df['actual_loss'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("modcost", svcg_df['modcost'].cast(DecimalType()))
  svcg_df = svcg_df.withColumn("eltv", svcg_df['eltv'].cast(DecimalType()))

  # COMMAND ----------

  orig_df.persist(StorageLevel.DISK_ONLY)
  svcg_df.persist(StorageLevel.DISK_ONLY)


  # COMMAND ----------

  # Create SVCG Detail Records
  # Step-1 >> Drop duplicates and Null Loan-ID if any from both the tables.
  orig_df = orig_df.dropDuplicates()
  svcg_df = svcg_df.dropDuplicates()

  orig_df = orig_df.na.drop(subset='id_loan')
  svcg_df = svcg_df.na.drop(subset='id_loan')

  # Step-2 >> Create SVCG DETAIL table.
  orig_df.createOrReplaceTempView("orig_table")
  svcg_df.createOrReplaceTempView("svcg_table")

  df1 = (spark.sql("""select distinct a.*, b.orig_upb 
                      from svcg_table a, orig_table b
                      where a.id_loan  = b.id_loan
                      order by a.Id_loan"""))

  # Step-3 >> Define  WINDOW and Create LAG variables.
  window = Window.partitionBy("id_loan").orderBy("period")
  lagid = lag("id_loan", 1).over(window)
  lag2id = lag("id_loan", 2).over(window)
  lagactendgupb = lag("act_endg_upb", 1).over(window)
  lagdelqsts = lag("delq_sts", 1).over(window)
  lag2delqsts = lag("delq_sts", 2).over(window)
  lagperiod = lag("period", 1).over(window)
  lagnewintrt = lag("new_Int_rt", 1).over(window)
  lagnonintbrngupb = lag("amt_non_int_brng_upb", 1).over(window)


  df2 = df1.withColumn("lag_id_", lagid) \
           .withColumn("lag2_id_", lag2id) \
           .withColumn("lag_act_endg_upb_", lagactendgupb) \
           .withColumn("lag_delq_sts_", lagdelqsts) \
           .withColumn("lag2_delq_sts_", lag2delqsts) \
           .withColumn("lag_period_", lagperiod) \
           .withColumn("lag_new_int_rt_", lagnewintrt) \
           .withColumn("lag_non_int_brng_upb_", lagnonintbrngupb)


  # Step-4 >> Order by loan_id and period.
  df3 = df2.orderBy('id_loan','period')

  # Step-5 >> Calculate prior variables using pre-defined UDF.
  df4 = df3.withColumn("prior", calulate_prior_variables_udf(col("id_loan"), col("new_int_rt"), col("period"), col("delq_sts"), \
                                                              col("lag_id_"), col("lag2_id_"), col("lag_act_endg_upb_"), \
                                                              col("lag_delq_sts_"), col("lag2_delq_sts_"), col("lag_period_"), \
                                                              col("lag_new_int_rt_"), col("lag_non_int_brng_upb_")))


  # Step-6 >> Drop temporary lag variables.
  svcg_dtls = df4.drop("lag_act_endg_upb_", "lag2_id_", "lag_delq_sts_", "lag2_delq_sts_", "lag_period_", "lag_new_int_rt_")

  # COMMAND ----------

  svcg_dtls.persist(StorageLevel.DISK_ONLY)


  # COMMAND ----------

  # Create a TempView for svcg_dtls.
  svcg_dtls.createOrReplaceTempView("svcg_dtls_table")

  # COMMAND ----------

  # Create Terminated  Records
  # Step-1 >> Select all the records from SVCG DETAILS file.
  trm_rcd  = (spark.sql("""SELECT * FROM svcg_dtls_table ORDER BY id_loan, period"""))

  trm_rcd = trm_rcd \
  .withColumnRenamed("id_loan","t_id_loan") \
  .withColumnRenamed("period","t_period") \
  .withColumnRenamed("current_int_rt","t_current_int_rt") \
  .withColumnRenamed("repch_flag","t_repch_flag") \
  .withColumnRenamed("cd_zero_bal","t_cd_zero_bal") \
  .withColumnRenamed("dt_zero_bal","t_dt_zero_bal") \
  .withColumnRenamed("new_int_rt","t_new_int_rt") \
  .withColumnRenamed("expenses","t_expenses") \
  .withColumnRenamed("mi_recoveries","t_mi_recoveries") \
  .withColumnRenamed("non_mi_recoveries","t_non_mi_recoveries") \
  .withColumnRenamed("net_sale_proceeds","t_net_sale_proceeds") \
  .withColumnRenamed("actual_loss","t_actual_loss") \
  .withColumnRenamed("legal_costs","t_legal_costs") \
  .withColumnRenamed("taxes_ins_costs","t_taxes_ins_costs") \
  .withColumnRenamed("maint_pres_costs","t_maint_pres_costs") \
  .withColumnRenamed("misc_costs","t_misc_costs") \
  .withColumnRenamed("modcost","t_modcost") \
  .withColumnRenamed("dt_lst_pi","t_dt_lst_pi") \
  .withColumnRenamed("id_delq_sts","t_delq_sts") \
  .withColumnRenamed("delq_sts","t_delq_sts") \
  .withColumnRenamed("cd_zero_bal","t_cd_zero_bal") \
  .withColumnRenamed("act_endg_upb","t_act_endg_upb") \
  .withColumnRenamed("prior","t_prior") \
  .withColumnRenamed("orig_upb","t_orig_upb")


  # Step-2 >> Create a window by loan-id.  Select last period record by ordering in descending manner and selecting first row.
  w = Window.partitionBy("t_id_loan").orderBy(trm_rcd["t_period"].desc())
  trm_rcd = trm_rcd.withColumn("row",row_number().over(w)).filter("row == 1").drop("row")

  # Step-3 >> Select records with respect to value of cd_zero_bal. 
  trm_rcd = trm_rcd.filter((trm_rcd['t_cd_zero_bal'] >=3 ) & (trm_rcd['t_cd_zero_bal'] <= 9))

  # Step-4 >> calculate default-upb.
  trm_rcd = trm_rcd.withColumn("t_default_upb", calc_upb_udf(col("t_act_endg_upb"), col("t_prior.prior_upb"), col("t_orig_upb")))

  # Step-5 >> create terminated records after calulating current interest rate.
  trm_rcd = trm_rcd.withColumn("t_current_int_rt", calc_current_int_rt_udf(col("t_new_int_rt"), col("t_prior.prior_int_rt")))


  # COMMAND ----------

  trm_rcd.persist(StorageLevel.DISK_ONLY)


  # COMMAND ----------

  pop_1 = spark.sql("""select a.id_loan, a.act_endg_upb, a.prior.*, a.orig_upb, 1 as p1_dlq_ind
                         from svcg_dtls_table a , (select id_loan, min(period) as min_period from svcg_dtls_table group by id_loan) b
                         where a.id_loan=b.id_loan and a.period=b.min_period""")
  pop_1  = pop_1 \
  .withColumnRenamed("id_loan","p1_id_loan") \
  .withColumnRenamed("act_endg_upb","p1_act_endg_upb") \
  .withColumnRenamed("prior_upb","p1_prior_upb") \
  .withColumnRenamed("orig_upb","p1_orig_upb")

  pop_1 = pop_1.dropDuplicates()
  pop_1 = pop_1.orderBy("p1_id_loan")

  pop_1 = pop_1.withColumn("p1_dlq_upb", calc_upb_udf(col("p1_act_endg_upb"), col("p1_prior_upb"), col("p1_orig_upb")))

  # COMMAND ----------

  pop_1.persist(StorageLevel.DISK_ONLY)
  

  # COMMAND ----------

  pop_2 = spark.sql("""select a.id_loan, a.act_endg_upb, a.prior.*, a.orig_upb, 2 as p2_dlq_ind
                         from svcg_dtls_table a , (select id_loan, min(period) as min_period from svcg_dtls_table group by id_loan) b
                         where a.id_loan=b.id_loan and a.period=b.min_period""")
  pop_2  = pop_2 \
  .withColumnRenamed("id_loan","p2_id_loan") \
  .withColumnRenamed("act_endg_upb","p2_act_endg_upb") \
  .withColumnRenamed("prior_upb","p2_prior_upb") \
  .withColumnRenamed("orig_upb","p2_orig_upb")

  pop_2 = pop_2.dropDuplicates()
  pop_2 = pop_2.orderBy("p2_id_loan")

  pop_2 = pop_2.withColumn("p2_dlq_upb", calc_upb_udf(col("p2_act_endg_upb"), col("p2_prior_upb"), col("p2_orig_upb")))

  # COMMAND ----------

  pop_2.persist(StorageLevel.DISK_ONLY)


  # COMMAND ----------

  pop_3 = spark.sql("""select a.id_loan, a.act_endg_upb, a.prior.*, a.orig_upb, 3 as p3_dlq_ind
                         from svcg_dtls_table a , (select id_loan, min(period) as min_period from svcg_dtls_table group by id_loan) b
                         where a.id_loan=b.id_loan and a.period=b.min_period""")
  pop_3  = pop_3 \
  .withColumnRenamed("id_loan","p3_id_loan") \
  .withColumnRenamed("act_endg_upb","p3_act_endg_upb") \
  .withColumnRenamed("prior_upb","p3_prior_upb") \
  .withColumnRenamed("orig_upb","p3_orig_upb")

  pop_3 = pop_3.dropDuplicates()
  pop_3 = pop_3.orderBy("p3_id_loan")

  pop_3 = pop_3.withColumn("p3_dlq_upb", calc_upb_udf(col("p3_act_endg_upb"), col("p3_prior_upb"), col("p3_orig_upb")))

  # COMMAND ----------

  pop_3.persist(StorageLevel.DISK_ONLY)

  

  # COMMAND ----------

  pop_4 = spark.sql("""select a.id_loan, a.act_endg_upb, a.prior.*, a.orig_upb, 4 as p4_dlq_ind
                         from svcg_dtls_table a , (select id_loan, min(period) as min_period from svcg_dtls_table group by id_loan) b
                         where a.id_loan=b.id_loan and a.period=b.min_period""")
  pop_4  = pop_4 \
  .withColumnRenamed("id_loan","p4_id_loan") \
  .withColumnRenamed("act_endg_upb","p4_act_endg_upb") \
  .withColumnRenamed("prior_upb","p4_prior_upb") \
  .withColumnRenamed("orig_upb","p4_orig_upb")

  pop_4 = pop_4.dropDuplicates()
  pop_4 = pop_4.orderBy("p4_id_loan")

  pop_4 = pop_4.withColumn("p4_dlq_upb", calc_upb_udf(col("p4_act_endg_upb"), col("p4_prior_upb"), col("p4_orig_upb")))

  # COMMAND ----------

  pop_4.persist(StorageLevel.DISK_ONLY)

  

  # COMMAND ----------

  pop_6 = spark.sql("""select a.id_loan, a.act_endg_upb, a.prior.*, a.orig_upb, 6 as p6_dlq_ind
                         from svcg_dtls_table a , (select id_loan, min(period) as min_period from svcg_dtls_table group by id_loan) b
                         where a.id_loan=b.id_loan and a.period=b.min_period""")
  pop_6  = pop_6 \
  .withColumnRenamed("id_loan","p6_id_loan") \
  .withColumnRenamed("act_endg_upb","p6_act_endg_upb") \
  .withColumnRenamed("prior_upb","p6_prior_upb") \
  .withColumnRenamed("orig_upb","p6_orig_upb")

  pop_6 = pop_6.dropDuplicates()
  pop_6 = pop_6.orderBy("p6_id_loan")

  pop_6 = pop_6.withColumn("p6_dlq_upb", calc_upb_udf(col("p6_act_endg_upb"), col("p6_prior_upb"), col("p6_orig_upb")))

  # COMMAND ----------

  pop_6.persist(StorageLevel.DISK_ONLY)

  

  # COMMAND ----------

  # CREATE D180 INSTANCES
  d180_temp = (spark.sql("""SELECT * FROM svcg_dtls_table ORDER BY id_loan, period"""))
  d180 = d180_temp.filter(d180_temp['prior.delq_sts_new'] == 6)
  d180 = d180.orderBy('id_loan','period')


  # CREATE Pre D180 INSTANCES
  pre180_temp = (spark.sql("""SELECT * FROM svcg_dtls_table ORDER BY id_loan, period"""))
  pre_d180 = pre180_temp.drop(((pre180_temp['prior.delq_sts_new'] >= 6) & (pre180_temp['cd_zero_bal'] == 3)) | \
                            ((pre180_temp['prior.delq_sts_new'] >= 6) & (pre180_temp['delq_sts'] == 'R')))
  pre_d180 = pre_d180.orderBy('id_loan','period')


  # Merge D180 and Pre-D180 instances.
  d180_pr = d180.union(pre_d180)
  d180_pr = d180_pr.dropDuplicates()
  d180_pr = d180_pr.orderBy("id_loan")

  

  # Process merged records.
  d180_pr.createOrReplaceTempView("d180_pr_tab")
  pd_180 = (spark.sql("""select a.id_loan, a.act_endg_upb, a.orig_upb, a.prior.*
                         from d180_pr_tab a, (select id_loan, min(period) as min_period from d180_pr_tab group by id_loan) b
                         where a.id_loan=b.id_loan and a.period=b.min_period"""))

  pd_180  = pd_180 \
  .withColumnRenamed("id_loan","pd180_id_loan") \
  .withColumnRenamed("act_endg_upb","pd180_act_endg_upb") \
  .withColumnRenamed("prior_upb","pd180_prior_upb") \
  .withColumnRenamed("orig_upb","pd180_orig_upb") \

  pd_180 = pd_180.withColumn("pd180_d180_upb", calc_upb_udf(col("pd180_act_endg_upb"), col("pd180_prior_upb"), col("pd180_orig_upb")))
  pd_180 = pd_180.withColumn("pd180_d180_ind", lit(1))

  # COMMAND ----------

  pd_180.persist(StorageLevel.DISK_ONLY)

  # COMMAND ----------

  # CREATE MODIFIED RECORDS
  # Create a dataset containing Modified records.
  mod_loan  = (spark.sql("""SELECT distinct a.id_loan, a.period, a.act_endg_upb, a.orig_upb, a.prior.*
                             FROM svcg_dtls_table a , (SELECT id_loan FROM svcg_dtls_table WHERE flag_mod='Y' ) b
                             WHERE a.id_loan = b.id_loan
                             ORDER BY a.id_loan, a.period"""))

  # Create a mod indicator.
  mod_loan = mod_loan.withColumn("mod_ind", lit(1))

  # Order by loan-id and period.
  mod_loan = mod_loan.orderBy('id_loan','period')

  # Select First modified record.
  w = Window.partitionBy("id_loan").orderBy("period")
  mod_loan = mod_loan.withColumn("row",row_number().over(w)).filter("row == 1").drop("row")

  mod_loan  = mod_loan \
  .withColumnRenamed("id_loan","mod_id_loan") \
  .withColumnRenamed("act_endg_upb","mod_act_endg_upb") \
  .withColumnRenamed("prior_upb","mod_prior_upb") \
  .withColumnRenamed("orig_upb","mod_orig_upb") 

  mod_loan = mod_loan.withColumn("mod_upb", calc_upb_udf(col("mod_act_endg_upb"), col("mod_prior_upb"), col("mod_orig_upb")))

  # COMMAND ----------

  mod_loan.persist(StorageLevel.DISK_ONLY)
  

  # COMMAND ----------

  # Define SQL tables to be joined later.
  orig_df.createOrReplaceTempView("orig_tab")
  trm_rcd.createOrReplaceTempView("trm_rcd_tab")
  pop_1.createOrReplaceTempView("pop_1_tab")
  pop_2.createOrReplaceTempView("pop_2_tab")
  pop_3.createOrReplaceTempView("pop_3_tab")
  pop_4.createOrReplaceTempView("pop_4_tab")
  pop_6.createOrReplaceTempView("pop_6_tab")
  pd_180.createOrReplaceTempView("pd_d180_tab")
  mod_loan.createOrReplaceTempView("mod_loan_tab")

  # COMMAND ----------

  # LAST STEP >> JOIN ALL TABLES
  final = (spark.sql( """select
  o.*,
  t.t_current_int_rt,
  t.t_repch_flag,
  t.t_cd_zero_bal,
  t.t_dt_zero_bal as zero_bal_period,
  t.t_expenses,
  t.t_mi_recoveries,
  t.t_non_mi_recoveries,
  t.t_net_sale_proceeds,
  t.t_actual_loss,
  t.t_legal_costs,
  t.t_taxes_ins_costs  as maint_pres_costs,
  t.t_maint_pres_costs as taxes_ins_costs,
  t.t_misc_costs,
  t.t_modcost,
  t.t_dt_lst_pi,
  t.t_delq_sts as zero_bal_delq_sts,
  (case when t.t_cd_zero_bal in ('01','06') then 1 end) as prepay_count,
  (case when t.t_cd_zero_bal in ('03','09') then 1 end) as default_count,
  (case when t.t_cd_zero_bal in ('01','06') then t.t_prior.prior_upb end) as prepay_upb,
  (case when t.t_cd_zero_bal in ('') then t.t_act_endg_upb end) as rmng_upb,
  p1.p1_dlq_ind as dlq1_ever30_ind,
  p1.p1_dlq_upb as dlq1_ever30_upb,
  p2.p2_dlq_ind as dlq2_ever60_ind,
  p2.p2_dlq_upb as dlq2_ever60_upb,
  p3.p3_dlq_ind as dlq3_everd90_ind,
  p3.p3_dlq_upb as dlq3_everd90_upb,
  p4.p4_dlq_ind as dlq4_everd120_ind,
  p4.p4_dlq_upb as dlq4_everd120_upb,
  p6.p6_dlq_ind as dlq6_everd180_ind,
  p6.p6_dlq_upb as dlq6_everd180_upb,
  n.pd180_d180_ind as pd_d180_ind,
  n.pd180_d180_upb as pd_d180_upb,
  m.mod_ind as mod_ind,
  m.mod_upb as mod_upb

  from orig_tab o
  left join trm_rcd_tab t on o.id_loan = t.t_id_loan
  left join pop_1_tab p1 on o.id_loan = p1.p1_id_loan
  left join pop_2_tab p2 on o.id_loan = p2.p2_id_loan
  left join pop_3_tab p3 on o.id_loan = p3.p3_id_loan
  left join pop_4_tab p4 on o.id_loan = p4.p4_id_loan
  left join pop_6_tab p6 on o.id_loan = p6.p6_id_loan
  left join pd_d180_tab n on o.id_loan = n.pd180_id_loan
  left join mod_loan_tab m on o.id_loan = m.mod_id_loan

  order by o.id_loan
  """))

  # COMMAND ----------

  # Save final output file.
  finalOut = "s3a://freddie-mac-all/processed-all-persist-coalesce-20/"
  final.coalesce(20).write.option("delimiter", "|").option("header", "true").mode("overwrite").csv(finalOut)

  orig_df.unpersist()
  svcg_df.unpersist()
  svcg_dtls.unpersist()
  trm_rcd.unpersist()
  pop_1.unpersist()
  pop_2.unpersist()
  pop_3.unpersist()
  pop_4.unpersist()
  pop_6.unpersist()
  pd_180.unpersist()
  mod_loan.unpersist()

  # COMMAND ----------


