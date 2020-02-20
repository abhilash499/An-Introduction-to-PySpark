* Process flow;
* Step 1 : Read Origination & Performance files for all available quarters;
* Step 2 : Identify the first instances of loan being ever d30, ever d60, ever d90, ever d120, ever d180;
* Step 3 : Create a dataset containing D180 instance or pre-d180 default instance for every loan;
* Step 4 : Create a dataset with first instance of modification for every loan;
* Step 5 : Merge all datasets to create a master dataset;
* Step 6 : Create format variables in the master dataset;


option ERRORS=0;

libname tmp '<<path to store temp files';
%let dir=<<parent folder path where text files are copied to under quarterly sub folders>>;
%let outdir=<<output folder where summary file will be saved;


%let endd='31DEC2017'd;  *Set this to the orign file quarter end date;
%let nqtr= 76;           *Set this to the number of quarters between Q11999 and orign quarter end date;


%macro obscnt(dsn);
%global nobs dsnid;
%let nobs=.;

%let dsnid = %sysfunc(open(&dsn));

%if &dsnid %then %do;
     %let nobs=%sysfunc(attrn(&dsnid,nlobs));
     %let rc  =%sysfunc(close(&dsnid));
%end;
%else %do;
     %put Unable to open &dsn - %sysfunc(sysmsg());
%end;

%mend;


%macro extract(qtr_prd);

data svcg_&qtr_prd;
infile "&dir./historical_data1_time_&qtr_prd..txt"  dlm='|' MISSOVER DSD lrecl=32767 firstobs=1;
input
ID_loan              : $12.
Period               : 8.
Act_endg_upb         : 8.
delq_sts             : $8.
loan_age             : 8.
mths_remng           : 8.
repch_flag           : $1.
flag_mod             : $1.
CD_Zero_BAL          : $3.
Dt_zero_BAL          : 8.
New_Int_rt           : 8.
Amt_Non_Int_Brng_Upb : 12.
Dt_Lst_Pi            : 6.
MI_Recoveries        : 12.
Net_Sale_Proceeds    : $14.
Non_MI_Recoveries    : 12.
Expenses             : 12.
legal_costs          : 12.
maint_pres_costs     : 12.
taxes_ins_costs      : 12.
misc_costs           : 12.
actual_loss          : 12.
modcost              : 12.
stepmod_ind          : $1.
dpm_ind              : $1.
eltv                 : 12.1
;

data orig_&qtr_prd;
infile "&dir./historical_data1_&qtr_prd..txt" dlm='|' MISSOVER DSD lrecl=32767 firstobs=1 ;

input
fico           : 8.
dt_first_pi    : 8.
flag_fthb      : $1.
dt_matr        : 8.
cd_msa         : 8.
mi_pct         : 8.
cnt_units      : 8.
occpy_sts      : $1.
cltv           : 8.
dti            : 8.
orig_upb       : 8.
ltv            : 8.
int_rt         : 8.
channel        : $1.
ppmt_pnlty     : $1.
prod_type      : $5.
st             : $2.
prop_type      : $2.
zipcode        : $5.
id_loan        : $16.
loan_purpose   : $5.
orig_loan_term : 8.
cnt_borr       : $2.
seller_name    : $30.
servicer_name  : $30.
flag_sc        : $1.
;
run;

proc sort data=svcg_&qtr_prd;
by id_loan period;
run;

proc sql;
create table svcg_dtls_&qtr_prd as 
select distinct a.*, b.orig_upb 
from  
svcg_&qtr_prd a, orig_&qtr_prd b
where a.id_loan  = b.id_loan
order by a.id_loan, a.period;
run;

data svcg_dtls_&qtr_prd;
set  svcg_dtls_&qtr_prd;
by id_loan period;
lag_id_loan      = lag(id_loan);
lag2_id_loan     = lag2(id_loan);
lag_act_endg_upb = lag(act_endg_upb);
lag_delq_sts     = lag(delq_sts);
lag2_delq_sts    = lag2(delq_sts);
lag_period   = lag(period);
lag_new_int_rt   = lag(new_int_rt); 
lag_non_int_brng_upb =lag(amt_non_int_brng_upb);

if first.id_loan then do;
  prior_upb=0; 
  prior_int_rt=new_int_rt;
  prior_delq_sts='00';
  prior_delq_sts_2='00';
  prior_period=.;
  prior_frb_upb = .;
end;
else do;
  prior_delq_sts=lag_delq_sts;
  if id_loan=lag2_id_loan then prior_delq_sts_2=lag2_delq_sts;
  prior_period=lag_period;
  prior_upb=lag_act_endg_upb;
  prior_int_rt=lag_new_int_rt; 
  prior_frb_upb = lag_non_int_brng_upb;
end;

if delq_sts ne 'R' then delq_sts_new = put(delq_sts,$6.) ; 

period_diff= -1* intck('month',mdy( substr(put(period,$6.),5,2),1, substr(put(period,$6.),1,4) ) , mdy( substr(put(prior_period,$6.),5,2),1, substr(put(prior_period,$6.),1,4) ));

if delq_sts='R' and period_diff = 1 and prior_delq_sts='5' then delq_sts_new = '6';
if delq_sts='R' and period_diff = 1 and prior_delq_sts='3' then delq_sts_new = '4';
if delq_sts='R' and period_diff = 1 and prior_delq_sts='2' then delq_sts_new = '3';

drop lag_act_endg_upb lag2_id_loan lag_delq_sts lag2_delq_sts lag_period lag_new_int_rt ;

run;

%macro min_dlq(i);

%let dlq_bkt=&i;

data pop_&dlq_bkt.plus;
set  svcg_dtls_&qtr_prd;
where delq_sts_new="&dlq_bkt";
run;

proc sql;
create table pop_&dlq_bkt._&qtr_prd as
select a.*, 1 as dlq_ind_&dlq_bkt 
from pop_&dlq_bkt.plus a,
(select id_loan, min(period) as period from pop_&dlq_bkt.plus
group by id_loan) b
where a.id_loan=b.id_loan and a.period=b.period;
run;

proc sort data=pop_&dlq_bkt._&qtr_prd noduprecs;
by id_loan; run;

data pop_&dlq_bkt._&qtr_prd;
set  pop_&dlq_bkt._&qtr_prd;
if act_endg_upb not in (0,.) then dlq_upb_&dlq_bkt = act_endg_upb;
else if prior_upb not in (0,.) then dlq_upb_&dlq_bkt = prior_upb;
else dlq_upb_&dlq_bkt = orig_upb;
run;


proc append base=tmp.pop_&dlq_bkt._final data=pop_&dlq_bkt._&qtr_prd; 
run;

%mend;

* Identify the first instances of loan being ever d30, ever d60, ever d90, ever d120, ever d180;

%min_dlq(1);
%min_dlq(2);
%min_dlq(3);
%min_dlq(4);
%min_dlq(6);


data d180_&qtr_prd;
set  svcg_dtls_&qtr_prd;
if delq_sts_new='6';
run;

proc sort data=svcg_dtls_&qtr_prd;
by id_loan period;
run;

data pred180_&qtr_prd;
set  svcg_dtls_&qtr_prd;
by id_loan period;
 if cd_zero_bal='03' or  delq_sts='R';
 if cd_zero_bal='03' and delq_sts_new >= 6 then delete;
 if delq_sts='R'     and delq_sts_new >= 6 then delete;
run;

proc sort data=pred180_&qtr_prd;
by id_loan period; run;


proc append base=d180_pr_&qtr_prd data=d180_&qtr_prd force; run;
proc append base=d180_pr_&qtr_prd data=pred180_&qtr_prd force; run;

proc sort data=d180_pr_&qtr_prd noduprecs; by id_loan; run;

proc sql;
create table pd180_&qtr_prd as
select a.*
from d180_pr_&qtr_prd a,
(select id_loan, min(period) as period from d180_pr_&qtr_prd
group by id_loan) b
where a.id_loan=b.id_loan and a.period=b.period;
run;

data pd180_&qtr_prd;
set pd180_&qtr_prd;
if act_endg_upb not in (0,.) then pd_d180_upb=act_endg_upb;
else if  act_endg_upb in (0,.) then do;
if  prior_upb not in (0,.) then pd_d180_upb=prior_upb;
else pd_d180_upb =orig_upb;
end;
pd_d180_ind=1;
run;

* Create a dataset containing D180 instance or pre-d180 default instance for every loan;

proc append base=tmp.pd_d180 data=pd180_&qtr_prd force; run;

* Create a dataset containing modification records;

Proc sql;
create table mod_loan_&qtr_prd as
SELECT distinct a.*,  c.orig_upb
FROM svcg_dtls_&qtr_prd A , (SELECT ID_LOAN FROM svcg_dtls_&qtr_prd WHERE FLAG_MOD='Y' ) B , orig_&qtr_prd C
 WHERE A.ID_LOAN = B.ID_LOAN
 and   B.ID_LOAN = C.ID_LOAN
 order by a.id_loan, a.period;
 run;

%obscnt(mod_loan_&qtr_prd) ;

%if &nobs >  0 %then %do;
 
data mod_loan_&qtr_prd;
set  mod_loan_&qtr_prd;
by id_loan period;
prior_upb = lag(act_endg_upb);
mod_ind=1;
if flag_mod='Y' then output;
run;

proc sort data=mod_loan_&qtr_prd;
by id_loan period; run;

data mod_loan_&qtr_prd;
set  mod_loan_&qtr_prd;
by id_loan period;
if first.id_loan then output;
if act_endg_upb not in (0,.) then mod_upb=act_endg_upb;
else if prior_upb not in (0,. ) then mod_upb=prior_upb;
else mod_upb = orig_upb;
run;

proc append base=tmp.mod_rcd data=mod_loan_&qtr_prd; 
run;

%end;

data trm_rcd_&qtr_prd;
set  svcg_dtls_&qtr_prd;
by   id_loan period;
if   last.id_loan then output;
run;

data trm_rcd_&qtr_prd;
set  trm_rcd_&qtr_prd;
if cd_zero_bal in ('03','09') then do;
if act_endg_upb not in (0,.) then default_upb=act_endg_upb;
else if act_endg_upb in (0,.) then do;
     if prior_upb in (0,.) then default_upb=orig_upb;
else if prior_upb not in (0,.) then default_upb=prior_upb;
end;
end;

delq_string = prior_delq_sts ||" to "|| delq_sts;
month_string=put(prior_period,$6.) || " to " || put(period,$6.);

     if new_int_rt not in (0,.) then current_int_rt=new_int_rt;
else if new_int_rt in (0,.) then current_int_rt=prior_int_rt;

orign_qtr="&qtr_prd";
vintage = substr(id_loan,3,2);
run;

data dflt_&qtr_prd;
set  trm_rcd_&qtr_prd;
if cd_zero_bal in ('03','09');
if cd_zero_bal = '03' then dflt_delq_sts=delq_sts;
else if cd_zero_bal='09' then do;
    if prior_delq_sts ne 'R' then dflt_delq_sts=prior_delq_sts;
    else dflt_delq_sts=prior_delq_sts_2;
    if prior_delq_sts ne 'R' then acqn_to_dispn=0;
    else acqn_to_dispn=intck('month', mdy( substr(put(prior_period,$6.),5,2), 1,substr(put(prior_period,$6.),1,4)) , mdy( substr(put(period,$6.),5,2), 1,substr(put(period,$6.),1,4)) );
end;

mths_dlq_dflt_dispn=sum(dflt_delq_sts, acqn_to_dispn);
mths_dlq_dflt_acqn=sum(dflt_delq_sts,0);
frb_upb = prior_frb_upb ;

vintage=substr(id_loan,3,2);
run;

proc append base=tmp.all_orign data=orig_&qtr_prd;run;
proc append base=tmp.all_dflt data=dflt_&qtr_prd force; run;
proc append base=tmp.all_trm_rcd data=trm_rcd_&qtr_prd force; run;

proc datasets lib=work nolist kill; quit; run


%mend extract;


%macro loopthru();

%do i = 1 %to &nqtr ;

 data _null_;
   qtr_beg1=intnx('quarter', &endd,1-&i,'b');
   qtr_end1=intnx('quarter', &endd,1-&i,'e');
   qtr_prd = 'Q'||strip(qtr(qtr_end1))||strip(year(qtr_end1));
   call symput('qtr_beg', "'"||put(qtr_beg1,mmddyy10.)||"'");
   call symput('qtr_end', "'"||put(qtr_end1,mmddyy10.)||"'");
   call symput('qtr_prd',qtr_prd);
 run;

%extract(&qtr_prd);

%end;

%mend loopthru;

proc sql;
drop table tmp.all_orign;
drop table tmp.all_dflt;
drop table tmp.all_trm_rcd;
drop table tmp.mod_rcd;
drop table tmp.pd_d180;
drop table tmp.pop_1_final;
drop table tmp.pop_2_final;
drop table tmp.pop_3_final;
drop table tmp.pop_4_final;
drop table tmp.pop_6_final;
run;
quit;

%loopthru(); 

* Calculate dlq_accrued interest, and collateral loss for each liquidated loan;

data tmp.dflts_with_loss_data;
set tmp.all_trm_rcd;
where cd_zero_Bal in ('03','09');
days_dlq = intck('month', mdy(substr(put(dt_lst_pi,$6.),5,2), 1, substr(put(dt_lst_pi,$6.),1,4)),mdy(substr(put(period,$6.),5,2), 1, substr(put(period,$6.),1,4)) ) * 30;
if substr(put(period,$6.),1,4) ne '1999' then do;
  if Net_Sale_Proceeds = 'C' then do;
    collateral_Deficiency=0;
    dlq_accrued_interest=0;
  end;
  else do;
    collateral_deficiency=sum(default_upb, -1* input(net_sale_proceeds,8.));
    dlq_accrued_interest=round(sum(default_upb,-prior_frb_upb)*(current_int_rt-0.35)* days_dlq/360/100,.1);
  end;
end;

run;

proc sql;
create table tmp.all_orign_dtl as 
select
  a.*, 
  i.collateral_deficiency,
  f.current_int_rt,
  f.repch_flag,
  f.cd_zero_bal,
  f.dt_zero_bal as zero_bal_period,
  f.expenses,
  f.mi_recoveries,
  f.non_mi_recoveries,
  f.net_sale_proceeds,
  f.actual_loss,
  f.legal_costs,
  f.taxes_ins_costs  as maint_pres_costs,
  f.maint_pres_costs as taxes_ins_costs,
  f.misc_costs,
  f.modcost,
  f.dt_lst_pi,
  f.delq_sts as zero_bal_delq_sts,
  i.default_upb,
  i.dlq_accrued_interest,
  a1.dlq_ind_1 as dlq_ever30_ind,
  a1.dlq_upb_1 as dlq_ever30_upb,
  a2.dlq_ind_2 as dlq_ever60_ind,
  a2.dlq_upb_2 as dlq_ever60_upb,
  a3.dlq_ind_3 as dlq_everd90_ind,
  a3.dlq_upb_3 as dlq_everd90_upb,
  a4.dlq_ind_4 as dlq_everd120_ind,
  a4.dlq_upb_4 as dlq_everd120_upb,
  a5.dlq_ind_6 as dlq_everd180_ind,
  a5.dlq_upb_6 as dlq_everd180_upb,
  (case when f.cd_zero_bal in ('01','06') then 1 end) as prepay_count,
  (case when f.cd_zero_bal in ('03','09') then 1 end) as default_count,
  (case when f.cd_zero_bal in ('01','06') then f.prior_upb end) as prepay_upb,
  (case when f.cd_zero_bal in ('') then f.act_endg_upb end) as rmng_upb,
  m.mod_ind as mod_ind,
  m.mod_upb as mod_upb,
  n.pd_d180_ind as pd_d180_ind,
  n.pd_d180_upb as pd_d180_upb
from
  tmp.all_orign a
left join
  tmp.all_trm_rcd f
  on
  a.id_loan = f.id_loan
left join
  tmp.dflts_with_loss_data i 
  on
  a.id_loan = i.id_loan
left join
  tmp.pop_1_final a1
  on
  a.id_loan = a1.id_loan
left join
  tmp.pop_2_final a2
  on
  a.id_loan = a2.id_loan
left join
  tmp.pop_3_final a3
  on
  a.id_loan = a3.id_loan
left join
  tmp.pop_4_final a4
  on
  a.id_loan = a4.id_loan
left join
  tmp.pop_6_final a5
  on
  a.id_loan = a5.id_loan
left join
  tmp.mod_rcd m
  on
  a.id_loan = m.id_loan
left join
  tmp.pd_d180   n
  on
  a.id_loan = n.id_loan 
order by
  a.id_loan
; quit;