library(PUlasso)
library(pudms)
library(stringr)
library(dplyr)


parent_path <- ""
setwd(parent_path)

pos_path <-paste(parent_path, '/pos_data/', sep = '')
un_path <-paste(parent_path, '/un_data/', sep = '')
pos_file_list <- c(list.files(path = pos_path))

test_pos = 'test_sets/test_pos.txt'
test_neg = 'test_sets/test_neg.txt'
pudata_test = create_protein_dat(path_l = test_pos, path_u = test_neg) 

for (file in pos_file_list){
  truncate_factor <- str_match(file,'pos_([0-9].*)_meta_([0-3]).txt')[,2]
  metaset <- str_match(file,'pos_([0-9].*)_meta_([0-3]).txt')[,3]

  pos_file <- paste(pos_path, 'pos_', truncate_factor, '_meta_', metaset, '.txt', sep = '')
  unlab_file <- paste(un_path, 'un_', truncate_factor, '_meta_', metaset, '.txt', sep = '')
  
  py =NULL         # Proportion of positive sequences in unlabeled set (i.e. fraction functional).
  # NULL scans a range of possible py values between 1e-3 and 0.5
  order = 1       # Model order: 1 for main effects or 2 for pairwise
  refstate = NULL   # Reference state for regression.  
  refstate = as.list(strsplit('TPMGGRPRYFLSPSGVA',''))[[1]] #consensus sequence of full data
  nobs_thresh = 10  # Filters out columns in X that sum to less than nobs_thresh
  n_eff_prop = 1    # Scales the p-values to account for redundant sequence sampling at the NGS step. 
  
  # OUTPUT FILES
  outroc = 'Rocker_CV_ROC.png'
  outcsv<- paste('Rocker_parameters_trunc_', truncate_factor, '.csv', sep = '')

  # CREATE A PROTEIN DATA SET
  pudata = create_protein_dat(path_l = pos_file, path_u = unlab_file) 

  # PERFORM CROSS-VALIDATED FITTING OF PU MODEL
  cvfit = v.pudms(protein_dat = pudata,
                  py1 = py,
                  order = order,
                  refstate = refstate,
                  nobs_thresh = nobs_thresh,
                  n_eff_prop = n_eff_prop,
                  nhyperparam = 10, # The number of py values to scan. Log spaced between 1e-3 and 0.5
                  nfolds = 5,       # The number of cross-validation folds
                  nCores = 10)      # The number of threads to use for CV.  
                  
  # REFIT ALL THE DATA WITH THE OPTIMAL PY VALUE AND WRITE MODEL PARAMETERS/PVALUES TO CSV
  optpy = cvfit$py1.opt
  cat("The optimal py value is", optpy, "\nRefitting model on all the data with this py value\n")
  fit <- pudms(protein_dat = pudata, 
              py1 = optpy,
              order = order,
              refstate = refstate,
              nobs_thresh = nobs_thresh,
              n_eff_prop = n_eff_prop,
              outfile = outcsv) 

  roc = adjusted_roc_curve(coef = coef(fit$fit),
                           test_grouped_dat = pudata_test,
                           order = order,
                           #refstate = fit$refstate,
                           refstate = refstate,
                           verbose = T,
                           py1 = optpy,
                           plot = T)
  
  predsoutfile <- paste('predictions/test_preds_', truncate_factor, '_order_', order, '_meta_', metaset, '.csv', sep = '')

  write.csv(roc$roc_data, file= predsoutfile)
}