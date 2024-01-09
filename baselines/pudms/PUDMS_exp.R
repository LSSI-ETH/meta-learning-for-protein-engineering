library(PUlasso)
library(pudms)
library(stringr)
library(dplyr)


parent_path <- ""
setwd(parent_path)


edit_distance_list <- c(5,6,7)

for (edit_distance in edit_distance_list){
  pos_path <- paste(parent_path, '/pos_data_edit_distance/ed_', edit_distance, '/', sep = '')
  un_path <- paste(parent_path, '/un_data_edit_distance/ed_', edit_distance, '/', sep = '')
  pos_file_list <- c(list.files(path = pos_path))

  test_pos <- paste('test_sets/test_ed_', edit_distance, '_pos.txt', sep = '')
  test_neg <- paste('test_sets/test_ed_', edit_distance, '_neg.txt', sep = '')
  pudata_test = create_protein_dat(path_l = test_pos, path_u = test_neg) 

  for (file in pos_file_list){
    truncate_factor <- str_match(file,'pos_ed_[0-9].*_([0-9].*)_meta_([0-3]).txt')[,2]
    metaset <- str_match(file,'pos_ed_[0-9].*_([0-9].*)_meta_([0-3]).txt')[,3]

    pos_file <- paste(pos_path, 'pos_ed_', edit_distance, '_', truncate_factor, '_meta_', metaset, '.txt', sep = '')
    unlab_file <- paste(un_path, 'un_ed_', edit_distance, '_',  truncate_factor, '_meta_', metaset, '.txt', sep = '')
    
    
    py =NULL         # Proportion of positive sequences in unlabeled set (i.e. fraction functional).
    # NULL scans a range of possible py values between 1e-3 and 0.5
    order = 1       # Model order: 1 for main effects or 2 for pairwise
    refstate = NULL   # Reference state for regression.  
    refstate = as.list(strsplit('TPMGGRPRYFLSPSGVA',''))[[1]] #consensus sequence of full data
    #nobs_thresh = 10  # Filters out columns in X that sum to less than nobs_thresh
    refstate = as.list(strsplit('TPMGGRPRYFLSPSGVA',''))[[1]] #consensus sequence of full data
    
    # Filters out columns in X that sum to less than nobs_thresh
    if(edit_distance<5){ nobs_thresh = 1 }
    else{nobs_thresh = 10}
    
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
    
    predsoutfile <- paste('predictions/ed_', edit_distance, '/test_preds_', truncate_factor, '_order_', order, '_meta_', metaset, '_ed_', edit_distance, '.csv', sep = '')

    write.csv(roc$roc_data, file= predsoutfile)
  }
}