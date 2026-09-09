**1. Setup & Planning**

Agree on a shared data representation (e.g., numeric cols first, categorical after, target last)  
Agree on code structure/interfaces (e.g., every method implements the same predict(X_train, y_train, X_query) signature)  
Set up shared repo (GitHub) and decide task split between the two of you

**2. Data Acquisition & Preprocessing (per dataset — do 1 fully before scaling to 6)**

Download all 6 datasets from UCI  
Read each .names file carefully  
Handle missing values (impute or drop) — note Congressional Vote's ? means "abstain," not missing  
Drop unusable columns (Sample code number, vendor/model name, ERP for Computer Hardware)  
Encode categorical features (VDM, or one-hot for regression cases)  
Normalize numeric features (min-max or z-score)  
Handle special cases: Abalone's sex feature (one-hot or discard), Forest Fires' cyclical month/day encoding, Forest Fires' log-transform of area  
Produce clean, consistent data structure for each of the 6 datasets  

**3. Core Building Blocks**

Minkowski distance function (parameterized by p)  
VDM (or alternative categorical distance)  
Combined numeric+categorical distance function  
Classification error metric  
Mean squared error metric  

**4. Null Models**

Classification null model (plurality class)  
Regression null model (mean of outputs)  

**5. Standard k-NN**

k-NN classifier (plurality vote)  
k-NN regressor (Gaussian/RBF kernel prediction)  
Test both against null model on one dataset to sanity check  

**6. Edited k-NN**

Implement editing loop (k=1, remove misclassified/out-of-threshold points)  
Implement convergence check  
Handle both classification and regression (ε threshold) versions  
Validate against all 6 datasets  

**7. Condensed k-NN**

Implement condensing loop (k=1, build up Z, add misclassified/out-of-threshold points)  
Implement convergence check  
Handle order/seed sensitivity  
Handle both classification and regression versions  
Validate against all 6 datasets  

**8. Hyperparameter Tuning Infrastructure**

Build inner train/validation split logic  
Tune k (grid search)  
Tune p (small grid, e.g., {1,2})  
Tune γ for RBF kernel (log-scale grid)  
Tune ε for regression correctness threshold  
Decide/document tuning methodology clearly (for the report)  

**9. 5×2 Cross-Validation Harness**

Implement 5×2 CV splitting logic  
Wire up all 4 methods (null, k-NN, edited, condensed) × both tasks  
Run experiments across all 6 datasets  
Collect and store results (raw predictions + scores) — don't just print, save to file  
Run statistical comparison between methods (this ties back to your testable hypothesis)  

**10. Results Analysis**

Aggregate results into tables  
Generate figures (e.g., validation error vs. k plot, performance comparisons across methods/datasets)  
Compare reduced dataset sizes (edited/condensed) vs. performance  
Statistical significance testing across the 5×2 folds  

**11. Report Writing (JMLR format, ≤15 pages + appendices)**

Title and authors  
Problem statement + testable hypothesis  
Experimental approach / project design explanation  
Algorithm explanations (conceptual, no code)  
Results (tables/figures + explanation)  
Analysis and discussion  
Conclusion  
References  
Appendix: lessons learned  
Appendix: who-did-what breakdown  

**12. Final Checks**

Verify all math is typeset properly (LaTeX math editor, not plain text)  
Verify figures are vector/high-quality, not screenshots  
Proofread against JMLR formatting requirements  
Confirm page limit (15 pages, excluding appendices)  
Export to PDF  
Submit to "P1 Paper" on Canvas  
