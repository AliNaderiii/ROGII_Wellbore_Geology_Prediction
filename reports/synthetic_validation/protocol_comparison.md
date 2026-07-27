# Validation protocol comparison

Protocols are reported separately; no score is averaged across protocols.

| protocol          |   n_wells |   n_scored_points |   prefix_min |   prefix_median |   suffix_min |   suffix_median |   global_rmse |   median_well_rmse |   worst10_rmse |   failure_count |
|:------------------|----------:|------------------:|-------------:|----------------:|-------------:|----------------:|--------------:|-------------------:|---------------:|----------------:|
| INVALID_in_sample |        40 |            682696 |         1682 |            3831 |         1291 |          2381   |       3.22275 |            1.06118 |        9.05089 |               0 |
| same_well_masked  |        40 |            852066 |          200 |            1655 |         1291 |          2352.5 |       2.54855 |            1.22528 |        8.5416  |               0 |
| unseen_well       |        40 |            877752 |         1682 |            3831 |         1291 |          2381   |       3.23406 |            1.59401 |        9.05841 |               0 |
