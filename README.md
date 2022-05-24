### Train a classifier using xgboost via sklearn interface

# Prerequisites

Install xgboost and scikit-learn, e.g. via conda

# Training

Copy the training file (coffea output) into the local folder and run `train.py`

# Optimising the training

A simple algorithm to tune BDT trainings:
- Choose an as large number of estimators (aka boosting rounds aka trees) as you find reasonable - typical final values are around 1000, with the training time roughly going linearly with the number of estimators
- Tune the `eta` and `max_depth` parameters
- I've always had good experience with using subsamples and putting the value around 0.5, but this can also be experimented with

# Notes

Usually, the training should also work, and perform better, with weights. In first tries, the training didn't seem to perform properly when passing weights. Maybe there's some parameter (e.g. minimum leaf weight) that prevents the training from working properly with the default weights. It may be an option to understand this better and possibly scale the weights so that the training performs better.


More documentation is here: https://xgboost.readthedocs.io/en/stable/parameter.html
