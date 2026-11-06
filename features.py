import numpy as np
import pandas as pd

CAT_COLS = ["Month", "VisitorType", "OperatingSystems", "Browser", "Region", "TrafficType"]

def add_features(X):
    X = X.copy()
    X["TotalPages"] = X["Administrative"] + X["Informational"] + X["ProductRelated"]
    X["TotalDuration"] = X["Administrative_Duration"] + X["Informational_Duration"] + X["ProductRelated_Duration"]
    pages = X["TotalPages"].replace(0, np.nan)
    X["AvgTimePerPage"] = (X["TotalDuration"] / pages).fillna(0)
    X["ProductShare"] = (X["ProductRelated"] / pages).fillna(0)
    X["HasPageValue"] = (X["PageValues"] > 0).astype(int)
    X["Weekend"] = X["Weekend"].astype(int)
    for c in CAT_COLS:
        X[c] = X[c].astype(str)
    return X
