#!/bin/bash

rm -rf output
../../deepMedicRun -dev cuda -train ./configFiles/train/trainConfig.cfg -resetOptimizer -model ../DM_V3_R/output/cnnModels/trainSession/*final*.save

