#!/bin/bash

rm -rf output
../../deepMedicRun -dev cuda -train ./configFiles/train/trainConfig.cfg -resetOptimizer -model ../DM_V2/output/cnnModels/trainSession/*final*.save

