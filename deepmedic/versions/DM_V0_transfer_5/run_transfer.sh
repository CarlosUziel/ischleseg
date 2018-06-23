#!/bin/bash

rm -rf output
../../deepMedicRun -dev cuda -train ./configFiles/train/trainConfig.cfg -resetOptimizer -model ../DM_V0_X/output/cnnModels/trainSession/*final*.save

