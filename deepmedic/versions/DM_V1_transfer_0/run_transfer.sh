#!/bin/bash

../../deepMedicRun -dev cuda -train ./configFiles/train/trainConfig.cfg -resetOptimizer -model ../DM_V1_0/output/cnnModels/trainSession/*final*.save #&>run_output

