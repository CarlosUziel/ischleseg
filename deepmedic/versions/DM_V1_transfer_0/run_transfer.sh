#!/bin/bash

../../deepMedicRun -dev cuda -train ./configFiles/train/trainConfig.cfg -resetOptimizer -model ../DM_V1_X/output/cnnModels/trainSession/*final*.save #&>run_output

