#!/bin/bash

../../deepMedicRun -dev cuda -train ./configFiles/train/trainConfig.cfg -resetOptimizer -model ~/DISS/milestones/DM_V0_X/output/cnnModels/trainSession/*final*.save #&>run_output

