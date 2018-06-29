#!/bin/bash

rm -rf ./DM_V0_0/output
../deepMedicRun -dev cuda -newModel ./DM_V0_0/configFiles/model/modelConfig.cfg -train ./DM_V0_0/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V0_transfer_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_0/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V0_1/output
../deepMedicRun -dev cuda -newModel ./DM_V0_1/configFiles/model/modelConfig.cfg -train ./DM_V0_1/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V0_transfer_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_1/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V0_2/output
../deepMedicRun -dev cuda -newModel ./DM_V0_2/configFiles/model/modelConfig.cfg -train ./DM_V0_2/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V0_transfer_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_2/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V0_3/output
../deepMedicRun -dev cuda -newModel ./DM_V0_3/configFiles/model/modelConfig.cfg -train ./DM_V0_3/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V0_transfer_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_3/output/cnnModels/trainSession/*final*.save

rm -rf ./DM_V0_4/output
../deepMedicRun -dev cuda -newModel ./DM_V0_4/configFiles/model/modelConfig.cfg -train ./DM_V0_4/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V0_transfer_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_4/output/cnnModels/trainSession/*final*.save
