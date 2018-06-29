#!/bin/bash

../deepMedicRun -dev cuda -newModel ./DM_V1_0/configFiles/model/modelConfig.cfg -train ./DM_V1_0/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V1_transfer_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_0/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_1/configFiles/model/modelConfig.cfg -train ./DM_V1_1/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V1_transfer_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_1/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_2/configFiles/model/modelConfig.cfg -train ./DM_V1_2/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V1_transfer_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_2/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_3/configFiles/model/modelConfig.cfg -train ./DM_V1_3/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V1_transfer_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_3/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V1_4/configFiles/model/modelConfig.cfg -train ./DM_V1_4/configFiles/train/trainConfig.cfg
#../deepMedicRun -dev cuda -train ./DM_V1_transfer_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V1_4/output/cnnModels/trainSession/*final*.save
