#!/bin/bash

../deepMedicRun -dev cuda -newModel ./DM_V0_0/configFiles/model/modelConfig.cfg -train ./DM_V0_0/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_0/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_0/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V0_1/configFiles/model/modelConfig.cfg -train ./DM_V0_1/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_1/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_1/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V0_2/configFiles/model/modelConfig.cfg -train ./DM_V0_2/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_2/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_2/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V0_3/configFiles/model/modelConfig.cfg -train ./DM_V0_3/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_3/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_3/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V0_4/configFiles/model/modelConfig.cfg -train ./DM_V0_4/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_4/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_4/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V0_5/configFiles/model/modelConfig.cfg -train ./DM_V0_5/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_5/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_5/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V0_6/configFiles/model/modelConfig.cfg -train ./DM_V0_6/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_6/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_6/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V0_7/configFiles/model/modelConfig.cfg -train ./DM_V0_7/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_7/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_7/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V0_8/configFiles/model/modelConfig.cfg -train ./DM_V0_8/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_8/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_8/output/cnnModels/trainSession/*final*.save

../deepMedicRun -dev cuda -newModel ./DM_V0_9/configFiles/model/modelConfig.cfg -train ./DM_V0_9/configFiles/train/trainConfig.cfg
../deepMedicRun -dev cuda -train ./DM_V0_transfer_9/configFiles/train/trainConfig.cfg -resetOptimizer -model ./DM_V0_9/output/cnnModels/trainSession/*final*.save
