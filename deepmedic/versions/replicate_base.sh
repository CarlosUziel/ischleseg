#!/bin/bash
echo DM_V0_{0..4} | xargs -n 1 cp DM_V0_base/run_train.sh
echo DM_V0_{0..4} | xargs -n 1 cp DM_V0_base/run_test.sh
echo DM_V0_{0..4}/configFiles/train/trainConfig.cfg | xargs -n 1 cp DM_V0_base/configFiles/train/trainConfig.cfg
echo DM_V0_{0..4}/configFiles/test/testConfig.cfg | xargs -n 1 cp DM_V0_base/configFiles/test/testConfig.cfg

echo DM_V1_{0..4} | xargs -n 1 cp DM_V1_base/run_train.sh
echo DM_V1_{0..4} | xargs -n 1 cp DM_V1_base/run_test.sh
echo DM_V1_{0..4}/configFiles/train/trainConfig.cfg | xargs -n 1 cp DM_V1_base/configFiles/train/trainConfig.cfg
echo DM_V1_{0..4}/configFiles/test/testConfig.cfg | xargs -n 1 cp DM_V1_base/configFiles/test/testConfig.cfg

echo DM_V0_transfer_{0..4} | xargs -n 1 cp DM_V0_transfer_base/run_transfer.sh
echo DM_V0_transfer_{0..4} | xargs -n 1 cp DM_V0_transfer_base/run_test.sh
echo DM_V0_transfer_{0..4}/configFiles/train/trainConfig.cfg | xargs -n 1 cp DM_V0_transfer_base/configFiles/train/trainConfig.cfg
echo DM_V0_transfer_{0..4}/configFiles/test/testConfig.cfg | xargs -n 1 cp DM_V0_transfer_base/configFiles/test/testConfig.cfg

echo DM_V1_transfer_{0..4} | xargs -n 1 cp DM_V1_transfer_base/run_transfer.sh
echo DM_V1_transfer_{0..4} | xargs -n 1 cp DM_V1_transfer_base/run_test.sh
echo DM_V1_transfer_{0..4}/configFiles/train/trainConfig.cfg | xargs -n 1 cp DM_V1_transfer_base/configFiles/train/trainConfig.cfg
echo DM_V1_transfer_{0..4}/configFiles/test/testConfig.cfg | xargs -n 1 cp DM_V1_transfer_base/configFiles/test/testConfig.cfg

