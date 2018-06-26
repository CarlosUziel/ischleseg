#!/bin/bash

../../deepMedicRun -dev cuda -test ./configFiles/test/testConfig_0.cfg -model ./output/cnnModels/trainSession/*final*.save
../../deepMedicRun -dev cuda -test ./configFiles/test/testConfig_1.cfg -model ./output/cnnModels/trainSession/*final*.save
