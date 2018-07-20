#!/bin/bash
../../deepMedicRun -dev cuda -test ./configFiles/test/testConfig.cfg -model ./output/cnnModels/trainSession/*final*.save
../../deepMedicRun -dev cuda -test ./configFiles/validation/valConfig.cfg -model ./output/cnnModels/trainSession/*final*.save

