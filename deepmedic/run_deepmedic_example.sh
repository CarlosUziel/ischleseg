#!/bin/bash

./deepMedicRun -dev cuda -newModel ./examples/configFiles/deepMedic/model/modelConfig.cfg \
                       		-train ./examples/configFiles/deepMedic/train/trainConfig.cfg
