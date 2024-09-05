# ----------------------------------------------------------------------
#
# File: MulTileConstraint.py
#
# Created: 05.09.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from Deeploy.Targets.Generic.TileConstraints.MulTileConstraint import MulTileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from typing import Dict, List, Literal, Optional, Tuple, Union

class PULPMulTileConstraint(MulTileConstraint):

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer1Name = parseDict[cls.dataIn1Name]
        inputBuffer2Name = parseDict[cls.dataIn2Name]
        outputBufferName = parseDict[cls.dataOutName]

        tilerModel.addTensorNumOfEltToModel(ctxt, inputBuffer1Name)
        numElIn1 = tilerModel.getTensorNumberOfEltVar(inputBuffer1Name)
        tilerModel.addConstraint(numElIn1 < 15000)

        tilerModel.addTensorNumOfEltToModel(ctxt, inputBuffer2Name)
        numElIn2 = tilerModel.getTensorNumberOfEltVar(inputBuffer2Name)
        tilerModel.addConstraint(numElIn2 < 15000)

        tilerModel.addTensorNumOfEltToModel(ctxt, outputBufferName)
        numElOut = tilerModel.getTensorNumberOfEltVar(outputBufferName)
        tilerModel.addConstraint(numElOut < 15000)

        return tilerModel
