#!/usr/env/python
import argparse
import math
import random

# Activation Functions
def relU_active_func(neuronOutput):
  if neuronOutput <= 0:
    return 0
  else
    return neuronOutput*(math.floor(neuronOutput)+1)
def tanh_active_func(neuronOutput):
  return math.tanh(neuronOutput)

# Loss Functions
def modulus_loss_func(actual, expected, accumulatedLoss=0):
  return accumulatedLoss+abs(actual-expected)

# Weight Adjust Functions
def apply_weights_ajustment(loss, comparableLoss, weights)
def buzz_wadjust_func(weights):
  return weight+random.choice([-0.01,0,0.01])
def random_guess_wadjust_func(weights):
  return random.randrange(-100,101)/100

# Input
input=""
neuronsStructure=()
weights=[]
trainBatchNumber=0
trainBatchSize=0
numOfTrainBatches=0
activeFunc=
lossFunc=
wAdjustFunc=
expectedGoodOutcome=2
expectedBadOutcome=0

# Initialization and Validation
numOfWeights=0
for n in range(1,len(neuronsStructure)):
  numofWeights=numOfWeights+neuronsStructure[n]*neuronsStructure[n-1]
if len(weights} == 0:
  for i in range(0,numOfWeights):
    weighs+=random.randrange(-100,101)/100
else:
  if len(weights) != numOfWeights:
    raise Exception("Number of weights specified must be the same as the links in the fully connected neurons structure. Provided: {} Expected: {}".format(len(weights), numofWeights))

# Training
print("Start of Training...")
for batchCount in range(0, trainBatchNumber):
  loss=0
  for exampleCount in range(0, trainBatchSize):
    # Good Example
    print("Batch #{} Example #{}".format(batchCount,exampleCount))
    digit=str(random.randrange(0,10))
    size=random.randrange(0,6)
    example=""
    for i in range(0,size):
      example+=digit
     thought=activeFunc(think(example))
    thought=activeFunc(think(example))
    previousLoss=loss
    loss=lossFunc(thought, expectedGoodOutcome)
    apply_weights_adjustment(loss, previousLoss, weights)
    # Bad Example
    print("Batch #{} Example #{}".format(batchCount,exampleCount))
    size=random.randrange(0,6)
    example=""
    for i in range(0,size):
      digit=str(random.randrange(0,10))
      example+=digit
    thought=activeFunc(think(example))
    previousLoss=loss
    loss=lossFunc(thought, expectedBadOutcome)
    apply_weights_adjustment(loss, previousLoss, weights)
  print("Loss: {}\nWeights: {}".format(loss, weights))

# Evaluation
thought=activeFunc(think(input))
print(thought)
