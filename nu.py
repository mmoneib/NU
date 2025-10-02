#!/usr/env/python
import argparse
import math
import random

# Internal Utility Functions
def __get_functions_ending_with(postfix, directory):
  choices=[]
  for f in directory:
    func=eval(f) # Convert the function name to the function itself. Doesn't really invoke the function.
    if f.endswith(postfix) and callable(func):
      choices.append(f.replace(postfix,""))
  return choices

# Activation Functions
def relU_active_func(neuronOutputs):
  outputs=[]
  for neuronOutput in neuronOutputs:
    if neuronOutput <= 0:
      outputs.append(0)
    else:
      outputs.append(neuronOutput*(math.floor(neuronOutput)+1))
  return outputs
def tanh_active_func(neuronOutput):
  outputs=[]
  for neuronOutput in neuronOutputs:
    outputs.append(math.tanh(neuronOutput))
  return outputs

# Loss Functions
def modulus_loss_func(actual, expected, accumulatedLoss=0):
  return accumulatedLoss+abs(actual-expected)

# Weight Adjust Functions
def buzz_wadjust_func(weights):
  return weight+random.choice([-0.01,0,0.01])
def random_guess_wadjust_func(weights):
  return random.randrange(-100,101)/100

# Thinking
def think(inputToProcess, neuronsStructure, weights):
  neuronOutputs=[]
  numOfInputNeurons=int(neuronsStructure[0])
  inputsToLayer=[]
  inputDivisor=0
  for i in range(0, numOfInputNeurons):
    inputsToLayer.append(int(str(inputToProcess)[inputDivisor:numOfInputNeurons+inputDivisor]))
    inputDivisor+=numOfInputNeurons
  weightsIndex=0
  for layerIndex in range(1,len(neuronsStructure)):
    for neuronInLayerIndex in range(0, int(neuronsStructure[layerIndex])):
      neuronOutput=0
      bias=neuronInLayerIndex+layerIndex+1
      neuronOutput+=bias
      for inputsToLayerIndex in range(0, len(inputsToLayer)):
        neuronOutput+=inputsToLayer[inputsToLayerIndex]*weights[weightsIndex]
        weightsIndex+=1
      neuronOutputs.append(neuronOutput)
    inputsToLayer=neuronOutputs[len(neuronOutputs)-int(neuronsStructure[layerIndex]):]
  return neuronOutputs[len(neuronOutputs)-int(neuronsStructure[len(neuronsStructure)-1]):]    

# Input
parser=argparse.ArgumentParser(prog="NU", description="Neural Net Implementation Using Python")
parser.add_argument("input", type=int, help="Input to be processed.")
parser.add_argument("-a", "--activation", choices=__get_functions_ending_with("_active_func", dir()), required=True, help="Name of activation function.")
parser.add_argument("-s", "--structure", nargs="*", required=True, help="Structure of the newural network as a list of number of nodes per layer.")
parser.add_argument("-v", "--verbose", action="store_true", help="Display debugging statements and detailed output.")
parser.add_argument("-w", "--weights", nargs="*", help="List of weights, one for each link in the fully connected network. The order follows that of the neurons inputs, with each weight being an output of a neuron from the previous layer.")
trainingGroup=parser.add_argument_group("Training")
trainingGroup.add_argument("-b", "--batches", help="Training batches specification as a product: number_of_batches*number_of_examples_per_batch.")
trainingGroup.add_argument("-j", "--weight-adjust", choices=__get_functions_ending_with("_wadjust_func", dir()), required=True, help="Name of weights adjustment function.")
trainingGroup.add_argument("-l", "--loss", required=True, choices=__get_functions_ending_with("_loss_func", dir()), help="Name of loss function.")
args=parser.parse_args()
print(args.input)
print(args.activation)
print(args.weight_adjust)
print(args.loss)
print(args.structure)
print(args.weights)
input=args.input
structure=args.structure
weights=args.weights
trainBatchNumber=0
trainBatchSize=0
numOfTrainBatches=0
activeFunc=eval(args.activation+"_active_func")
lossFunc=eval(args.loss+"_loss_func")
wAdjustFunc=eval(args.weight_adjust+"_wadjust_func")
expectedGoodOutcome=2
expectedBadOutcome=0

# Initialization and Validation
numOfWeights=0
for n in range(1,len(structure)):
  numOfWeights=numOfWeights+int(structure[n])*int(structure[n-1])
  print(numOfWeights)
if weights==None or len(weights) == 0:
  weights=[]
  for i in range(0,numOfWeights):
    weights.append(random.randrange(-100,101)/100)
else:
  if len(weights) != numOfWeights:
    raise Exception("Number of weights specified must be the same as the links in the fully connected neurons structure. Provided: {} Expected: {}".format(len(weights), numofWeights))

# Training
print("Start of Training...")
loss=float("inf")
for batchCount in range(0, trainBatchNumber):
  print("Batch {} started processing...".format(batchCount))
  previousLoss=loss
  oldWeights=weights
  batchLoss=0
  for exampleCount in range(0, trainBatchSize):
    # Good Example
    print("Batch #{} Example #{}".format(batchCount,exampleCount))
    digit=str(random.randrange(0,10))
    size=random.randrange(0,6)
    example=""
    for i in range(0,size):
      example+=digit
    thought=activeFunc(think(example, structure, weights))
    weights=wAdjustFunc(loss, previousLoss, weights)   
    loss=lossFunc(thought, expectedGoodOutcome)
    batchLoss+=loss
    # Bad Example
    print("Batch #{} Example #{}".format(batchCount,exampleCount))
    size=random.randrange(0,6)
    example=""
    for i in range(0,size):
      digit=str(random.randrange(0,10))
      example+=digit
    thought=activeFunc(think(example, structure, weights))
    loss=lossFunc(thought, expectedBadOutcome)
    batchLoss+=loss
  print("Batch {} finished processing. Loss: {} -- Weights: {}".format(batchCount, loss, weights))
  if [ loss > previousLoss ]:
    loss=previousLoss
    weights=oldWeights
    print("Reverted back to old weights as they provided better loss. Loss: {} -- Weights: {}".format(loss, weights))
  if [ batchCount+1 != trainBatchNumber  ]:
    print("Adjusting weights for upcoming batch {}".format(batchCount+1))
    weights=wAdjustFunc(weights)

# Evaluation
thought=activeFunc(think(input, structure, weights))
print(thought)
