#!/usr/env/python
import argparse
import math
import random

# Functions
## Internal Utility Functions
def __get_functions_ending_with(postfix, directory):
  choices=[]
  for f in directory:
    func=eval(f) # Convert the function name to the function itself. Doesn't really invoke the function.
    if f.endswith(postfix) and callable(func):
      choices.append(f.replace(postfix,""))
  return choices
def __parse_comma_separated_list(csl):
  return csl.split(",")
def __print_comma_separated_list(csl):
  cslStr=""
  for wi in csl: cslStr+=str(wi)+","
  return cslStr
## Activation Functions
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
## Loss Functions
def modulus_loss_func(actual, expected, accumulatedLoss=0):
  for i in range(0, len(actual)):
    accumulatedLoss=accumulatedLoss+abs(actual[i]-expected[i]) #TODO: Validate last layer has same size as expected.
  return accumulatedLoss
## Weight Adjust Functions
def buzz_wadjust_func(weights):
  for i in range(0, len(weights)):
    weights[i]=float(weights[i])+random.choice([-0.01,0,0.01])
  return weights
def random_guess_wadjust_func(weights):
  for i in range(0, len(weights)):
    weights[i]=random.randrange(-100,101)/100
  return weights
## Thinking
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
        neuronOutput+=inputsToLayer[inputsToLayerIndex]*float(weights[weightsIndex])
        weightsIndex+=1
      neuronOutputs.append(neuronOutput)
    inputsToLayer=neuronOutputs[len(neuronOutputs)-int(neuronsStructure[layerIndex]):]
  return neuronOutputs[len(neuronOutputs)-int(neuronsStructure[len(neuronsStructure)-1]):]    

# Configuration
## Input Retrieval
parser=argparse.ArgumentParser(prog="NU", description="Neural Net Implementation Using Python")
parser.add_argument("input", type=int, help="Input to be processed.")
parser.add_argument("-a", "--activation", choices=__get_functions_ending_with("_active_func", dir()), required=True, help="Name of activation function.")
parser.add_argument("-s", "--structure", type=__parse_comma_separated_list, required=True, help="Structure of the newural network as a list of number of nodes per layer.")
parser.add_argument("-v", "--verbose", action="store_true", help="Display debugging statements and detailed output.")
parser.add_argument("-w", "--weights", type=__parse_comma_separated_list, help="List of weights, one for each link in the fully connected network. The order follows that of the neurons inputs, with each weight being an output of a neuron from the previous layer.")
trainingGroup=parser.add_argument_group("Training")
trainingGroup.add_argument("-b", "--batches", help="Training batches specification as a product: number_of_batches*number_of_examples_per_batch.")
trainingGroup.add_argument("-j", "--weight-adjust", choices=__get_functions_ending_with("_wadjust_func", dir()), required=True, help="Name of weights adjustment function.")
trainingGroup.add_argument("-l", "--loss", required=True, choices=__get_functions_ending_with("_loss_func", dir()), help="Name of loss function.")
args=parser.parse_args()
## Internal Initialization
input=args.input
structure=args.structure
weights=args.weights
trainBatchNumber=int(args.batches.split("*")[0]) # TODO: Strict formatting of batches parameter using argparse.
trainBatchSize=int(args.batches.split("*")[1])
numOfTrainBatches=0
activeFunc=eval(args.activation+"_active_func")
lossFunc=eval(args.loss+"_loss_func")
wAdjustFunc=eval(args.weight_adjust+"_wadjust_func")
expectedGoodOutcome=[2]
expectedBadOutcome=[0]
numOfWeights=0
for n in range(1,len(structure)):
  numOfWeights=numOfWeights+int(structure[n])*int(structure[n-1])
## Printing
if args.verbose: 
  print("Configuration loaded...")
  print("Input Parameters: {}".format(args))
  print("Expected Number of Weights: {}".format(numOfWeights))
  print("Expected Good Outcome: {}".format(expectedGoodOutcome))
  print("Expected Bad Outcome: {}".format(expectedBadOutcome))
  print("Number of Training Batches: {}".format(trainBatchNumber))
  print("Size of Training Batch: {}".format(trainBatchSize))
# Internal Validation
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
  #previousLoss=loss
  batchLoss=0
  exampleCount=0
  while exampleCount < trainBatchSize:
    # Good Example
    digit=str(random.randrange(0,10))
    size=random.randrange(1,6)
    example=""
    for i in range(0,size):
      example+=digit
    if args.verbose: print("Batch #{} Example #{} -- Good Example: {}".format(batchCount,exampleCount,example))
    thought=activeFunc(think(example, structure, weights))
    exampleLoss=lossFunc(thought, expectedGoodOutcome)
    batchLoss+=exampleLoss
    exampleCount=exampleCount+1
    # Bad Example
    size=random.randrange(2,6) # No singles.
    example=""
    repCharsCount=1
    firstDigit=str(random.randrange(0,10))
    example+=firstDigit
    for i in range(1,size):
      digit=str(random.randrange(0,10))
      if digit==firstDigit: repCharsCount=repCharsCount+1
      example+=digit
    if repCharsCount==len(example):
      charIndex=random.randrange(0,len(example))
      if charIndex+1 < len(example):
        example=example[0:charIndex]+str((int(example[charIndex])+9)%10)+example[charIndex+1]
      else:
        example=example[0:charIndex]+str((int(example[charIndex])+9)%10)
    if args.verbose: print("Batch #{} Example #{} -- Bad Example: {}".format(batchCount,exampleCount,example))
    thought=activeFunc(think(example, structure, weights))
    exampleLoss=lossFunc(thought, expectedBadOutcome)
    batchLoss+=exampleLoss
    exampleCount=exampleCount+1
  # Same weghts can produce different losses across different batches, but they should be close.
  print("Batch {} finished processing. Loss: {} -- Weights: {}".format(batchCount, batchLoss, __print_comma_separated_list(weights)))
  if batchLoss > loss :
    #loss=previousLoss
    weights=oldWeights.copy() # Backtracking makes sense with stochastic weights choices.
    print("No loss improvement. Reverted back to old weights. Loss: {} -- Weights: {}".format(loss, __print_comma_separated_list(weights)))
  else:
    loss=batchLoss
  if batchCount+1 != trainBatchNumber: # Skip adjusting for last layer.
    print("Adjusting weights for upcoming batch {}".format(batchCount+1))
    oldWeights=weights.copy()
    weights=wAdjustFunc(weights)

# Evaluation
thought=activeFunc(think(input, structure, weights))
print(thought)
