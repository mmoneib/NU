#!/usr/env/python
import argparse
import math
import random

# Internal Utility Functions
def __get_functions_ending_with(postfix):
  choices=[]
  for f in dir():
    if callable(f) and f.endswith(postfix):
      choices+=f
  return choices

# Activation Functions
def relU_active_func(neuronOutput):
  if neuronOutput <= 0:
    return 0
  else:
    return neuronOutput*(math.floor(neuronOutput)+1)
def tanh_active_func(neuronOutput):
  return math.tanh(neuronOutput)

# Loss Functions
def modulus_loss_func(actual, expected, accumulatedLoss=0):
  return accumulatedLoss+abs(actual-expected)

# Weight Adjust Functions
def apply_weights_adjustment(loss, comparableLoss, weights):
  wAdjustFunc(weights)
def buzz_wadjust_func(weights):
  return weight+random.choice([-0.01,0,0.01])
def random_guess_wadjust_func(weights):
  return random.randrange(-100,101)/100

# Thinking
def think(inputToProcess, neuronsStructure, weights):
  neuronOutputs=[]
  numOfInputNeurons=neuronsStructure[0]
  inputsToLayer=[]
  inputDivisor=0
  for i in range(0, numOfInputNeurons):
    inputsToLayer+=inputToProcess[inputDivisor:numOfInputNeurons+inputDivisor]
    inputDivisor+=numofInputNeurons
  weightsIndex=0
  for layerIndex in range(1,len(neuronsStructure)):
    for neuronInLayerIndex in range(0, neuronsStructure[layerIndex]):
      neuronOutput=0
      bias=neuronInLayerIndex+layerIndex+1
      neuronOutput+=bias
      for inputsTolayerIndex in range(0, len(inputsToLayer)):
        neuronOutput+=inputsToLayer[inputsToLayerIndex]*weights[weightsIndex]
        weightsIndex+=1
      neuronOutputs+=neuronOutput
    inputsToLayer=neuronOutputs[len(neuronOutputs)-neuronStructure[layerIndex]]
  return neurounOutputs[len(neuronOutputs)-neuronStructure[len(neuronStructure)-1]:]    

# Input
parser=argparse.ArgumentParser(prog="NU", description="Neural Net Implementation Using Python")
parser.add_argument("input", type=int, required=True, help="Input to be processed.")
parser.add_argument("-a", "--activation", choices=__get_functions_ending_with("_activation"), required=True, help="Name of activation function.")
parser.add_argument("-s", "--structure", nargs="*", required=True, help="Structure of the newural network as a list of number of nodes per layer.")
parser.add_argument("-v", "--verbose", action="store_true", help="Display debugging statements and detailed output.")
parser.add_argument("-w", "--weights", nargs="*", help="List of weights, one for each link in the fully connected network. The order follows that of the neurons inputs, with each weight being an output of a neuron from the previous layer.")
trainingGroup=parser.add_argument_group("Training")
trainingGroup.add_argument("-b", "--batches", help="Training batches specification as a product: number_of_batches*number_of_examples_per_batch.")
trainingGroup.add_argument("-j", "--weight-adjust", required=True, help="Name of weights adjustment function.")
trainingGroup.add_argument("-l", "--loss", required=True, help="Name of loss function.")
args.parse_args()
print(args.input)
print(args.activation)
print(args,weight_adjust)
print(args.loss)
print(args.structure)
print(args.training)
print(args.weights)
input=args.input
neuronsStructure=args.structure
weights=args.weights
trainBatchNumber=0
trainBatchSize=0
numOfTrainBatches=0
activeFunc=args.activation
lossFunc=args.loss
wAdjustFunc=args.weight_adjust
expectedGoodOutcome=2
expectedBadOutcome=0

# Initialization and Validation
numOfWeights=0
for n in range(1,len(neuronsStructure)):
  numofWeights=numOfWeights+neuronsStructure[n]*neuronsStructure[n-1]
if len(weights) == 0:
  for i in range(0,numOfWeights):
    weighs+=random.randrange(-100,101)/100
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
    thought=activeFunc(think(example))
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
    thought=activeFunc(think(example))
    loss=lossFunc(thought, expectedBadOutcome)
    batchLoss+=loss
  print("Batch {} finished processing. Loss: {} -- Weights: {}".format(, batchCount, loss, weights))
  if [ loss > previousLoss ]:
    loss=previousLoss
    weights=oldWeights
    print("Reverted back to old weights as they provided better loss. Loss: {} -- Weights: {}".format(loss, weights))
  if [ batchCount+1 != trainBatchNumber  ]:
    print("Adjusting weights for upcoming batch {}".format(batchCount+1))
    weights=wAdjustFunc(weights)

# Evaluation
thought=activeFunc(think(input))
print(thought)
