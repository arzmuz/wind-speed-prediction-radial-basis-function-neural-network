% Reset Workspace and Command Window
close all;
clear all;
clc;

data = importdata('Madrid_Speed.xlsx');    % Import data
dataSize = max(size(data));                % Number of rows in data

lookAheadTime = 9;
t = lookAheadTime - 1;

trainingSamples = 6000;     % Number of data points to train (250 days)
testingSamples = 2760;      % Number of data points to predict (116 days)
numOfInputs = 4;            % Number of network inputs
testingPoint = dataSize - testingSamples - numOfInputs;   % Point to start prediction from

normalized_data = (data - mean(data))/std(data);    % Zero mean, unit variance normalization

% Creation of matrix for the network to train and predict
netData = zeros(dataSize-numOfInputs,numOfInputs+1);
for i = 1:dataSize-numOfInputs
    netData(i,:) = normalized_data(i:(i+numOfInputs))';
end

% Creation of training matrices
P = netData(1:trainingSamples-numOfInputs,1:numOfInputs);   % Input samples
T = netData(1:trainingSamples-numOfInputs,numOfInputs+1);   % Expected targets

P = P(1:end-t,:);
T = T(t+1:end);

clearvars normalized_data;      % To free memory of variables no longer needed

net = newrb(P',T',0,10,30);     % newrb(P,T,GOAL,SPREAD,MN,DF)

% Declaration of matrices to calculate testing forecast errors
testing_actualOutput = zeros(testingSamples-numOfInputs,1);
testing_expectedOutput = zeros(testingSamples-numOfInputs,1);
testingError = zeros(testingSamples-numOfInputs,1);

for k = 1:testingSamples-numOfInputs 
testing_actualOutput(k,1) = sim(net,netData(k+testingPoint,1:numOfInputs)');    % Use network to predict testing data
testing_expectedOutput(k,1) = netData(k+testingPoint,numOfInputs+1);            % Expected outputs for comparison

% Denormalization of expected and predicted values
testing_actualOutput(k,1) = (testing_actualOutput(k,1)*std(data))+mean(data);
testing_expectedOutput(k,1) = (testing_expectedOutput(k,1)*std(data))+mean(data);

testingError(k,1) = abs(testing_actualOutput(k,1) - testing_expectedOutput(k,1));
end

% Calculate and display training performance paramters
testingRMS = sqrt((sum(testingError.^2))/testingSamples)
testingMAE = mae(testingError,testing_actualOutput)