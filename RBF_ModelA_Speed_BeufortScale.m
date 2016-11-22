% Reset Workspace and Command Window
close all;
clear all;
clc;

data = importdata('Madrid_Speed.xlsx');    % Import data
dataSize = max(size(data));                % Number of rows in data

% Declaration of matrix to hold beufort scale values 
beufort = zeros(dataSize,1);

% Creation of matrix to assign beufort values to data
for i = 1:size(data)
    
    if (data(i) < 1)
        force = 0;
    elseif (data(i) >= 1 && data(i) <= 3)
        force = 1;
    elseif (data(i) >= 4 && data(i) <= 6)
        force = 2;
    elseif (data(i) >= 7 && data(i) <= 10)
        force = 3;
    elseif (data(i) >= 11 && data(i) <= 16)
        force = 4;
    elseif (data(i) >= 17 && data(i) <= 21)
        force = 5;
    elseif (data(i) >= 22 && data(i) <= 27)
        force = 6;
    elseif (data(i) >= 28 && data(i) <= 33)
        force = 7;
    elseif (data(i) >= 34 && data(i) <= 40)
        force = 8;
    elseif (data(i) >= 41 && data(i) <= 47)
        force = 9;
    elseif (data(i) >= 48 && data(i) <= 55)
        force = 10;
    elseif (data(i) >= 56 && data(i) <= 63)
        force = 11;
    else
        force = 12;
    end
    
    beufort(i) = force; 
end

trainingSamples = 5000;     % Number of data points to train (250 days)
testingSamples = 2784;      % Number of data points to predict (116 days)
numOfInputs = 4;            % Number of network inputs
testingPoint = dataSize - testingSamples - numOfInputs;   % Point to start prediction from

% Creation of matrix for the network to train and predict
netData = zeros(dataSize-numOfInputs,numOfInputs+1);
for i = 1:dataSize-numOfInputs
    netData(i,:) = beufort(i:(i+numOfInputs))';
end

% Creation of training matrices
P = netData(1:trainingSamples-numOfInputs,1:numOfInputs);   % Input samples
T = netData(1:trainingSamples-numOfInputs,numOfInputs+1);   % Expected targets

clearvars normalized_data;      % To free memory of variables no longer needed

net = newrb(P',T',0,10,15,1);     % newrb(P,T,GOAL,SPREAD,MN,DF)

% Declaration of matrices to calculate training forecast errors
training_actualOutput = zeros(trainingSamples-numOfInputs,1);
training_expectedOutput = zeros(trainingSamples-numOfInputs,1);
trainingError = zeros(trainingSamples-numOfInputs,1);

for j = 1:trainingSamples-numOfInputs
training_actualOutput(j,1) = sim(net,netData(j,1:numOfInputs)');    % Use network to predict training data
training_expectedOutput(j,1) = netData(j,numOfInputs+1);            % Expected outputs for comparison

% Rounding off
training_actualOutput(j,1) = round(training_actualOutput(j,1));

trainingError(j,1) = abs(training_actualOutput(j,1) - training_expectedOutput(j,1));
end

% To display figure on the left half of the screen
screen_size = get(0, 'ScreenSize');    %To obtain the screen resolution
set(figure('name','Time Series Performance of Training Time Series'), 'Position', [0 0 screen_size(3)/2 screen_size(4)] );    % Make use of screen width and height

axes1 = axes('FontSize',16,'FontName','Verdana');
box(axes1,'on');
hold(axes1,'all');

% To plot training expected and predicted time series
plot(training_expectedOutput(1:trainingSamples-numOfInputs),'r-')
hold on;
plot(training_actualOutput(1:trainingSamples-numOfInputs),'b')

xlim([0 trainingSamples]);
xlabel('Time (Hours)','FontWeight','bold','FontSize',16,...
    'FontName','Verdana');
ylabel('Wind Speed Forecast (Knots)','FontWeight','bold','FontSize',16,...
    'FontName','Verdana');
title('Time Series Performance of Training Time Series','FontWeight','bold',...
    'FontSize',18,...
    'FontName','Verdana');
legend('Expected', 'Predicted');

% Calculate and display training performance paramters
trainingRMS = sqrt((sum(trainingError.^2))/trainingSamples)
trainingMAE = mae(trainingError,training_actualOutput)
trainingMAPE = (sum(trainingError./training_actualOutput))/trainingSamples

% Declaration of matrices to calculate testing forecast errors
testing_actualOutput = zeros(testingSamples-numOfInputs,1);
testing_expectedOutput = zeros(testingSamples-numOfInputs,1);
testingError = zeros(testingSamples-numOfInputs,1);

% Variables for classification
correct = 0;
incorrect = 0;
incorrectByOne = 0;     % Incorrect cases differing by 1

for k = 1:testingSamples-numOfInputs 
testing_actualOutput(k,1) = sim(net,netData(k+testingPoint,1:numOfInputs)');    % Use network to predict testing data
testing_expectedOutput(k,1) = netData(k+testingPoint,numOfInputs+1);            % Expected outputs for comparison

%Round off
testing_actualOutput(k,1) = round(testing_actualOutput(k,1));

testingError(k,1) = abs(testing_actualOutput(k,1) - testing_expectedOutput(k,1));

    if (testing_expectedOutput(k,1) == testing_actualOutput(k,1))
        correct = correct + 1;
    else
    	incorrect = incorrect + 1;
        if (abs(testing_expectedOutput(k,1) - testing_actualOutput(k,1)) <= 1)
           incorrectByOne = incorrectByOne+1;  
        end
    end

end

% To display figure on the right half of the screen
screen_size = get(0, 'ScreenSize');    %To obtain the screen resolution
set(figure('name','Time Series Performance of Training Time Series'), 'Position', [screen_size(3)/2 0 screen_size(3)/2 screen_size(4)] );  % Make use of screen width and height

axes1 = axes('FontSize',16,'FontName','Verdana');
box(axes1,'on');
hold(axes1,'all');

% To plot testing expected and predicted time series
plot(testing_expectedOutput(1:testingSamples-numOfInputs),'r-')
hold on;
plot(testing_actualOutput(1:testingSamples-numOfInputs),'b')

xlim([0 testingSamples]);
xlabel('Time (Hours)','FontWeight','bold','FontSize',16,...
    'FontName','Verdana');
ylabel('Wind Speed Forecast (Knots)','FontWeight','bold','FontSize',16,...
    'FontName','Verdana');
title('Time Series Performance of Testing Time Series','FontWeight','bold',...
    'FontSize',18,...
    'FontName','Verdana');
legend('Expected', 'Predicted');

% Calculate and display training performance paramters
testingRMS = sqrt((sum(testingError.^2))/testingSamples)
testingMAE = mae(testingError,testing_actualOutput)
testingMAPE = (sum(testingError./testing_actualOutput))/testingSamples

% Display Classifications
CorrectClassifications = (correct/(testingSamples - numOfInputs))*100
IncorrectByOnlyFactorOfOne = (incorrectByOne/incorrect)*100