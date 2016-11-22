% Reset Workspace and Command Window
close all;
clear all;
clc;

data = importdata('Madrid_Speed_Direction_SurfaceAirTemp.xlsx');   % Import data
dataSize = max(size(data(:,1)));                                   % Number of rows in data

% Mean & Std calculated for normalization/denormalization calculations
speedMEAN = mean(data(:,1));
speedSTD = std(data(:,1));
dirMEAN = mean(data(:,2));
dirSTD = std(data(:,2));
satMEAN = mean(data(:,3));
satSTD = std(data(:,3));

% Zero mean, unit variance normalization
normalized_speed = (((data(:,1))-speedMEAN)/speedSTD);
normalized_dir = (((data(:,2))-dirMEAN)/dirSTD);
normalized_sat = (((data(:,3))-satMEAN)/satSTD);

trainingSamples = 5000;     % Number of data points to train (250 days)
testingSamples = 2784;      % Number of data points to predict (116 days)
numOfInputs = 12;            % Number of network inputs
testingPoint = dataSize - testingSamples - (numOfInputs/3);     % Point to start prediction from

% Creation of matrix for the network to train and predict
netData = zeros(dataSize(1)-(numOfInputs/3),numOfInputs+1);
for i = 1:(dataSize(1)-(numOfInputs/3))  
    netData(i,1:(numOfInputs/3)) = normalized_speed(i:(i+((numOfInputs/3)-1)))';
    netData(i,((numOfInputs/3)+1):((numOfInputs/3)*2)) = normalized_dir(i:(i+((numOfInputs/3)-1)))';
    netData(i,(((numOfInputs/3)*2)+1):numOfInputs) = normalized_sat(i:(i+((numOfInputs/3)-1)))';
    
    netData(i,(numOfInputs+1)) = normalized_speed(i+((numOfInputs/3)));
end

% Creation of training matrices
P = netData(1:trainingSamples-(numOfInputs/3),1:numOfInputs);
T = netData(1:trainingSamples-(numOfInputs/3),numOfInputs+1);

clearvars data normalized_speed normalized_sat;  % To free memory of variables no longer needed

net = newrb(P',T',0,25,50);

% Declaration of matrices to calculate training forecast errors
training_actualOutput = zeros(trainingSamples-(numOfInputs/3),1);
training_expectedOutput = zeros(trainingSamples-(numOfInputs/3),1);
trainingError = zeros(trainingSamples-(numOfInputs/3),1);

for j = 1:trainingSamples-(numOfInputs/3)
training_actualOutput(j,1) = sim(net,netData(j,1:numOfInputs)');    % Use network to predict training data
training_expectedOutput(j,1) = netData(j,numOfInputs+1);            % Expected outputs for comparison

% Denormalization of expected and predicted values
training_actualOutput(j,1) = (training_actualOutput(j,1)*speedSTD)+speedMEAN;
training_expectedOutput(j,1) = (training_expectedOutput(j,1)*speedSTD)+speedMEAN;

trainingError(j,1) = abs(training_actualOutput(j,1) - training_expectedOutput(j,1));
end


% To display figure on the left half of the screen
screen_size = get(0, 'ScreenSize');    %To obtain the screen resolution
set(figure('name','Time Series Performance of Training Time Series'), 'Position', [0 0 screen_size(3)/2 screen_size(4)] );    % Make use of screen width and height

axes1 = axes('FontSize',16,'FontName','Verdana');
box(axes1,'on');
hold(axes1,'all');

% To plot training expected and predicted time series
plot(training_expectedOutput(1:trainingSamples - (numOfInputs/3)),'r-')
hold on;
plot(training_actualOutput(1:trainingSamples - (numOfInputs/3)),'b')

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
testing_actualOutput = zeros(testingSamples-(numOfInputs/3),1);
testing_expectedOutput = zeros(testingSamples-(numOfInputs/3),1);
testingError = zeros(testingSamples-(numOfInputs/3),1);

for k = 1:testingSamples-(numOfInputs/3) 
testing_actualOutput(k,1) = sim(net,netData(k+testingPoint,1:numOfInputs)');    % Use network to predict testing data
testing_expectedOutput(k,1) = netData(k+testingPoint,numOfInputs+1);            % Expected outputs for comparison

% Denormalization of expected and predicted values
testing_actualOutput(k,1) = (testing_actualOutput(k,1)*speedSTD)+speedMEAN;
testing_expectedOutput(k,1) = (testing_expectedOutput(k,1)*speedSTD)+speedMEAN;

testingError(k,1) = abs(testing_actualOutput(k,1) - testing_expectedOutput(k,1));
end

% To display figure on the right half of the screen
screen_size = get(0, 'ScreenSize');    %To obtain the screen resolution
set(figure('name','Time Series Performance of Training Time Series'), 'Position', [screen_size(3)/2 0 screen_size(3)/2 screen_size(4)] );  % Make use of screen width and height

axes1 = axes('FontSize',16,'FontName','Verdana');
box(axes1,'on');
hold(axes1,'all');

% To plot training expected and predicted time series
plot(testing_expectedOutput(1:testingSamples - (numOfInputs/3)),'r-')
hold on;
plot(testing_actualOutput(1:testingSamples - (numOfInputs/3)),'b')

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