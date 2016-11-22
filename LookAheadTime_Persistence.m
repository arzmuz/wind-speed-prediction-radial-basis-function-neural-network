% Reset Workspace and Command Window
close all;
clear all;
clc;

data = importdata('Madrid_Speed.xlsx');   % Import data

testingPoint = 6000;        % Number of data points to train (250 days)
testingSamples = 2760;      % Number of data points to predict (115 days)

iterations = 10;          

RMSvals = zeros(iterations,1);
MAEvals = zeros(iterations,1);

for j = 1 : iterations
    
    lookAheadTime = j;      % Prediction of the nth hour
    
    % Target or actual wind speed data for prediction set 
    expected = data(testingPoint+lookAheadTime : testingPoint+testingSamples);

    % Predicted wind speed data using persistence theorem
    prediction = data(testingPoint : testingPoint+testingSamples-lookAheadTime);

    testingError = zeros(testingSamples-lookAheadTime+1,1);

    for k = 1:max(size(prediction))
    testingError(k,1) = abs(prediction(k,1) - expected(k,1));
    end

    RMSvals(j) = sqrt((sum(testingError.^2))/testingSamples);
    MAEvals(j) = mae(testingError,prediction);
    
end

% To display figure on the left half of the screen
screen_size = get(0, 'ScreenSize');    %To obtain the screen resolution
set(figure('name','Persistence Root Mean Square Error Trend'), 'Position', [0 0 screen_size(3)/2 screen_size(4)] );    % Make use of screen width and height

axes1 = axes('FontSize',16,'FontName','Verdana');
box(axes1,'on');
hold(axes1,'all');

plot(RMSvals);
xlim([0 iterations]);
ylim([0 max(RMSvals)]);
xlabel('Look Ahead Time (Hours)','FontWeight','bold','FontSize',16,...
    'FontName','Verdana');
ylabel('Root Mean Square Error','FontWeight','bold','FontSize',16,...
    'FontName','Verdana');
title('Persistence Root Mean Square Error Trend','FontWeight','bold',...
    'FontSize',18,...
    'FontName','Verdana');

% To display figure on the right half of the screen
screen_size = get(0, 'ScreenSize');    %To obtain the screen resolution
set(figure('name','Persistence Mean Absolute Error Trend'), 'Position', [screen_size(3)/2 0 screen_size(3)/2 screen_size(4)] );  % Make use of screen width and height

axes1 = axes('FontSize',16,'FontName','Verdana');
box(axes1,'on');
hold(axes1,'all');

plot(MAEvals);
xlim([0 iterations]);
ylim([0 max(MAEvals)]);
xlabel('Look Ahead Time (Hours)','FontWeight','bold','FontSize',16,...
    'FontName','Verdana');
ylabel('Mean Absolute Error','FontWeight','bold','FontSize',16,...
    'FontName','Verdana');
title('Persistence Mean Absolute Error Trend','FontWeight','bold',...
    'FontSize',18,...
    'FontName','Verdana');