clear; clc;

%% Image aquisition
trainRatio = 0.6;
valRatio = 0.2;
testRatio = 0.2;
% parse the entire dataset into a table
data = parseDataset();
% randomize the rows of the table
tableRows = size(data);
tableRows = tableRows(1);
randIndices = randperm(size(data, 1));
randomizedTable = data(randIndices, :);
data = randomizedTable;
% split into different sets for: training, validation, and testing
totalRows = height(data);
% train
trainRows = round(totalRows * trainRatio);
trainData = data(1:trainRows,:);
% val
valRows = round(totalRows*valRatio);
valData = data(trainRows:valRows+trainRows,:);
% test
testRows = round(totalRows*testRatio);
testData = data(trainRows+valRows:trainRows+valRows+testRows,:);

%% Training
% YOLOv4 model
if exist('yolov4ObjDetector.mat', 'file') == 2
    yolov4ObjDetector = load('yolov4ObjDetector.mat').yolov4ObjDetector;
else
    % No object detector exists, so create one
    yolov4ObjDetector = trainSSD(trainData, valData);
    save('yolov4ObjDetector.mat', 'yolov4ObjDetector');
end

% SSD model
if exist('ssdObjDetector.mat', 'file') == 2
    ssdObjDetector = load('ssdObjDetector.mat').ssdObjDetector;
else
    % No object detector exists, so create one
    ssdObjDetector = trainSSD(trainData, valData);
    save('ssdObjDetector.mat', 'ssdObjDetector');
end

%% Testing
% store the testing data
imds = imageDatastore(testData.imageFilename);
blds = boxLabelDatastore(testData(:,3));
% combine into single datastore
ds = combine(imds, blds);

% test
[licensePlateDetectionTime1, myResultsMap1] = test(yolov4ObjDetector, testData, '/tmp_cropped_plates_yolo');
[licensePlateDetectionTime2, myResultsMap2] = test(ssdObjDetector, testData, '/tmp_cropped_plates_ssd');

%% Analysis
logAnalysis("YOLOv4", yolov4ObjDetector, testData, licensePlateDetectionTime1, myResultsMap1);
logAnalysis("SSD", ssdObjDetector, testData, licensePlateDetectionTime2, myResultsMap2);