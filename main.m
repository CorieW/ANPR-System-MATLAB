clear; clc;

%% Image aquisition
trainRatio = 0.6;
valRatio = 0.2;
testRatio = 0.2;
% parse the entire dataset into a table
data = parseDataset();
% randomize the parsed dataset into three randomized datasets for the
% training, validation, and test sets
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
% yolov4
% display times taken
disp('yolov4');
disp(['fps: ' num2str(licensePlateDetectionTime1)]);
disp(['total images: ' num2str(size(testData.imageFilename, 1))]);
disp(['elapsed: ' num2str(toc)]);
% find average successful plate detection
totSuccess = 0;
for i = 1:size(myResultsMap1)
    predicted = myResultsMap1{i,3};
    ground = myResultsMap1{i,4};
    if strcmp(predicted, ground)
        totSuccess = totSuccess + 1;
    end
end
disp(['total successful recognitions: ' num2str(totSuccess)]);
% evaluate predicted bounding boxes against ground truth
imds = imageDatastore(testData.imageFilename);
blds = boxLabelDatastore(testData(:, 3));
ds = combine(imds, blds);
results = detect(yolov4ObjDetector,ds,'MiniBatchSize',8);
[ap,recall,precision] = evaluateDetectionPrecision(results,ds);
disp("ap: " + num2str(ap));
disp("detected: " + num2str(size(precision, 1)));
% calculate false-positives
falsePositives = 0;
for i = 1:size(precision, 1)
    if (precision(i) < 0.8)
        falsePositives = falsePositives + 1;
    end
end
disp("detections false positives: " + num2str(falsePositives));

% ssd
% display times taken
disp('ssd');
disp(['fps: ' num2str(licensePlateDetectionTime2)]);
disp(['total images: ' num2str(size(testData.imageFilename, 1))]);
disp(['elapsed: ' num2str(toc)]);
totSuccess = 0;
% find average successful plate detection
for i = 1:size(myResultsMap2)
    predicted = myResultsMap2{i,3};
    ground = myResultsMap2{i,4};
    if strcmp(predicted, ground)
        totSuccess = totSuccess + 1;
    end
end
disp(['total successful recognitions: ' num2str(totSuccess)]);
% evaluate predicted bounding boxes against ground truth
imds = imageDatastore(testData.imageFilename);
blds = boxLabelDatastore(testData(:, 3));
ds = combine(imds, blds);
results = detect(ssdObjDetector,ds,'MiniBatchSize',8);
[ap,recall,precision] = evaluateDetectionPrecision(results,ds);
disp("ap: " + num2str(ap));
disp("detected: " + num2str(size(precision, 1)));
% calculate false-positives
falsePositives = 0;
for i = 1:size(precision, 1)
    if (precision(i) < 0.8)
        falsePositives = falsePositives + 1;
    end
end
disp("detections false positives: " + num2str(falsePositives));