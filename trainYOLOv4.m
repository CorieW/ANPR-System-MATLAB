function detector = trainYOLOv4(trainData, valData)
    %% Data pre-processing
    % store the training data in datastores
    imds = imageDatastore(trainData.imageFilename);
    blds = boxLabelDatastore(trainData(:,3));
    % combine the image and label data into a single datastore of the training
    % stores
    trainDS = combine(imds, blds);
    
    % store the validation data in datastores
    imds = imageDatastore(valData.imageFilename);
    blds = boxLabelDatastore(valData(:,3));
    % combine the image and label data into a single datastore of the
    % validation stores
    valDS = combine(imds, blds);
    
    % Configure
    inputSize = [224 224 3];
    
    % Rescale the images to fit
    transformedTrainedDS = transform(trainDS,@(data)preprocessData(data,inputSize));
    transformedValidationDS = transform(valDS,@(data)preprocessData(data,inputSize));
    
    %% implement model
    % Generate anchor boxes based on the training data
    numAnchors = 6;
    [anchors] = estimateAnchorBoxes(transformedTrainedDS,numAnchors);
    area = anchors(:,1).*anchors(:,2);
    [~,idx] = sort(area,"descend");
    anchors = anchors(idx,:);
    anchorBoxes = {anchors(1:3,:);anchors(4:6,:)};
    
    % Define the classes
    classes = {'licensePlate'};
    
    % Define the detector
    detector = yolov4ObjectDetector('tiny-yolov4-coco',classes,anchorBoxes,InputSize=inputSize);
    
    % Specify the training options
    options = trainingOptions('adam', ...
        InitialLearnRate=0.001, ...
        LearnRateDropFactor=0.1, ...
        LearnRateSchedule='piecewise',...
        LearnRateDropPeriod=5,...
        Plots='training-progress',...
        MiniBatchSize=32,...
        MaxEpochs=100, ...
        BatchNormalizationStatistics="moving",...
        ResetInputNormalization=false,...
        ValidationData=transformedValidationDS,...
        VerboseFrequency=1);
    
    detector = trainYOLOv4ObjectDetector(trainDS,detector,options);
    
    function data = preprocessData(data,targetSize)
        for num = 1:size(data,1)
            I = data{num,1};
            imgSize = size(I);
            bboxes = data{num,2};
            I = im2single(imresize(I,targetSize(1:2)));
            scale = targetSize(1:2)./imgSize(1:2);
            bboxes = bboxresize(bboxes,scale);
            data(num,1:2) = {I,bboxes};
        end
    end
end