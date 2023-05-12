function detector = trainSSD(trainData, valData)
    %% Data pre-processing
    % load the training data into data stores
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

    % input size of the model
    inputSize = [512 512 3];
    
    % rescale the images to 
    transformedTrainedDS = transform(trainDS,@(trainData)preprocessData(trainData,inputSize));
    transformedValidationDS = transform(valDS,@(valData)preprocessData(valData,inputSize));

    %% implement model
    % generate anchor boxes based on training data
    numAnchors = 6;
    [anchors] = estimateAnchorBoxes(transformedTrainedDS,numAnchors);
    area = anchors(:,1).*anchors(:,2);
    [~,idx] = sort(area,"descend");
    anchors = anchors(idx,:);
    anchorBoxes = {anchors(1:3,:);anchors(4:6,:)};
    
    % define the classes
    classes = {'licensePlate'};
    
    % define the layers
    layersToConnect =  ["activation_22_relu" "activation_40_relu"];
    
    % define the detector
    detector = ssdObjectDetector(layerGraph(resnet50),classes,anchorBoxes,DetectionNetworkSource=layersToConnect);
    
    % define training options
    options = trainingOptions('adam', ...
        InitialLearnRate=0.001, ...
        LearnRateDropFactor=0.001, ...
        LearnRateSchedule='piecewise',...
        LearnRateDropPeriod=5,...
        MiniBatchSize=16,...
        MaxEpochs=50,...
        BatchNormalizationStatistics="moving",...
        ResetInputNormalization=false,...
        ValidationData=valDS,...
        ValidationFrequency=30,...
        VerboseFrequency=1);
    
    detector = trainSSDObjectDetector(trainDS,detector,options);

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