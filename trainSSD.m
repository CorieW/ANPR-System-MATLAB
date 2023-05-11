function detector = trainSSD(trainData, valData)
    %% Data pre-processing
    % Load the training data
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
    
    %% implement model
    % Generate anchor boxes
    anchorBoxes = {[30 60; 60 30; 50 50; 100 100], ...
                   [40 70; 70 40; 60 60; 120 120]};
    
    % Define the classes
    classes = {'licensePlate'};
    
    % Define the layers
    layersToConnect =  ["activation_22_relu" "activation_40_relu"];
    
    % Define the detector
    detector = ssdObjectDetector(layerGraph(resnet50),classes,anchorBoxes,DetectionNetworkSource=layersToConnect);
    
    % Specify the training options
    options = trainingOptions('adam', ...
        InitialLearnRate=0.001, ...
        LearnRateDropFactor=0.1, ...
        LearnRateSchedule='piecewise',...
        LearnRateDropPeriod=5,...
        Plots='training-progress',...
        MiniBatchSize=8,...
        MaxEpochs=100, ...
        BatchNormalizationStatistics="moving",...
        ResetInputNormalization=false,...
        ValidationData=valDS,...
        VerboseFrequency=1);
    
    detector = trainSSDObjectDetector(trainDS,detector,options);
end