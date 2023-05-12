function output = logAnalysis(modelName,objDetector,data,licensePlateDetectionTime, myResultsMap)
    % display times taken
    disp(modelName);
    disp(['fps: ' num2str(licensePlateDetectionTime)]);
    disp(['total images: ' num2str(size(data.imageFilename, 1))]);

    % find successful plate detection
    totSuccess = 0;
    for i = 1:size(myResultsMap)
        predicted = myResultsMap{i,3};
        ground = myResultsMap{i,4};
        if strcmp(predicted, ground)
            totSuccess = totSuccess + 1;
        end
    end
    disp(['total successful recognitions: ' num2str(totSuccess)]);

    % evaluate predicted bounding boxes against ground truth
    imds = imageDatastore(data.imageFilename);
    blds = boxLabelDatastore(data(:, 3));
    ds = combine(imds, blds);
    results = detect(objDetector,ds,'MiniBatchSize',8);
    [ap,recall,precision] = evaluateDetectionPrecision(results,ds,0.2);
    disp("ap: " + num2str(ap));
    disp("detected: " + num2str(size(precision, 1)));

    % calculate false-positives
    falsePositives = 0;
    totPrecision = 0;
    for i = 1:size(precision, 1)
        if (precision(i) < 0.8)
            falsePositives = falsePositives + 1;
        end
    
        totPrecision = totPrecision + precision(i);
    end
    totPrecision = totPrecision / size(precision, 1);
    disp("average precision without missed detections: " + num2str(totPrecision));
    disp("detections false positives: " + num2str(falsePositives));
end