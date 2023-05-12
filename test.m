function [licensePlateDetectionTime, myResultsMap] = test(objDetector, testData, temp)
    %% Number plate recognition
    % create temp folder if it doesn't exist
    croppedPlatesTempFolder = [pwd temp];
    if exist(croppedPlatesTempFolder, 'dir') == 0
        mkdir(croppedPlatesTempFolder);
    end
    % remove prior testing temp images
    delete(fullfile(croppedPlatesTempFolder, '*'));
    
    myResultsMap = cell(0, 4);

    tic
    %% Preprocessing
    % loop though each image
    for i = 1:numel(testData.imageFilename)
        filePath = testData.imageFilename{i};
    
        % load the image
        img = imread(filePath);
    
        % detect the license plates
        try
            [bboxes, scores] = detect(objDetector, img);
        catch ex
            % something likely went wrong with the input size
        end
    
        % select only the strongest bounding boxes
        [selectedBboxs] = bboxes;
    
        ocrResults = '';
        % loop through each detected bounding box
        for i2 = 1:size(selectedBboxs, 1)
            % should only be max 1 license plate
            if (i2 == 2)
                break;
            end

            x = selectedBboxs(i2,1);
            y = selectedBboxs(i2,2);
            w = selectedBboxs(i2,3);
            h = selectedBboxs(i2,4);
        
            % crop detected license plates and save image in temp folder
            croppedImg = imcrop(img, [x y w h]);

            % images to grayscale
            grayImg = rgb2gray(croppedImg);

            % noise removal by applying median filtering
            filteredImage = medfilt2(grayImg);

            %% Character recognition
            try
                ocrResults = ocr(filteredImage).Text;
            catch ex
            end
        end
        % no bounding box detected on image
        if size(selectedBboxs, 1) == 0
            continue;
        end

        % store predicted bounding box
        myResultsMap{i, 1} = selectedBboxs(:);
        % store ground truth bounding box
        myResultsMap{i, 2} = testData.licensePlate{i,:};
        % store detected license plate in map
        % will also remove spaces from license plate number
        predicted = regexprep(ocrResults, '\s', '');
        myResultsMap{i, 3} = predicted;
        % store ground truth license plate
        myResultsMap{i, 4} = testData.number{i, 1};
    end
    licensePlateDetectionTime = size(testData.imageFilename, 1) / toc;
end