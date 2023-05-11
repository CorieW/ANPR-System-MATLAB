function output = parseDataset()
    datasetDir = ['dataset/'];
    
    % get list of xml files with the labelling in the directory
    labelsFiles = dir(fullfile(datasetDir, '*.xml'));
    
    % create a cell array to temporarily store the labels
    columns = 3;
    dataset = cell(0, columns);
    
    %iterate over each of the xml files
    for i = 1:numel(labelsFiles)
        try
            % load xml file
            labelsFile = fullfile(datasetDir, labelsFiles(i).name);
            xml = xmlread(labelsFile);
    
            % convert the xml to a struct
            dataStruct = xml2struct(xml);
    
            % extract the data using the struct
            fileName = fullfile(datasetDir, dataStruct.annotation.filename.Text);
            plate = dataStruct.annotation.object;
            plateNumber = plate.name.Text;
            xmin = str2num(plate.bndbox.xmin.Text) ;
            ymin = str2num(plate.bndbox.ymin.Text) ;
            xmax = str2num(plate.bndbox.xmax.Text) ;
            ymax = str2num(plate.bndbox.ymax.Text) ;
            bndBox = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1];
    
            % store the extracted data in the cell array
            dataset{i, 1} = fileName;
            dataset{i, 2} = plateNumber;
            dataset{i, 3} = bndBox;
        catch ex
            % problem with parsing
            disp("Problem with parsing file: " + labelsFiles(i).name);
            disp(getReport(ex));
        end
    end
    
    % output as a table
    dataset = table(dataset(:, 1), dataset(:, 2), dataset(:, 3), 'VariableNames', {'imageFilename', 'number', 'licensePlate'});
    output = dataset;
end