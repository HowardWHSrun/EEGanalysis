function SVM_adult_infant_compare_with_BP(adultDataPath, adultFileName, infantDataPath, infantFileName, outputDir)
    % Load adult data
    adultData = load(fullfile(adultDataPath, adultFileName));
    if ~isfield(adultData, 'band_powers') || ~isfield(adultData, 'S')
        error('The adult .mat file does not contain the expected variables.');
    end
    adultBandPowers = normalizeBandPowers(adultData.band_powers);

    % Load infant data
    infantData = load(fullfile(infantDataPath, infantFileName));
    if ~isfield(infantData, 'band_powers') || ~isfield(infantData, 'S')
        error('The infant .mat file does not contain the expected variables.');
    end
    infantBandPowers = normalizeBandPowers(infantData.band_powers);

    % Specific channels to compare
    infantChannels = [7,9,17,19,37,42,47,51,57,61,68,76,79,86,89,95,100,102];
    adultChannels = [29,2,1,3,7,8,12,13,11,14,15,16,17,18,23,24,28,27];

    % Verify channel indices
    if max(infantChannels) > size(infantBandPowers.delta, 1)
        error('One or more infant channel indices exceed the available channels.');
    end
    if max(adultChannels) > size(adultBandPowers.delta, 1)
        error('One or more adult channel indices exceed the available channels.');
    end

    % Extract features for each band
    bands = {'delta', 'theta', 'alpha', 'beta', 'gamma'};
    features = struct();
    for band = bands
        band = band{1};
        features.(band).adult = extractSpecificChannels(adultBandPowers.(band), adultChannels);
        features.(band).infant = extractSpecificChannels(infantBandPowers.(band), infantChannels);
    end

    % Create the output directory if it does not exist
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Train and test SVM for each band separately
    for band = bands
        band = band{1};
        SVMModel = trainAndTestSVM(features.(band).adult, features.(band).infant, band, outputDir);
    end

    % Combine features from all bands for the final SVM
    combinedAdultFeatures = [];
    combinedInfantFeatures = [];

    for band = bands
        band = band{1};
        combinedAdultFeatures = [combinedAdultFeatures, features.(band).adult];
        combinedInfantFeatures = [combinedInfantFeatures, features.(band).infant];
    end

    SVMModel = trainAndTestSVM(combinedAdultFeatures, combinedInfantFeatures, 'AllBands', outputDir);
end

function SVMModel = trainAndTestSVM(adultFeatures, infantFeatures, band, outputDir)
    % Combine features and labels
    combinedFeatures = [adultFeatures; infantFeatures];
    combinedLabels = [ones(size(adultFeatures, 1), 1); 2 * ones(size(infantFeatures, 1), 1)]; % 1 for adults, 2 for infants

    % Split data into training and test sets
    numSamples = size(combinedFeatures, 1);
    indices = randperm(numSamples);
    trainIdx = indices(1:round(0.7 * numSamples)); % 70% for training
    testIdx = indices(round(0.7 * numSamples) + 1:end); % 30% for testing

    X_train = combinedFeatures(trainIdx, :);
    y_train = combinedLabels(trainIdx);
    X_test = combinedFeatures(testIdx, :);
    y_test = combinedLabels(testIdx);

    % Train SVM with linear kernel using fitcecoc for binary classification
    SVMModel = fitcecoc(X_train, y_train, 'Learners', 'linear', 'Coding', 'onevsall');

    % Perform k-fold cross-validation
    k = 5;
    cv = cvpartition(y_train, 'KFold', k);
    accuracies = zeros(k, 1);

    for i = 1:k
        trainIdxCV = training(cv, i);
        valIdxCV = test(cv, i);

        X_cv_train = X_train(trainIdxCV, :);
        y_cv_train = y_train(trainIdxCV);
        X_cv_val = X_train(valIdxCV, :);
        y_cv_val = y_train(valIdxCV);

        % Train on the training partition
        CVModel = fitcecoc(X_cv_train, y_cv_train, 'Learners', 'linear', 'Coding', 'onevsall');

        % Validate on the validation partition
        y_cv_pred = predict(CVModel, X_cv_val);

        % Calculate accuracy
        accuracies(i) = sum(y_cv_pred == y_cv_val) / length(y_cv_val);
    end

    % Calculate and display the mean cross-validation accuracy
    meanCVAccuracy = mean(accuracies);
    fprintf('%s Band - Mean Cross-Validation Accuracy: %.2f%%\n', band, meanCVAccuracy * 100);

    % Predict on test data
    y_pred = predict(SVMModel, X_test);

    % Calculate accuracy
    accuracy = sum(y_pred == y_test) / length(y_test);
    fprintf('%s Band - Test Accuracy: %.2f%%\n', band, accuracy * 100);

    % Confusion matrix
    confusionMatrix = confusionmat(y_test, y_pred);
    disp([band ' Band - Confusion Matrix:']);
    disp(confusionMatrix);

    % Generate and annotate confusion matrix plot
    figure;
    cm = confusionchart(y_test, y_pred);
    title(sprintf('%s Band - Confusion Matrix', band));

    % Calculate p-values using permutation testing
    numPermutations = 10;
    permAccuracies = zeros(numPermutations, 1);

    for i = 1:numPermutations
        % Shuffle the labels
        shuffledLabels = y_train(randperm(length(y_train)));

        % Train the SVM with shuffled labels
        permModel = fitcecoc(X_train, shuffledLabels, 'Learners', 'linear', 'Coding', 'onevsall');

        % Predict on test data
        y_perm_pred = predict(permModel, X_test);

        % Calculate permutation accuracy
        permAccuracies(i) = sum(y_perm_pred == y_test) / length(y_test);
    end

    % Calculate p-value for actual data
    p_value = mean(permAccuracies >= accuracy);
    fprintf('%s Band - p-value for actual data: %.4f\n', band, p_value);

    % Annotate and save the figure
    annotation('textbox', [0.2, 0.8, 0.1, 0.1], 'String', sprintf('Test Accuracy: %.2f%%', accuracy * 100), 'FitBoxToText', 'on');
    annotation('textbox', [0.2, 0.7, 0.1, 0.1], 'String', sprintf('p-value: %.4f', p_value), 'FitBoxToText', 'on');
    saveas(gcf, fullfile(outputDir, sprintf('confusion_matrix_%s_annotated.png', band)));

    % Close figure to avoid excessive memory usage
    close(gcf);
end

function features = extractSpecificChannels(bandPowers, channels)
    % Function to extract features for specific channels across a frequency band
    features = bandPowers(channels, :)';
end

function normalizedBand = normalizeBandPowers(bandPowers)
    % Normalize each band's power to the range [0, 1]
    fields = fieldnames(bandPowers);
    normalizedBand = struct();
    for i = 1:numel(fields)
        normalizedBand.(fields{i}) = normalize(bandPowers.(fields{i}), 'range');
    end
end

% SVM_adult_infant_compare_with_BP('/Users/howardwang/Desktop/Research/Bayet_Zinser_data_2020/infant-EEG-MVPA-tutorial-master/data', 'Adults_included_band_powers_ratios.mat', '/Users/howardwang/Desktop/Research/Bayet_Zinser_data_2020/infant-EEG-MVPA-tutorial-master/data', 'Infants_included_band_powers_ratios.mat', '/Users/howardwang/Desktop/Research/Bayet_Zinser_data_2020/infant-EEG-MVPA-tutorial-master/results');

