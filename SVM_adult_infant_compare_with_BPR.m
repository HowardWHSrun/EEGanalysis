function SVM_adult_infant_compare_with_BPR(adultDataPath, adultFileName, infantDataPath, infantFileName, outputDir)
    % Load adult data
    adultData = load(fullfile(adultDataPath, adultFileName));
    if ~isfield(adultData, 'band_powers') || ~isfield(adultData, 'S')
        error('The adult .mat file does not contain the expected variables.');
    end
    adultBandPowers = adultData.band_powers;

    % Load infant data
    infantData = load(fullfile(infantDataPath, infantFileName));
    if ~isfield(infantData, 'band_powers') || ~isfield(infantData, 'S')
        error('The infant .mat file does not contain the expected variables.');
    end
    infantBandPowers = infantData.band_powers;

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

    % Extract features and band power ratios
    adultFeatures = extractSpecificChannels(adultBandPowers, adultChannels);
    infantFeatures = extractSpecificChannels(infantBandPowers, infantChannels);

    % Combine features and labels
    combinedFeatures = structfun(@(x) [x; x], adultFeatures, 'UniformOutput', false);  % Initialize combinedFeatures
    for field = fieldnames(adultFeatures)'
        field = field{1};
        combinedFeatures.(field) = [adultFeatures.(field); infantFeatures.(field)];
    end
    combinedLabels = [ones(size(adultFeatures.theta_alpha, 1), 1); 2 * ones(size(infantFeatures.theta_alpha, 1), 1)]; % 1 for adults, 2 for infants

    % Validate the number of samples
    numSamples = size(combinedFeatures.theta_alpha, 1);
    if numSamples ~= length(combinedLabels)
        error('The number of samples in combined features does not match the number of labels.');
    end

    % Split data into training and test sets
    indices = randperm(numSamples);
    trainIdx = indices(1:round(0.7 * numSamples)); % 70% for training
    testIdx = indices(round(0.7 * numSamples) + 1:end); % 30% for testing

    % Validate indices
    if max(trainIdx) > numSamples || max(testIdx) > numSamples
        error('Training or test indices exceed the number of available samples.');
    end

    % Train and test SVM for each feature set
    featureSets = {'theta_alpha', 'beta_alpha', 'theta_alpha_beta', 'theta_beta', 'theta_alpha_beta_ratio', 'gamma_delta'};
    for featureSet = featureSets
        featureSet = featureSet{1};

        % Check if the feature set exists
        if ~isfield(combinedFeatures, featureSet)
            error('Feature set %s does not exist in combinedFeatures.', featureSet);
        end

        X = combinedFeatures.(featureSet);
        y = combinedLabels;

        % Validate dimensions
        if size(X, 1) ~= length(y)
            error('The number of samples in %s does not match the number of labels.', featureSet);
        end

        X_train = X(trainIdx, :);
        y_train = y(trainIdx);
        X_test = X(testIdx, :);
        y_test = y(testIdx);

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
        fprintf('%s Ratio - Mean Cross-Validation Accuracy: %.2f%%\n', featureSet, meanCVAccuracy * 100);

        % Predict on test data
        y_pred = predict(SVMModel, X_test);

        % Calculate accuracy
        accuracy = sum(y_pred == y_test) / length(y_test);
        fprintf('%s Ratio - Test Accuracy: %.2f%%\n', featureSet, accuracy * 100);

        % Confusion matrix
        confusionMatrix = confusionmat(y_test, y_pred);
        disp([featureSet ' Ratio - Confusion Matrix:']);
        disp(confusionMatrix);

        % Generate and save the fully annotated confusion matrix plot
        figure;
        cm = confusionchart(y_test, y_pred);
        title(sprintf('%s Ratio - Confusion Matrix', featureSet));
        annotation('textbox', [0.2, 0.8, 0.1, 0.1], 'String', sprintf('Test Accuracy: %.2f%%', accuracy * 100), 'FitBoxToText', 'on');

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
        fprintf('%s Ratio - p-value for actual data: %.4f\n', featureSet, p_value);

        % Annotate and save the figure
        annotation('textbox', [0.2, 0.7, 0.1, 0.1], 'String', sprintf('p-value: %.4f', p_value), 'FitBoxToText', 'on');
        saveas(gcf, fullfile(outputDir, sprintf('confusion_matrix_%s_annotated.png', featureSet)));
    end

    % Save results to a text file
    resultsFile = fullfile(outputDir, 'SVM_results.txt');
    fid = fopen(resultsFile, 'w');
    fprintf(fid, 'Mean Cross-Validation Accuracies:\n');
    for featureSet = featureSets
        featureSet = featureSet{1};
        fprintf(fid, '%s Ratio: %.2f%%\n', featureSet, meanCVAccuracy * 100);
    end
    fclose(fid);

    % Save workspace variables to a MAT file
    save(fullfile(outputDir, 'SVM_results.mat'), 'SVMModel', 'confusionMatrix', 'meanCVAccuracy', 'accuracy', 'p_value');
end

% Helper function to extract power band ratios for specific channels
function features = extractSpecificChannels(bandPowers, channels)
    % Assume bandPowers is a structure with fields for different bands
    thetaPower = bandPowers.theta(channels, :)'; % Transpose to get samples x channels
    alphaPower = bandPowers.alpha(channels, :)';
    betaPower = bandPowers.beta(channels, :)';
    gammaPower = bandPowers.gamma(channels, :)';
    deltaPower = bandPowers.delta(channels, :)';

    % Calculate power band ratios
    theta_alpha = thetaPower ./ alphaPower;
    beta_alpha = betaPower ./ alphaPower;
    theta_alpha_beta = thetaPower ./ (alphaPower + betaPower);
    theta_beta = thetaPower ./ betaPower;
    theta_alpha_beta_ratio = thetaPower ./ (alphaPower + betaPower + deltaPower);
    gamma_delta = gammaPower ./ deltaPower;

    % Combine ratios into a structure
    features.theta_alpha = theta_alpha;
    features.beta_alpha = beta_alpha;
    features.theta_alpha_beta = theta_alpha_beta;
    features.theta_beta = theta_beta;
    features.theta_alpha_beta_ratio = theta_alpha_beta_ratio;
    features.gamma_delta = gamma_delta;
end


% Example usage:
% SVM_adult_infant_compare_with_BPR('/Users/howardwang/Desktop/Research/Bayet_Zinser_data_2020/infant-EEG-MVPA-tutorial-master/data', 'Adults_included_band_powers_ratios.mat', '/Users/howardwang/Desktop/Research/Bayet_Zinser_data_2020/infant-EEG-MVPA-tutorial-master/data', 'Infants_included_band_powers_ratios.mat', '/Users/howardwang/Desktop/Research/Bayet_Zinser_data_2020/infant-EEG-MVPA-tutorial-master/results');
