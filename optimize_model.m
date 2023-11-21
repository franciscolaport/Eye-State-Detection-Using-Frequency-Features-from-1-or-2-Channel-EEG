function fit_model = optimize_model(trainData, trainLabel, classifier)

% Divide train data into train and validation sets.
c = cvpartition(size(trainData,1), 'KFold', 3);
params = struct('Optimizer', 'bayesopt', 'AcquisitionFunctionName', ...
    'expected-improvement-plus', 'ShowPlots', false, ...
    'CVPartition', c, 'Verbose', 0);

%LDA
if strcmp(classifier, 'lda')
    % Search for optimal hyperparameters according to train/validation.
    fit_model = fitcdiscr(trainData, trainLabel,'DiscrimType', 'linear', ...
        'OptimizeHyperparameters','auto', ...
        'HyperparameterOptimizationOptions', params);
%SVM
elseif strcmp(classifier, 'svm')
    fit_model = fitcsvm(trainData, trainLabel,'KernelFunction', 'linear', ...
        'Standardize' , true, 'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', params);
    
% Decission Tree
elseif strcmp(classifier, 'tree')
    fit_model = fitctree(trainData, trainLabel, ...
        'OptimizeHyperparameters', 'auto', ...
        'HyperparameterOptimizationOptions', params);
end
end