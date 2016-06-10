function [f_s,b_s,r_s,l_s] = person_orientation(experiment,who)
% [FVsBModel,FVsLModel, FVsRModel, BVsLModel, BVsRModel, LVsRModel,...
%     front_test_sample, back_test_sample, right_test_sample, left_test_sample]
%% Gather samples from the different folders.
% Major Folder
if strcmp('dog',who)
    who = [who, ' '];
end
disp('Starting Sample Creation...')
Folder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking';
% Front Samples
FrontFolder = 'C:\Users\Iyanu\Dropbox\people-crops\pedestrian front';
[front_samples,front_imagenames] = combine_return(Folder, FrontFolder, 'front',who);
% Back Samples
BackFolder = 'C:\Users\Iyanu\Dropbox\people-crops\pedestrian back';
[back_samples,back_imagenames] = combine_return(Folder, BackFolder, 'back',who);
% my-right Samples
MRFolder = 'C:\Users\Iyanu\Dropbox\people-crops\pedestrian my-right';
[my_right_samples,right_imagenames] = combine_return(Folder, MRFolder, 'my-right',who);
% my-left Samples
MLFolder = 'C:\Users\Iyanu\Dropbox\people-crops\pedestrian my-left';
[my_left_samples,left_imagenames] = combine_return(Folder, MLFolder, 'my-left',who);

%% Training
% Lets do some accounting
[f_s,~] = size(front_samples);
[b_s,~] = size(back_samples);
[r_s,~] = size(my_right_samples);
[l_s,~] = size(my_left_samples);

% Experiment 1 - Using All pairs classification for discrete classes
% For All Pairs classificaion. I need to train k(k-1)/2 classifiers
% This means i'll have: Front Vs Back, Front Vs Right,
% Front Vs Right, Back Vs Left, Back Vs Right and Left Vs Right. Making a
% total of 6 classifiers/SVMModel in this case.
% Use 3/4 of each sample for Testing and reserve 1/4 for Testing

if experiment == 1
    disp('All Pairs Classification Experiment For')
    disp('Pedestrian Direction classification')
%     sprintf('Creation of Models/SVM Fitting...\n\n')
%     sprintf('Total number of Pedestrian Front Samples: %d\n Training: %d\n Test: %d\n',...
%         f_s,floor(f_s * 0.75),f_s-floor(f_s * 0.75))
%     sprintf('Total number of Pedestrian Back Samples: %d\n Training: %d\n Test: %d\n',...
%         b_s,floor(b_s * 0.75),b_s-floor(b_s * 0.75))
%     sprintf('Total number of Pedestrian Left Samples: %d\n Training: %d\n Test: %d\n',...
%         l_s,floor(l_s * 0.75),l_s-floor(l_s * 0.75))
%     sprintf('Total number of Pedestrian Right Samples: %d\n Training: %d\n Test: %d\n',...
%         r_s,floor(r_s * 0.75),r_s-floor(r_s * 0.75))
    disp('Creating the All Pair Classifiers')
    % F Vs B Classifier
    [FVsBModel, front_test_sample, back_test_sample] = ...
        return_model(front_samples, back_samples, f_s, b_s, 0.75, 'F','B');

    % F Vs R Classifier
    [FVsRModel, ~, right_test_sample] = ...
        return_model(front_samples, my_right_samples, f_s, r_s, 0.75, 'F','R');

    % F Vs L classifier
    [FVsLModel, ~, left_test_sample] = ...
        return_model(front_samples, my_left_samples, f_s, l_s, 0.75, 'F','L');

    % B Vs L Classifier
    [BVsLModel, ~, ~] = ...
        return_model(back_samples, my_left_samples, b_s, l_s, 0.75, 'B','L');

    % B Vs R Classifier
    [BVsRModel, ~, ~] = ...
        return_model(back_samples, my_right_samples, b_s, r_s, 0.75, 'B','R');

    % L Vs R Classifier
    [LVsRModel, ~, ~] = ...
        return_model(my_left_samples, my_right_samples, l_s, r_s, 0.75, 'L','R');


    %% Testing
    % Each classifier has to vote. The most voted class is selected as the
    % predicted class. We break ties arbitrarily 
    % First we need to combine test samples.
    disp('Testing...')
    test_sample = [front_test_sample;right_test_sample;back_test_sample;...
        left_test_sample];
    test_labels = [repmat({'F'},1,f_s-floor(f_s * 0.75)),...
        repmat({'R'},1,r_s-floor(r_s * 0.75)),...
        repmat({'B'},1,b_s-floor(b_s * 0.75)),...
        repmat({'L'},1,l_s-floor(l_s * 0.75))];

    [w,~] = size(test_sample);
    predicted = repmat({'A'},1,w);

    acc = 0;
%     front = 0; %zeros(1,24);
%     back = 0; %zeros(1,17);
    %countf = 1;
    %countb = 1;
    for k=1 : w    
        temp = [predict(FVsBModel, test_sample(k,:)), ...
            predict(FVsRModel, test_sample(k,:)), ...
            predict(FVsLModel, test_sample(k,:)), ...
            predict(BVsLModel, test_sample(k,:)), ...
            predict(BVsRModel, test_sample(k,:)), ...
            predict(LVsRModel, test_sample(k,:))];

        % Get most frequent character
        y = unique(temp);
        n = zeros(length(y), 1);
        for iy = 1:length(y)
          n(iy) = length(find(strcmp(y{iy}, temp)));
        end
        [~, itemp] = max(n);

        % record result
        predicted(k) = y(itemp);

        % If prediction is same as actual
        if strcmp(y(itemp),test_labels(k))
            acc = acc + 1;
        % Display misclassified back and left Samples
        elseif strcmp(test_labels(k),{'L'}) && strcmp(y(itemp),{'B'})
            d = left_imagenames(k-85);
            disp(d)
            disp(temp)
            disp(' ')
        % elseif strcmp(test_labels(k),{'B'}) && strcmp(y(itemp),{'F'})
            %back(countb) = k;
            %countb = countb + 1;
%             d = back_imagenames(k+88);
%             disp(d)
%             disp('Actual-B, predicted-F')
%             disp(temp)
%             disp(' ')
%             back = back + 1;
        % elseif strcmp(test_labels(k),{'F'}) && strcmp(y(itemp),{'B'})
            %front(countf) = k;
            %countf = countf + 1;
%             d = front_imagenames(k+285);
%             disp(d)
%             disp('Actual-F, predicted-B')
%             disp(temp)
%             disp(' ')
%             front = front + 1;
        end
    end

    % Accuracy Calculation
    acc = acc/w;
    disp(acc)
    %disp(countb)
    %disp(countf)

    % Plot Confusion Matrix
    labels = [{'F'},{'R'},{'B'},{'L'}];
    [Conf_Mat,order] = confusionmat(test_labels,predicted);
    disp(Conf_Mat)
    disp(order)
    heatmap(Conf_Mat, labels, labels, 1,'Colormap','red','ShowAllTicks'...
        ,1,'UseLogColorMap',true,'Colorbar',true);
    % plotconfusion(test_labels,predicted)
    % stat = confusionmatStats(test_labels,predicted);
end

% Experiment 2 - Combining front and back into one class and right and left
% into one.
if experiment == 2
    disp('Experiment Combining Front&Back Samples and Left&Right Samples')
    disp('An SVM is used as the Binary Classifier')
    disp('SVM Fitting....')
    % Training
    % FB Vs LR Classifier
    tr_sample = ...
        [front_samples((1:floor(f_s * 0.75)),:);...
        back_samples((1:floor(b_s * 0.75)),:);...
        my_right_samples((1:floor(r_s * 0.75)),:);...
        my_left_samples((1:floor(l_s * 0.75)),:)];
    
    test_sample = ...
        [front_samples((floor(f_s * 0.75)+1:f_s),:);...
        back_samples((floor(b_s * 0.75)+1:b_s),:);...
        my_left_samples((floor(l_s * 0.75)+1:l_s),:);...
        my_right_samples((floor(r_s * 0.75)+1:r_s),:)];
    
    t_labels = ...
        [repmat({'FB'},1,f_s-floor(f_s * 0.75)+b_s-floor(b_s * 0.75)),...
        repmat({'LR'},1,r_s-floor(r_s * 0.75)+l_s-floor(l_s * 0.75))];
    
    tr_labels = ...
        [repmat({'FB'},1,floor(f_s * 0.75)+floor(b_s * 0.75)),...
        repmat({'LR'},1,floor(r_s * 0.75)+floor(l_s * 0.75))];
    
    % SVM Classifier
    FBVsLRModel = fitcsvm(tr_sample,tr_labels);
    
    % Testing
    disp('Testing...')
    [w,~] = size(test_sample);
    predicted = repmat({'A'},1,w);
    acc = 0;
    for k=1 : w
        % record result
        predicted(k) = predict(FBVsLRModel,test_sample(k,:));
        % Compare with expected result
        if strcmp(predict(FBVsLRModel,test_sample(k,:)),t_labels(k))
            acc = acc + 1;
        end
    end
    
    % Accuracy Calculation
    acc = acc/w;
    disp(acc)
       
    % Plot Confusion Matrix
    labels = [{'FB'},{'LR'}];
    [Conf_Mat,order] = confusionmat(t_labels,predicted);
    disp(Conf_Mat)
    disp(order)
    heatmap(Conf_Mat, labels, labels, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
    % cfmatrix2(char(t_labels),char(predicted))
    % stat = confusionmatStats(test_labels,predicted);
end