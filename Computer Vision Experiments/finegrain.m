function finegrain(experiment, who)

if strcmp('dog',who)
    who = [who, ' '];
end
disp('Creating and Combining Samples...')
FPortlandFolder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking\Front';
FFolder = 'C:\Users\Iyanu\Dropbox\people-crops\New\pedestrian front';
[FSamples, ~] = fg_combine_return(FPortlandFolder, FFolder, who);

F_LPortlandFolder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking\Front-left';
F_LFolder = 'C:\Users\Iyanu\Dropbox\people-crops\New\pedestrian front-left';
[FLSamples, ~] = fg_combine_return(F_LPortlandFolder, F_LFolder, who);

F_RPortlandFolder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking\Front-right';
F_RFolder = 'C:\Users\Iyanu\Dropbox\people-crops\New\pedestrian front-right';
[FRSamples, ~] = fg_combine_return(F_RPortlandFolder, F_RFolder, who);

BPortlandFolder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking\Back';
BFolder = 'C:\Users\Iyanu\Dropbox\people-crops\New\pedestrian back';
[BSamples, ~] = fg_combine_return(BPortlandFolder, BFolder, who);

B_LPortlandFolder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking\Back-left';
B_LFolder = 'C:\Users\Iyanu\Dropbox\people-crops\New\pedestrian back-left';
[BLSamples, ~] = fg_combine_return(B_LPortlandFolder, B_LFolder, who);

B_RPortlandFolder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking\Back-right';
B_RFolder = 'C:\Users\Iyanu\Dropbox\people-crops\New\pedestrian back-right';
[BRSamples, ~] = fg_combine_return(B_RPortlandFolder, B_RFolder, who);

LPortlandFolder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking\left';
LFolder = 'C:\Users\Iyanu\Dropbox\people-crops\New\pedestrian my-left';
[LSamples, ~] = fg_combine_return(LPortlandFolder, LFolder, who);

RPortlandFolder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking\right';
RFolder = 'C:\Users\Iyanu\Dropbox\people-crops\New\pedestrian my-right';
[RSamples, ~] = fg_combine_return(RPortlandFolder, RFolder, who);

[f,~] = size(FSamples);
[f_l,~] = size(FLSamples);
[f_r,~] = size(FRSamples);
[b,~] = size(BSamples);
[b_l,~] = size(BLSamples);
[b_r,~] = size(BRSamples);
[l,~] = size(LSamples);
[r,~] = size(RSamples);

if experiment == 2
    disp('Experiment Combining All Front&Back Samples and Left&Right Samples')
    disp('An SVM is used as the Binary Classifier')
    disp('SVM Fitting....')
    
    tr_sample = ...
        [FSamples((1:floor(f * 0.75)),:);...
        FLSamples((1:floor(f_l * 0.75)),:);...
        FRSamples((1:floor(f_r * 0.75)),:);...
        BSamples((1:floor(b * 0.75)),:);
        BLSamples((1:floor(b_l * 0.75)),:);...
        BRSamples((1:floor(b_r * 0.75)),:);...
        LSamples((1:floor(l * 0.75)),:);...
        RSamples((1:floor(r * 0.75)),:)];
    
    test_sample = ...
        [FSamples((floor(f * 0.75)+1:f),:);...
        FLSamples((floor(f_l * 0.75)+1:f_l),:);...
        FRSamples((floor(f_r * 0.75)+1:f_r),:);...
        BSamples((floor(b * 0.75)+1:b),:);...
        BLSamples((floor(b_l * 0.75)+1:b_l),:);...
        BRSamples((floor(b_r * 0.75)+1:b_r),:);...
        LSamples((floor(l * 0.75)+1:l),:);...
        RSamples((floor(r * 0.75)+1:r),:)];
    
    % use 75% for training and 25% for testing
    
    t_labels = ...
        [repmat({'F-FL-FR-B-BL-BR'},1,...
        f - floor(f * 0.75) + f_l - floor(f_l * 0.75) + ...
        f_r - floor(f_r * 0.75) + b - floor(b * 0.75) + ...
        b_l - floor(b_l * 0.75) + b_r - floor(b_r * 0.75)),...
        repmat({'L-R'},1, r - floor(r * 0.75) + l - floor(l * 0.75))];
    
    tr_labels = ...
        [repmat({'F-FL-FR-B-BL-BR'},1,...
            floor(f * 0.75) + floor(f_l * 0.75) + floor(f_r * 0.75) +...
            floor(b_r * 0.75) + floor(b * 0.75) + floor(b_l * 0.75)),...
        repmat({'L-R'},1, floor(r * 0.75) + floor(l * 0.75))];
    
    % SVM Classifier
    Model = fitcsvm(tr_sample,tr_labels);
    
    % Testing
    disp('Testing...')
    [w,~] = size(test_sample);
    predicted = repmat({'A'},1,w);
    acc = 0;
    for k=1 : w
        % record result
        predicted(k) = predict(Model,test_sample(k,:));
        % Compare with expected result
        if strcmp(predict(Model,test_sample(k,:)),t_labels(k))
            acc = acc + 1;
        end
    end
    
    % Accuracy Calculation
    acc = acc/w;
    disp(acc)
       
    % Plot Confusion Matrix
    labels = {'F-FL-FR-B-BL-BR' 'L-R'};
    [Conf_Mat,order] = confusionmat(t_labels,predicted);
    disp(Conf_Mat)
    disp(order)
    heatmap(Conf_Mat, labels, labels, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
    % cfmatrix2(char(t_labels),char(predicted))
    % stat = confusionmatStats(test_labels,predicted);
    
else
    disp('Starting Experiment....')
    classes = {'F' 'FL' 'FR' 'B' 'BL' 'BR' 'L' 'R'};
    samples = {FSamples;FLSamples;FRSamples;BSamples;BLSamples; ...
        BRSamples;LSamples;RSamples};
    sizes = [f;f_l;f_r;b;b_l;b_r;l;r];
    
%     FSamples(1,:)
    
    map = struct('class',1,'sample',[],'size',1);
    for i=1 : length(classes)
        map(i).class = classes(i);
        map(i).sample = samples(i,:);
        map(i).size = sizes(i);
    end
       
%   disp('Done with hashing') 
    % Each pair of classes to train
    % 28 classes expected. Each represented as a struct
    Models = struct('class',1,'models',[]);
    tcounter = 1;
    disp('Starting Model Creation...')
    for i=1 : 7
        for j=i+1 : 8
            % The classes in question
            class1 = map(i).class;
            class2 = map(j).class;
            
            sample1 = map(i).sample;
            sample2 = map(j).sample;
                        
            size1 = map(i).size;
            size2 = map(j).size;
            
            [Model, ~, ~] = ...
                fg_return_model(sample1, sample2, size1, size2,...
                0.75, char(class1),char(class2));
            
            Models(tcounter).class = strcat(char(class1),'VS',char(class2));
            Models(tcounter).models = Model;
            tcounter = tcounter + 1;
        end
    end
    disp('Gathering Test Samples...')
    test_sample = ...
        [FSamples((floor(f * 0.75)+1:f),:);...
        FLSamples((floor(f_l * 0.75)+1:f_l),:);...
        FRSamples((floor(f_r * 0.75)+1:f_r),:);...
        BSamples((floor(b * 0.75)+1:b),:);...
        BLSamples((floor(b_l * 0.75)+1:b_l),:);...
        BRSamples((floor(b_r * 0.75)+1:b_r),:);...
        LSamples((floor(l * 0.75)+1:l),:);...
        RSamples((floor(r * 0.75)+1:r),:)];
    disp('Gathering Test Labels...')
    t_labels = ...
        [repmat({'F'},1,f - floor(f * 0.75)),...
        repmat({'FL'},1,f_l - floor(f_l * 0.75)), ...
        repmat({'FR'},1,f_r - floor(f_r * 0.75)), ...
        repmat({'B'},1,b - floor(b * 0.75)), ...
        repmat({'BL'},1,b_l - floor(b_l * 0.75)), ...
        repmat({'BR'},1,b_r - floor(b_r * 0.75)), ...
        repmat({'L'},1,l - floor(l * 0.75)),...
        repmat({'R'},1,r - floor(r * 0.75))];
    disp('Starting Testing...')
    % testing
    [w,~] = size(test_sample);
    predicted = repmat({'A'},1,w);

    acc = 0;
    for k=1 : w
       temp = repmat({'A'},1,length(Models));
       % All models vote here
       for i=1 : length(Models)
           temp(i) = predict(Models(i).models,test_sample(k,:));
       end
       
       % Pick the mode class as the predicted class
       y = unique(temp);
       n = zeros(length(y), 1);
       for iy = 1:length(y)
         n(iy) = length(find(strcmp(y{iy}, temp)));
       end
       [~, itemp] = max(n);
       
       % record result
       predicted(k) = y(itemp);
       % If prediction is same as actual
        if strcmp(y(itemp),t_labels(k))
            acc = acc + 1;
        end
       
    end  
       
    % Accuracy Calculation
    disp(acc)
    acc = acc/w;
    disp(acc)
        
    % Plot Confusion Matrix
    labels = {'F' 'FL' 'FR' 'B' 'BL' 'BR' 'L' 'R'};
    [Conf_Mat,order] = confusionmat(t_labels,predicted);
    disp(Conf_Mat)
    disp(order)
    heatmap(Conf_Mat, labels, labels, 1,'Colormap','red','ShowAllTicks'...
       ,1,'UseLogColorMap',true,'Colorbar',true);
end


end
