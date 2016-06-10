function [SVMModel,accuracy,labels,sample_array] = hog(who)
%% Generate trianing samples from images
Folder = 'C:\Users\Iyanu\Dropbox\DogWalkingData\PortlandSimpleDogWalking';

% 375(375 +ves, 375 -ves) training 
% 125(125 +ves, 125 -ves) Test

% Check to make sure that folder actually exists.  Warn user if it doesn't.
if ~isdir(Folder)
  errorMessage = sprintf...
      ('Error: The following folders do not exist:\n%s\n%s',...
      trainFolder,testFolder);
  uiwait(warndlg(errorMessage));
  return;
end

% Get a list of all files in the folder with the desired file name pattern.
imagesFiles = dir(fullfile(Folder, '*.jpg'));
labelFiles = dir(fullfile(Folder, '*.labl'));

sample_array = zeros(1000, 20736);
labels = zeros(1,1000);
counter = 1; % variable for insertion into matrix
disp('Starting Sample Creation...')

for k = 1 : length(imagesFiles)
  baseImagesName = imagesFiles(k).name;
  baseLabelName = labelFiles(k).name;
  
  fullImageName = fullfile(Folder, baseImagesName);
  fullLabelName = fullfile(Folder, baseLabelName);
  
  bbox_dimension = prasing(fullLabelName,who);
  
  im = imresize(imcrop(imread(fullImageName), bbox_dimension), [200,200]);
  [hog,~] = extractHOGFeatures(im);
  sample_array(counter,:) = hog;
  labels(counter) = 1;
  
  [w,~,~] = size(imread(fullImageName));
  
  for k2 = 1 : 1
      counter = counter + 1; % Increase counter
%       Generate new bounding box for negative sample
      if bbox_dimension(2) == 0
         randomBoundingBox = [randperm(bbox_dimension(1),1),0,...
             bbox_dimension(3),bbox_dimension(4)];
      elseif bbox_dimension(1) == 0
          randomBoundingBox = [0,randperm(bbox_dimension(2),1),...
             bbox_dimension(3),bbox_dimension(4)];
      else
         randomBoundingBox = [randperm(bbox_dimension(1),1), ...
             randperm(bbox_dimension(2),1),bbox_dimension(3),...
             bbox_dimension(4)];
      end
      
      % Calculate IOU
      overlapRatio = bboxOverlapRatio(bbox_dimension,randomBoundingBox); 
      
      while overlapRatio > 0.5 % This might affect performance of SVM
          if bbox_dimension(2) == 0
            randomBoundingBox = [randperm(w-bbox_dimension(3),1),0,...
             bbox_dimension(3),bbox_dimension(4)];
          elseif bbox_dimension(1) == 0
              randomBoundingBox = [0,randperm(bbox_dimension(2),1),...
                 bbox_dimension(3),bbox_dimension(4)];
          else
             randomBoundingBox = [randperm(w-bbox_dimension(3),1), ...
             randperm(bbox_dimension(2),1),bbox_dimension(3),...
             bbox_dimension(4)];
          end
%           disp('inside the loop')
          overlapRatio = bboxOverlapRatio(bbox_dimension,randomBoundingBox);
      end
      
%       disp(fullImageName)
%       disp(overlapRatio)
%       disp(randomBoundingBox)
      n_im = imresize(imcrop(imread(fullImageName), randomBoundingBox),[200,200]);
      [neghog,~] = extractHOGFeatures(n_im);
      sample_array(counter,:) = neghog;
      labels(counter) = -1;
  end
  %fprintf('Done! Now Heading to next positive Sample')
  counter = counter + 1; % After nested iteration, counter should be increased
end
%% Train SVM
disp('Starting SVM Training')
SVMModel = fitcsvm(sample_array((1:750),:),labels((1:750)));
disp('Starting Test')

%% Testing SVM
accuracy = 0;
predicted = zeros(1,length(sample_array(751:1000)));
for j = 1 : length(sample_array(751:1000))
    predicted(j) = predict(SVMModel, sample_array((750+j),:));
    if predict(SVMModel, sample_array((750+j),:)) == labels(750+j)
        accuracy = accuracy + 1;  
   end
end
% Display accuracy
disp('Accuracy of SVM')
disp(accuracy/length(sample_array(751:1000)))

% Plot Confusion Matrix
    lab = unique(labels);
    [Conf_Mat,order] = confusionmat(labels(751:1000),predicted);
    disp(Conf_Mat)
    disp(order)
    heatmap(Conf_Mat, lab, lab, 1,'Colormap','red','ShowAllTicks',1,'UseLogColorMap',true,'Colorbar',true);
    % plotconfusion(test_labels,predicted)
    % stat = confusionmatStats(test_labels,predicted);