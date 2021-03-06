clc;clear all;close all;
format longG;

% "Hyperparams"
deviation = .05; %percentage of total value of decision rule, arbitrary
p = .1; %arbitary number, maybe change to zero

%one would normally read these from some file
dataset = [3 4 5 6;9 2 4 1; 23 4 2 1; 52 2 5 7; 2 5 6 1; 2 5 2 6];%the dataset
alphas = [1 -1 2 5 5 -8]; %the output from the svm
labels = [1 -1 1 -1 1 1]; 

frames = zeros(length(dataset));

for j = 1:length(dataset) %for each x_star
	x_star = dataset(j,:);
    s = zeros(length(dataset),2);
    for jj = 1:length(dataset)
        s(jj,1) = jj;
        s(jj,2) = labels(jj) * alphas(jj) * (dataset(jj,:)*x_star'); %for one x_star
    end
    total = sum(s(:,2));
    [tmp, i] = sort(abs(s(:,2)));
    s = s(i,:);
	while abs(sum(s(:,2))) > abs((1-deviation) * total) && abs(sum(s(:,2))) < abs((1 + deviation) * total)
        s = s(2:end,:);
	end
	frames(j,:) = [s(:,1)' zeros(1,6-length(s(:,1)))]; %indices of vector which are the most important
end

counts = histcounts(frames(:)); %how many times was each thing important over all vectors
counts = counts(2:end); %dont include zero


%get rid of the vectors that aren't important to this decision rule

frames = [];
for i = 1:length(dataset)
   num_important = counts(i);
   if num_important / sum(counts) > p %contribute enough to the decision rule
      frames = [frames;dataset(i,:)];
   end
end


csvwrite('output_frames.csv',frames)