dataset = X %the dataset
p = .95 %percentage of total value of decision rule  we want to maintain
for j = 1:length(dataset) %for each x_star
	x_star = dataset(j)
	vector = labels .* alphas .* K(x_i_star,x_star) %for one x_star
	
	percent_changes = zeros(length(vector),2)
	
	for i in range(length(vector))
		percent_changes(i,1) = i
		percent_changes(i,2) = abs(sum([vector(1:i-1) vector(i+1:end))) - abs(sum(vector)) / abs(sum(vector))
	end
	percent_changes = sort(percent_changes(:,2))
	n = 1
	while sum(percent_changes(:,2)) > p
		percent_changes = percent_changes(n:end,:)
		n = n+1
	end
	frames = percent_changes(:,1) %indices of vector which are the most important
end

%somehow take an average over all vectors in the dataset

csvwrite(frames,'output_frames.csv')