x = rand(361,12);

figure

A = zeros(100,100,3);

for ind = 1:30
    for jnd = 1:30
        A(ind,jnd,:) = [0, 1, 1];
    end
end

imshow(A)