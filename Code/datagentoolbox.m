clear, clc
trainingdata_noisy = zeros(64,64,1500);
trainingdata = zeros(64,64,500);
for i = 1:1:500
    [trueim, ima, imb, imc] = gen();
    trainingdata(:,:,i) = imresize(trueim,1.28);
    trainingdata_noisy(:,:,i) = imresize(ima,1.28);
    trainingdata_noisy(:,:,i+1) = imresize(imb,1.28);
    trainingdata_noisy(:,:,i+2) = imresize(imc,1.28);
end 
save('trainingdata','-v7.3','trainingdata_noisy','trainingdata');
testingdata_noisy = zeros(64,64,300);
testingdata = zeros(64,64,100);
for j = 1:1:100
    [trueim, ima, imb, imc] = gen();
    testingdata(:,:,j) = imresize(trueim,1.28);
    testingdata_noisy(:,:,j) = imresize(ima, 1.28);
    testingdata_noisy(:,:,j+1) = imresize(imb,1.28);
    testingdata_noisy(:,:,j+2) = imresize(imc,1.28);
end  
save('testdata','-v7.3','testingdata_noisy','testingdata');