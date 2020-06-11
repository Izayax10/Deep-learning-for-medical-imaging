% Script to produce a set of circular phantoms
% Andreas Hauptmann, 2017


recSize=64;
unitVec=linspace(-1,1,recSize);
[rectX,rectY]=meshgrid(unitVec,unitVec(end:-1:1));

save mesh_rect rectX rectY recSize

%% Train data
imagesTrue=zeros(recSize,recSize,1024);
imagesRecon=zeros(recSize,recSize,1024);

theta=0:20:179;

for iii=1:60000

            xya=rand*2*pi; 
            pos=rand*0.5+0.2;
            absorbVal=0.75+rand*0.25;
            hR=0.25;


            he1=1;
            he2=1;
            
            hc1=pos*cos(xya);
            hc2=pos*sin(xya);
            
            hd  = sqrt(he1*(rectX(:)-hc1).^2 + he2*(rectY(:)-hc2).^2);

            image=zeros(recSize);
            image(hd <= hR) = absorbVal;
                     
            
            
            sino=radon(image,theta);
            backproj=iradon(sino,theta,'Ram-Lak',1,recSize);

            imagesTrue(:,:,iii)=image;
            imagesRecon(:,:,iii)=backproj;

end

save('trainDataSet','-v7.3','imagesRecon','imagesTrue');

return

%% Test data
imagesTrue=zeros(recSize,recSize,128);
imagesRecon=zeros(recSize,recSize,128);

theta=0:20:179;
while(length(theta)>40)
k = randi([1 length(theta)]);
theta(k) = [];
end


for iii=1:10000

            xya=rand*2*pi; 
            pos=rand*0.5+0.2;
            absorbVal=0.75+rand*0.25;
            hR=0.25;


            he1=1;
            he2=1;
            
            hc1=pos*cos(xya);
            hc2=pos*sin(xya);
            
            hd  = sqrt(he1*(rectX(:)-hc1).^2 + he2*(rectY(:)-hc2).^2);

            image=zeros(recSize);
            image(hd <= hR) = absorbVal;
                     
            
            
            sino=radon(image,theta);
            backproj=iradon(sino,theta,'Ram-Lak',1,recSize); 

            imagesTrue(:,:,iii)=image;
            imagesRecon(:,:,iii)=backproj;

end

save('testDataSet','-v7.3','imagesRecon','imagesTrue');