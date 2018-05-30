%% Support Vector Machines-Linear Kernel

X1=[];
d=pwd;
srcfiles=dir(strcat(d,'\face1\*.jpg'));
for i=1:length(srcfiles)
    filename=strcat(strcat(d,'\face1\'),srcfiles(i).name);
    
    I=imread(filename);
    I=rgb2gray(I);
    
    I=imresize(I,[200,320]);
    [I,vis]=extractHOGFeatures(I);
    X1=[X1;I];
end

X2=[];
srcfiles=dir(strcat(d,'\face2\*.jpg'));
for i=1:length(srcfiles)
    filename=strcat(strcat(d,'\face2\'),srcfiles(i).name);
    I=imread(filename);
    I=rgb2gray(I);
    if i==length(srcfiles)
        wil=I;
    end
    I=imresize(I,[200,320]);
    [I,vis]=extractHOGFeatures(I);
    X2=[X2;I];
end
figure;
subplot(2,1,1);
imshow(wil);
subplot(2,1,2);
plot(vis);
title('Visualization from HOG feature extraction');

X1=[ones(186,1),X1];
X2=[ones(186,1),X2];

X=[X1;X2];
b=2;
while b<=33697
    X(:,b)=(X(:,b)-mean(X(:,b)))/std(X(:,b));
    b=b+1;
end
X=double(X);

y(1:186)=1;
y(187:372)=0;
y=y(:);
alpha=0.7;
C=10^(-7);
m=372;
theta=rand(m,1);
theta=[1;theta];

cost=[];



f=X*X';
f=[ones(length(f),1),f];

p=randperm(m,ceil(0.2*m));  
p=p(:);
xtest=f(p,:);
f(p,:)=[];
ytest=y(p);
y(p)=[];
m=length(f);
bh=[];

% Gradient descent

k=1;
while k<=200
    
    i=1;
    t=f*theta;
    del=C*((1-2*y)'*f)'+theta;
    del=del(:);
    theta=theta-alpha*del;
    cost=C*(y'*costone(t) + (1-y)'*costzero(t) )+ 0.5*sum(theta(2:end).^2);
    bh=[bh;cost];
    k=k+1;
end
bh=bh(:);

figure;
plot(bh);
title('Cost vs iteration');
xlabel('Iterations');
ylabel('Cost ');

% Prediction

[a,b]=size(xtest);
hyp=xtest*theta;
pol=hyp;
hyp=hyp>=0;
error=abs(hyp-ytest);
error=error>0;
sume=sum(error);
accuracy=(1-(sume/a))*100





    

    
    
    
    
    
    


