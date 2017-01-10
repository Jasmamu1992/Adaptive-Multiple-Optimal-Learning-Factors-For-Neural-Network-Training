%%-------------------------------------------------------------------------------------------------------------------------------------

%%Adaptive Multiple Optimal Learning Factors Algorithm

%%Author  :  Jeshwanth Challagundla
%%Advisor :  Dr. Michael T Manry

%%Copyright © 2015 by Jeshwanth Challagundla
%%All rights reserved. No part of this publication may be reproduced, distributed, or transmitted in any form or by any means, 
%%including photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the publisher, 
%%except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by copyright law

%%--------------------------------------------------------------------------------------------------------------------------------------

clear all;
close all;
clear memory;
clc;

%Choosing data set
data=1;
if(data==1)
    training_file='Twod.tra';
    N=8;
    M=7;
end
if(data==2)
    training_file='oh7.tra';
    N=20;
    M=3;
end
if(data==3)
    training_file='SINGLE2.tra';
    N=16;
    M=3;
end
if(data==4)
    training_file='Power12trn.dat';
    N=12;
    M=1;
end
if(data==5)
    training_file='mattrn.dat';
    N=4;
    M=4;
end
if(data==6)
    training_file='concrete.tra';
    N=8;
    M=1;
end
if(data==7)
    training_file='forest.tra';
    N=12;
    M=1;
end
if(data==8)
    training_file='housing.tra';
    N=13;
    M=1;
end
if(data==9)
    training_file='redwine.tra';
    N=11;
    M=1;
end
if(data==10)
    training_file='F17C.DAT';
    N=17;
    M=39;
end

%No. of hidden units
Nh = 10;
%No. if iterations 
Nit =100;

% The following code reads a text file and stores all the paterns in
% an Nv by (N+M) matrix
fid = fopen(training_file, 'r');
training_file_values = fscanf(fid, '%f');
fclose(fid);
Nv = numel(training_file_values)/(N+M);
fprintf('Nv = %d\n', Nv);
training_file_values = reshape(training_file_values, [(N+M) Nv])';

% Store the inputs in variable x and the desired outputs in variable t
x = training_file_values(:, 1:N);
t = training_file_values(:, N+1:N+M);
clear training_file_values;

%miltiplies
Nu=N+1+Nh;
Mowo_bp=Nv*(Nh*(M+2*N+3)+M*(2*Nu+1))+Nu*(Nu+1)*(M+(2*Nu+1)/6+Nv/2+3/2);
Mhwo= (N+1)*(N+2)*(Nh+(2*N+3)/6+Nv/2+3/2);
Mh=Nh*(N+1);
Mweight_update=Nh*(N+1);

%% net control
%findng mean of input vector
mx=mean(x);
%making input vector of zero mean
x=x-repmat(mx,[Nv,1]);
%cheking whether the input mean is zero
mx=mean(x);
fprintf('the mean of the inputs over all the paterns are:');
disp(mx);
%randomizing input weights
Wih=mlp_randn(Nh,N+1);
%computing net and output for hidden units
x = [ones(Nv,1) x];
net=x*Wih';
%calculating mean of net function
hm=mean(net);
%calculation of std of net function
hv=std(net,1);
%net control
hmean=0.5;
hvar=1.0;
Wih = Wih * hvar./ repmat(hv', [1 N+1]);
Wih(:,1) = Wih(:,1) + hmean - hm' .* hvar./ hv';
%computing net and output for hidden units again
net=x*Wih';
%cheking the mean and std of net function
%calculating mean of net function
hm=mean(net);
%calculation of std of net function
hv=std(net,1);
fprintf('the standard deviations of net function after net control are:');
disp(hv);
fprintf('the mean of net function after net control are:');
disp(hm);
Nu  =N+1+Nh;
fprintf('it\t\t\t\tMSE\n');

%Initializing variables
Mult=zeros(Nit,1);
Mult_acc=zeros(Nit,1);
errPerMult=zeros(Nit,1);
MSE=zeros(Nit,1);
average_clust=0;
errPerMult(1)=0;
dummy=0;

%% OWO
net=x*Wih';
O=act(net);
x1(:,1:N+1)= x;
x1(:,N+2:N+1+Nh)= O;
[R,C]=autocross(x1,t,Nv);

%performing OWO
[wo] = OLS(R,C,M);
Wo=wo';
Woi=Wo(:,1:N+1);
Woh=Wo(:,N+2:Nu);

%calculation of output
y=x1*Wo';
E=t-y;
MSE(1)=sum(sum(E.*E))/Nv;

%calculation of number of clusters for the first iteration. This step is
%optional(This step takes lot of time as it computes the best number of 
%clusters per hidden unit that create the biggest error decrease)
[clust]=find_Nclust(N,Nh,x,Wih,Nv,t,M);

%Dividing N+1 Weights(Weights connected to one hidden unit) into clust
%groups 
%%---------------------------------------------------------------------------------------------------------------
%clust   : number of clusters per hidden unit (This remains same for all hidden units)
%no_clust: Vector that holds the size of the clusters for one hidde unit(This remains same for all hidden units)
%%----------------------------------------------------------------------------------------------------------------
Num=round((N+1)/clust);
Num1=round((N+1)-((clust-1)*Num));
if(Num1<=0)
    Num=Num-1;
    Num1=round((N+1)-((clust-1)*Num));
end
no_clust=zeros(1,clust);
for b=1:clust
    no_clust(b)=Num;
    if (b==clust)
        no_clust(b)=Num1;
    end
end

%Start of iterations
for it=2:Nit
    %calculation of input weight gradient
    diffnet=O.*(1-O);
    do=2*(t-y);
    d=diffnet.*(do*Woh(:,1:Nh));
    G=d'*x/Nv;
    
    %HWO
    Ro=(x'*x)/Nv;
    G1=G';
    e=OLS(Ro,G1,Nh);
    e=e';
    
    %computation of curvatures(secong partial of error wrt input weights)
    h=diag(sum(Woh.^2))*((diffnet.^2)'*(x.^2));
    h=h*2/Nv;
    
    G1=(h');
    [~,I]=sort(G1,'descend');
    I=I';
    if (clust==N+1)
        dum=1:N+1;
        I=repmat(dum,Nh,1);
    end
    L=0;
    doy=zeros(Nv,M,Nh*clust);
    impG=1;
    for b=1:clust
        for k=1:Nh
            col=I(k,impG:impG+no_clust(b)-1);
            doy1=zeros(Nv,M);
            L=L+1;
            for a=1:no_clust(b)
                n=col(a);
                for p=1:Nv
                    for i=1:M
                        doy1(p,i)=doy1(p,i)+x(p,n)*diffnet(p,k)*Woh(i,k)*e(k,n);
                    end
                end
            end
            doy(:,:,L)=doy1;
        end
        impG=impG+no_clust(b);
    end
    g=zeros(L,1);
    for k=1:L
        g(k)=sum(sum((t-y).*(doy(:,:,k))))*2/Nv;
    end
    H=zeros(L, L);
    for a=1:L
        for b=1:L
            H(a,b)=sum(sum(doy(:,:,a).*doy(:,:,b)))*2/Nv;
        end
    end
    [z] = OLS(H,g,1);
    % weight update
    L=0;
    impG=1;
    for b=1:clust
        for k=1:Nh
            col=I(k,impG:impG+no_clust(b)-1);
            doy1=zeros(Nv,M);
            L=L+1;
            for a=1:no_clust(b)
                n=col(a);
                Wih(k,n)=Wih(k,n)+z(L)*e(k,n);
            end
        end
        impG=impG+no_clust(b);
    end
    net=x*Wih';
    O=act(net);
    x1(:,1:N+1)= x;
    x1(:,N+2:N+1+Nh)= O;
    [R,C]=autocross(x1,t,Nv);
    
    %performing OWO
    [wo] = OLS(R,C,M);
    Wo=wo';
    Woi=Wo(:,1:N+1);
    Woh=Wo(:,N+2:Nu);
    
    %calculation of output and MSE
    y=x1*Wo';
    E=t-y;
    MSE(it)=sum(sum(E.*E))/Nv;
    fprintf('%d\t\t\t%f\n',it,MSE(it));
    
    %Calculation of multiplies consumed by the current iteration
    Mdoy=Nv*Nh*(M+N+2);
    Mmolf_v2=Nv*M*L*((L+3)/2)+L*(L+1)*(5/2+(2*L+1)/6)+Mh+Mdoy+Mweight_update;
    dummy=dummy+Mowo_bp+Mhwo+Mmolf_v2;
    Mult(it)=Mowo_bp+Mhwo+Mmolf_v2;
    Mult_acc(it)=dummy;
    
    %Calculating Error change per multiply
    errPerMult(it)=(MSE(it-1)-MSE(it))/Mult(it);
    
    %Running average of number of clusters per hidden unit over all the
    %iterations
    average_clust=average_clust+clust;
    
    
    %%This section decides the number of clusters per hidden unit for next
    %%iteration. Number of clusters per hidden unit increases with the
    %%increase of error decrease per multiple and viceversa
    %%----------------------------------------------------------------------------------------------------------------------------
    if(it==1)
        clust=clust+1;
        clust=clust+1;
        [m,in]=max(no_clust);
        m1=round(m/2);
        m2=m-m1;
        no_clust=[no_clust(1:in-1) m1 m2 no_clust(in+1:end)];
    else
        %Subdividing a cluster
        if(errPerMult(it)>errPerMult(it-1)&&clust<N+1)
            clust=clust+1;
            [m,in]=max(no_clust);
            m1=round(m/2);
            m2=m-m1;
            no_clust=[no_clust(1:in-1) m1 m2 no_clust(in+1:end)];
        %Merging two clusters    
        elseif(clust>1)
            clust=clust-1;
            [m,I]=sort(no_clust,'ascend');
            m1=m(1)+m(2);
            if(I(1)<I(2))
                no_clust=[no_clust(1:(I(1)-1)) m1 no_clust((I(1)+1):(I(2)-1)) no_clust((I(2)+1):end)];
            end
            if(I(2)<I(1))
                no_clust=[no_clust(1:(I(2)-1)) m1 no_clust((I(2)+1):(I(1)-1)) no_clust((I(1)+1):end)];
            end
        end
    end
    %%----------------------------------------------------------------------------------------------------------------------------
end

%Average number of clusters per hidden unit over all the iterations
L=average_clust/(Nit-1);


it=1:1:Nit;
% figure(1);
% plot(log(Mult_acc(it)),MSE(it),'b');
% figure(2);
plot(it,MSE(it));
hold on


