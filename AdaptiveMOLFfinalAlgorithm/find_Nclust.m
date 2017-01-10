function [nclust]=find_Nclust(N,Nh,x,Wih,Nv,t,M)
Wih_initial=Wih;
MSE1=zeros(N+1,1);
Nu=N+Nh+1;
for clust=N+1:-1:1
    Num=round((N+1)/clust);
    Num1=round((N+1)-((clust-1)*Num));
    if(Num1<=0)
        Num=Num-1;
        Num1=round((N+1)-((clust-1)*Num));
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
    
    %calculation of output
    y=x1*Wo';
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
    %%
    if(clust==N+1)
        doy_newton=zeros(Nv,M,Nh,N+1);
        % NEWTON
        for k=1:Nh
            for n=1:N+1
                doy1_newton=zeros(Nv,M);
                for p=1:Nv
                    for i=1:M
                        doy1_newton(p,i)=doy1_newton(p,i)+x(p,n)*diffnet(p,k)*Woh(i,k)*e(k,n);
                    end
                end
                doy_newton(:,:,k,n)=doy1_newton;
            end
        end
        g_newton=zeros(Nh,N+1);
        for k=1:Nh
            for n=1:N+1
                g_newton(k,n)=sum(sum((t-y).*(doy_newton(:,:,k,n))))*2/Nv;
            end
        end
        g_newton=reshape(g_newton,Nh*(N+1),1);
        doy=reshape(doy_newton,Nv,M,Nh*(N+1),1);
        H_newton=zeros(Nh*(N+1), Nh*(N+1));
        for a=1:Nh*(N+1)
            for b=1:Nh*(N+1)
                H_newton(a,b)=sum(sum(doy(:,:,a).*doy(:,:,b)))*2/Nv;
            end
        end
        [r] = OLS(H_newton,g_newton,1);
        r=reshape(r,Nh,N+1);
        %updating weights
        Wih=Wih+r.*e;
    else
        %curvature
        %h=(Woh'*Woh)*((diffnet.^2)'*x.^2);
        h=diag(sum(Woh.^2))*((diffnet.^2)'*(x.^2));
        h=h*2/Nv;
        
        G1=(h');
        [~,I]=sort(G1,'descend');
        I=I';
        L=0;
        doy=zeros(Nv,M,Nh*clust);
        impG=1;
        for b=1:clust
            no_clust=Num;
            if(b==clust)
                no_clust=Num1;
            end
            for k=1:Nh
                col=I(k,impG:impG+no_clust-1);
                L=L+1;
                doy(:,:,L)=sum(doy_newton(:,:,k,col),4);
            end
            impG=impG+no_clust;
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
            no_clust=Num;
            if(b==clust)
                no_clust=Num1;
            end
            for k=1:Nh
                col=I(k,impG:impG+no_clust-1);
                doy1=zeros(Nv,M);
                L=L+1;
                for a=1:no_clust
                    n=col(a);
                    Wih(k,n)=Wih(k,n)+z(L)*e(k,n);
                end
            end
            impG=impG+no_clust;
        end
    end
    
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
    MSE1(clust)=sum(sum(E.*E))/Nv;

    Wih=Wih_initial;
    
end
[~,nclust]=min(MSE1);




