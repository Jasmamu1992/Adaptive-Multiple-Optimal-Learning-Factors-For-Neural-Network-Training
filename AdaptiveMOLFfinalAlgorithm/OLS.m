function [Wo] = OLS(R,Ci,M)
    Nu = size(R, 1);
    E1=10^-6;
    E2=E1*E1;
    
%first basis function
NLin=1;
g=R(1,1);

if g<E2,
    a(1,1)=0;
    NLin=NLin+1;
else
    g=sqrt(g);
    a(1,1)=1/g;
end

%second basis function
c(1)=a(1,1)*R(1,2);
b(2)=1;
b(1)=-c(1)*a(1,1);
g=R(2,2) - c(1)*c(1);
if g < E2,
    a(2,1)=0;
    a(2,2)=0;
    NLin=NLin+1;
else
    g=sqrt(g);
    a(2,1)=(1/g)*b(1);
    a(2,2)=(1/g)*b(2);
end
%Basis functions 3 through Nu
for n=3:Nu,
    for j=1:n-1,
        c(j)=0;
        for k=1:j,
            c(j)=c(j)+ a(j,k)*R(k,n);
        end
    end
    b(n)=1;
    for j=1:n-1,
        b(j)=0;
        for k=j:n-1,
            b(j)=b(j)-c(k)*a(k,j);
        end
    end
    g=0;
    for k=1:n-1,
        g=g+c(k)*c(k);
    end
    g=R(n,n)-g;
    if g<E2,
        for k=1:n,
            a(n,k)=0;
        end
        NLin=NLin+1;
    else
        g=1./sqrt(g);
        for k=1:n,
            a(n,k)=g*b(k);
        end
    end
end

%Find orthonormal system's output weight as
for i=1:M,
    for m=1:Nu,
        Wo1(i,m)=0;
        for k=1:m,
            Wo1(i,m)=Wo1(i,m)+a(m,k)*Ci(k,i);
        end
    end
end

for i=1:M,
    for k=1:Nu,
        Wo(i,k)=0;
        for m=k:Nu,
            Wo(i,k)=Wo(i,k)+a(m,k)*Wo1(i,m);
        end
    end 
end
Wo=Wo';
