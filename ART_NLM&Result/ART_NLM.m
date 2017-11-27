function X=ART_NLM(P,THETA,delta)
tic;
I=92160;            % 512 * 180条射线
lambda=0.25;    
X=zeros(65536,1);   % result
MAX_ITER=5;

for k=1:MAX_ITER
    for m=1:I
        R=weightjudge(m,THETA,delta); %求解在该射线下的矩阵R,加到X中
        if sum(R)~=0
            X=X+lambda*(P(m)-R*X)/(R*R')*R';
            X=max(X,0); % 非负条件
        end
    end
    X=NLM(X);
    %imshow(reshape(X,256,256));
end
toc;
end
    
function R=weightjudge(m,THETA,delta) % 

n=mod(m-1,512)+1;
t=floor((m-1)/512)+1;
theta=THETA(t);
a=100/256;
N=1:256;
X=(N-128.5)*a;
Y=-X;
d=(n-256.5)*delta;
R=zeros(256,256);

for i=1:256
    y_up=Y(i)+1/2*a;
    x_up=(d-sin(theta)*y_up)/cos(theta);
    y_down=Y(i)-1/2*a;
    x_down=(d-sin(theta)*y_down)/cos(theta);
    if (x_up<-50&&x_down<=-50)||(x_up>=50&&x_down>50)
        continue;
    end
    j_up=round(x_up/a+128.5);
    j_down=round(x_down/a+128.5);
    if j_up<j_down
        j_up=max(1,j_up)+1;
        j_down=min(j_down,256)-1;
        x_right=X(j_up-1)+1/2*a;
        y_right=(-cos(theta)*x_right+d)/sin(theta);
        R(i,j_up-1)=norm([x_right,y_right]-[x_up,y_up]);
        x_left=X(j_down+1)-1/2*a;
        y_left=(-cos(theta)*x_left+d)/sin(theta);
        R(i,j_down+1)=norm([x_left,y_left]-[x_down,y_down]);
        for j=j_up:j_down
            y_left=y_right;
            x_right=X(j)+1/2*a;
            y_right=(-cos(theta)*x_right+d)/sin(theta);
            R(i,j)=sqrt((y_right-y_left)^2+a^2);
        end
    elseif j_up==j_down
        R(i,j_up)=sqrt(a^2+(x_up-x_down)^2);
    else
        j_down=max(1,j_down)+1;
        j_up=min(j_up,256)-1;
        x_right=X(j_down-1)+1/2*a;
        y_right=(-cos(theta)*x_right+d)/sin(theta);
        R(i,j_down-1)=norm([x_right,y_right]-[x_down,y_down]);
        x_left=X(j_up+1)-1/2*a;
        y_left=(-cos(theta)*x_left+d)/sin(theta);
        R(i,j_up+1)=norm([x_left,y_left]-[x_up,y_up]);
        for j=j_down:j_up
            y_left=y_right;
            x_right=X(j)+1/2*a;
            y_right=(-cos(theta)*x_right+d)/sin(theta);
            R(i,j)=sqrt((y_right-y_left)^2+a^2);
        end
    end
end
R=reshape(R,1,65536);

end

function Y=NLM(X)
%5*5
h=0.5;
XX=reshape(X,256,256); % 图像reshape成256 256
YY=XX;
for m=5:252
    for n=5:252
        S=0;
        D=0;
        V0=XX(m-2:m+2,n-2:n+2);
        for i=-2:2
            for j=-2:2
                V=XX(m+i-2:m+i+2,n+j-2:n+j+2);
                u=sum(sum((V-V0).^2),2);
                S=S+exp(-u/h^2)*XX(m+i,n+j);
                D=D+exp(-u/h^2);
            end
        end
        YY(m,n)=S/D;
    end
end
Y=reshape(YY,65536,1);

end