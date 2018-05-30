function [ z] = gkernel( a,b,sig )

s=abs(a-b);
s=-sum(s.^2);
s=s/(2*sig*sig);
z=exp(s);
end

