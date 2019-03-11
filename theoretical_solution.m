% This function generates the replica theoretical results for the biologically constrained,
% single-neuron associative learning model described in the manuscript.
% The function produces the results for both the homogeneous and heterogeneous models based on
% Eqs. (27) and (22) of SI. These models include:
% (1) excitatory and inhibitory inputs with sign-constrained weights
% (2) l-1 norm constraint on input weights
% (3) constant threshold
% (4) pre- and postsynaptic spiking errors, synaptic and intrinsic noise 

% INPUT PARAMETERS:
% N: total number of inputs
% finh: fraction of inhibitory inputs
% w: l-1 norm (i.e. average absolute connection weight)
% f: firing probability
% betaInt: Intrinsic noise strength
% betaSyn: Synaptic noise strength
% PreSpikingErrorProb: Presynaptic spiking error probability
% PostSpikingErrorProb: Postsynaptic spiking error probability
% model: homo or heter (homogeneous or heterogeneous)

% OUTPUTS PARAMETERS:
% capacity: memory storage capacity
% Pcon: excitatory and inhibitory connection probabilities
% CV: coefficient of variation of non-zero excitatory and inhibitory connection weights
% J: means of non-zero excitatory and inhibitory connection weights

function [capacity,Pcon,CV,J] = theoretical_solution(N,finh,w,f,betaInt,betaSyn,PreSpikingErrorProb,PostSpikingErrorProb,model)

% VALIDATION OF PARAMETERS 
assert(N>0,'N must be positive')
assert((finh>=0 & finh<1),'finh must be in the [0 1) range')
assert(w>1/f , 'w must be greater than 1/f')
assert((f>0 & f<1),'f must be in the (0 1) range')
assert(betaInt>=0,'betaInt must be nonnegtive')
assert(betaSyn>=0,'betaSyn must be nonnegtive')
assert((PostSpikingErrorProb>=0 & PostSpikingErrorProb<2*f*(1-f)),'PostSpikingErrorProb must be in the [0 2*f*(1-f)) range')

capacity=[]; Pcon=[]; CV=[]; J=[];

fout = f;
Ninh = round(N*finh);
pout1 = PostSpikingErrorProb/2./(1-fout);
pout2 = (1-fout)./fout.*pout1;
Cout1 = sqrt(2)*erfinv(1-2*pout1);
Cout2 = sqrt(2)*erfinv(1-2*pout2);

E = @(x) (1+erf(x))/2;
F = @(x) exp(-x.^2)./pi^0.5+x.*(1+erf(x));
D = @(x) x.*F(x)+E(x);

g=zeros(N,1);
g(1:Ninh) = -1;
g(Ninh+1:end) = 1;

p1 = PreSpikingErrorProb/2./(1-f);
p2 = (1-f)./f.*p1;
Cj = (1-f).*p1.*(1-p1)+ f.*p2.*(1-p2);
Dj = f.*(1-f).*(1-p1-p2).^2;

switch model
    case 'homo'
        assert((nnz(PreSpikingErrorProb>=0)==1 & nnz(PreSpikingErrorProb<2*f*(1-f))==1),'PreSpikingErrorProb must be in the [0 2*f*(1-f)) range')
        options = optimset('Display','off','MaxIter',10^5,'MaxFunEvals',10000,'TolX',10^-8,'TolFun',10^-8);
        S = @(x) [(1-fout)*F(x(1))-fout*F(x(2));...
        ((1-finh)*F(x(4))+finh*F(x(3)))*(betaInt^2 + w*betaSyn*f)-(Dj*4*(x(1)+x(2))/(Cout1 + Cout2)^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1))) + Cj)*w^2*((x(3)-x(4))/w/f-(x(3)+x(4)));...
        ((1-finh)*F(x(4))-finh*F(x(3)))*(betaInt^2 + w*betaSyn*f)-(Dj*4*(x(1)+x(2))/(Cout1 + Cout2)^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1))) + Cj)*w*((x(3)-x(4))/w/f-(x(3)+x(4)))/f;...
        ((1-finh)*D(x(4))+finh*D(x(3)))*(betaInt^2 + w*betaSyn*f)*(2*Dj*(x(1)+x(2))^2/(Cout1 + Cout2)^2 - Cj)-(Dj*4*(x(1)+x(2))/(Cout1 + Cout2)^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1))) + Cj)^2*w^2*((x(3)-x(4))/w/f-(x(3)+x(4)))^2/2];
    for j = 1: 5000
        x0 = [0.1 1.5 0.5 -0.5]+0.1.*rand(1,4);
        [x,~,exitflag] = fsolve(S, x0, options);
        if exitflag == 1 && ((x(3)-x(4))/w/f-(x(3)+x(4)))*(x(1) + x(2))>0 && (x(1) + x(2))/(Cout1 + Cout2)>=0 
            break;
        end
    end
    if  exitflag == 1
        capacity = 16*(x(1)+x(2))^2/(Cout1 + Cout2)^4*(fout*D(x(2))+(1-fout)*D(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1)))^2*((1-finh)*D(x(4)) + finh*D(x(3)))*Dj^2/(Dj*4*(x(1)+x(2))/(Cout1 + Cout2)^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1))) + Cj)^2;
        Pconexc = E(x(4));
        Pconinh = E(x(3));
        CVinh = sqrt(2*D(x(3))*E(x(3))/F(x(3))^2 - 1);
        CVexc = sqrt(2*D(x(4))*E(x(4))/F(x(4))^2 - 1);
        sigma = sqrt(2)*(betaInt^2+w*betaSyn*f)/(Dj*4*(x(1)+x(2))/(Cout1+Cout2)^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1)))+Cj)/w^2/((x(3)-x(4))/w/f-(x(3)+x(4)));
        Jexc = w*sigma*F(x(4))/E(x(4))/sqrt(2);
        Jinh =  w*sigma*F(x(3))/E(x(3))/sqrt(2);
        Pcon = [Pconinh;Pconexc];
        CV = [CVinh;CVexc];
        J = [Jinh;Jexc];
    end
    
    case 'heter'
        assert((nnz(PreSpikingErrorProb>=0)==N & nnz(PreSpikingErrorProb<2*f*(1-f))==N),'Every elements in PreSpikingErrorProb must be in the [0 2*f*(1-f)) range')
        options = optimset('Display','off','MaxIter',10^5,'MaxFunEvals',10000,'TolX',10^-8,'TolFun',10^-8);
        
        S = @(x) [ (1-fout)*F(x(1))-fout*F(x(2));...
                   1/N*sum(sqrt(Dj)./(Cj + Dj*4*(x(1)+x(2))./(Cout1 + Cout2).^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1)))).*F(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj)))-2*w*x(5);...
                   1/N*sum(f.*g.*sqrt(Dj)./(Cj + Dj*4*(x(1)+x(2))./(Cout1 + Cout2).^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1)))).*F(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj))) - 2*x(5);...
                   1/N*sum(Dj./(Cj + Dj*4*(x(1)+x(2))./(Cout1 + Cout2).^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1)))).^2.*(Cj/2-Dj*(x(1)+x(2)).^2/(Cout1 + Cout2).^2).*D(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj))) - betaInt.^2*x(5).^2 + w*x(5)*x(3) + 2*x(5)*x(4);...
                   betaSyn/N*sum(f.*sqrt(Dj)./(Cj + Dj*4*(x(1)+x(2))./(Cout1 + Cout2).^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1)))).*F(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj)))-2*w*x(3)-4*x(4)+4*betaInt.^2*x(5);];
        for j = 1:5000
            x0 = [0 1 0 0 0 ]+0.1.*rand(1,5);
            [x,~,exitflag] = fsolve(S, x0, options);
            if exitflag ==1 && x(5)*(x(1) + x(2)) > 0 &&(x(1) + x(2))/(Cout1 + Cout2) >=0
                break;
            end
        end
        if  exitflag >0
            fac = 16*(x(1)+x(2)).^2/(Cout1+Cout2).^4*(fout*D(x(2))+(1-fout)*D(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1))).^2;
            capacity = fac* 1/N*sum(Dj.^2./(Cj + Dj*4*(x(1)+x(2))/(Cout1 + Cout2).^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1)))).^2.*D(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj)));
            Pcon = E(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj));
            CV = g.*sqrt(2*D(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj)).*E(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj))./F(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj)).^2 - 1);
            J = sqrt(Dj)./(Cj + Dj*4*(x(1)+x(2))./(Cout1 + Cout2).^2*(fout*E(x(2))+(1-fout)*E(x(1)))/(fout * F(x(2))+(1-fout)*F(x(1))))/x(5)/2.*F(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj))./E(-(x(3)+2*f.*g*x(4) + betaSyn*f*x(5))/2./sqrt(Dj));
        else
            capacity = NaN;
            Pcon = NaN;
            CV = NaN;
            J = NaN;
        end
    otherwise
        error('Unexpected model, use homo or heter.')
end