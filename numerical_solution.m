% This function generates the numerical results for the biologically constrained,
% single-neuron associative learning model described in the manuscript.
% The function produces the results based on Eqs. (29) of SI. The model includes:
% (1) excitatory and inhibitory inputs with sign-constrained weights
% (2) l-1 norm constraint on input weights
% (3) constant threshold, h=1
% (4) pre- and postsynaptic spiking errors, synaptic and intrinsic noise 

% This code works with MATLAB version R2017a or later

% INPUT PARAMETERS:
% X: input associations, N x number of associations
% Xp: outputs associations, 1 x number of associations
% finh: fraction of inhibitory inputs
% w: average absolute connection weight (l-1 norm constraint)
% f: firing probability
% betaInt: Intrinsic noise strength
% betaSyn: Synaptic noise strength
% PreSpikingErrorProb: Presynaptic spiking error probability
% PostSpikingErrorProb: Postsynaptic spiking error probability

% OUTPUTS PARAMETERS:
% W: input weights, Nx1
% flearned: fraction of successfully learned associations

function [W,flearned] = numerical_solution(X,Xp,finh,w,f,betaInt,betaSyn,PreSpikingErrorProb,PostSpikingErrorProb)

% VALIDATION OF PARAMETERS
assert(size(Xp,1)==1,'Xp must be 1 x number of associations, containing only zeros and ones')
assert(size(X,2)==size(Xp,2),'X and Xp must have the same second dimension size')
temp=unique(X(:));
if length(temp)==1
    assert((temp==0 || temp==1),'X must be N x number of associations, containing only zeros and ones')
elseif length(temp)==2
    assert(nnz(temp-[0;1])==0,'X must be N x number of associations, containing only zeros and ones')
else
    error('X must be N x number of associations, containing only zeros and ones')
end
temp=unique(Xp);
if length(temp)==1
    assert((temp==0 || temp==1),'Xp must be 1 x number of associations, containing only zeros and ones')
elseif length(temp)==2
    assert(nnz(temp-[0 1])==0,'Xp must be 1 x number of associations, containing only zeros and ones')
else
    error('Xp must be 1 x number of associations, containing only zeros and ones')
end
[N,m] = size(X);
assert((finh>=0 & finh<1),'finh must be in the [0 1) range')
assert(w>1/f , 'w must be greater than 1/f')
assert((f>0 & f<1),'f must be in the (0 1) range')
assert(betaInt>=0,'betaInt must be nonnegtive')
assert(betaSyn>=0,'betaSyn must be nonnegtive')
assert(size(PreSpikingErrorProb,1)==N,'PreSpikingErrorProb must be N x 1 column vector')
assert((nnz(PreSpikingErrorProb>=0)==N & nnz(PreSpikingErrorProb<2*f*(1-f))==N),'Every elements in PreSpikingErrorProb must be in the [0 2*f*(1-f)) range')
assert((PostSpikingErrorProb>=0 & PostSpikingErrorProb<2*f*(1-f)),'PostSpikingErrorProb must be in the [0 2*f*(1-f)) range')

p1 = PreSpikingErrorProb/2/(1-f);
p2 = (1-f)*p1/f;
fout = f;
Ninh = round(N*finh);
X1 = (1 - p2*ones(1,m)).*X + (p1*ones(1,m)).*(1-X) ;
X2 = (p2 * ones(1,m)).*(1 - p2 * ones(1,m)).*X + (p1 * ones(1,m)).*(1 - p1 * ones(1,m)).*(1-X);
C = sqrt(2)*( erfinv(1-PostSpikingErrorProb/fout)*Xp + erfinv(1-PostSpikingErrorProb/(1-fout))*(1-Xp));

delta = 10^-8;
opts = optimoptions(@fmincon,'Display','off','Algorithm','interior-point',...
    'SpecifyObjectiveGradient',false,'StepTolerance',delta,...
    'TolCon',delta,'TolX',delta,'TolFun',delta,'MaxIter',3000,...
    'MaxFunctionEvaluations',3000000);

g=[-ones(1,Ninh),ones(1,N-Ninh)];
fun = @(x) [zeros(1,N),ones(1,m)]*x;
AA = [-diag(g),zeros(N,m)];%sign constrainted
BB = [zeros(m,N),-eye(m,m)];
A = [AA;BB];
b = zeros(N+m,1);
Aeq = [g,zeros(1,m)];
beq = N*w;

nonlcon = @(x) nonlincon(x,X,X1,X2,g,Xp,C,betaInt,betaSyn);
x0 = [rand(N,1)*w;w*ones(m,1)];

[SV,~,~] = fmincon(fun,x0,A,b,Aeq,beq,[],[],nonlcon,opts);
W = SV(1:N);
flearned = mean(SV(N+1:end)<10^-5);
end

function [y,yeq] = nonlincon(x,X,X1,X2,g,Xp,C,betaInt,betaSyn)
[N,m] = size(X);
J = x(1:N);
epsilon = x(N+1:end);
y = C./N.*sqrt(N*betaInt^2 + betaSyn*(J.*g')'*X + sum(( J.^2 * ones(1,m)).*X2)) - (2*Xp-1).*( J'*X1/N - 1) - epsilon';
yeq = [];
end


