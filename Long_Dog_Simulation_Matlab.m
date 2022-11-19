run('Robot_assembly_DataFile.m');
open_system('Long_Dog_Simulation.slx')
obsInfo = rlNumericSpec([29 1],...
    'LowerLimit',[-115 -156 -100 -100 -151 -120 -100 -100 -150 70 -100 -100 94 70 -100 -100 -2*pi -2*pi -2*pi -2*pi -1000 -1000 -1000 -100 -10 -50 -100 -100 -100]',...
    'UpperLimit',[-70 -96 100 100 -91 -70 100 100 -90 115 100 100 154 115 100 100 2*pi 2*pi 2*pi 2*pi 1000 1000 1000 100 100 50 100 100 100]');
obsInfo.Name = 'observations';
numObservations = obsInfo.Dimension(1);
actInfo = rlNumericSpec([8 1],'LowerLimit',-1,'UpperLimit',1);
numActions = actInfo.Dimension(1);
env = rlSimulinkEnv('Long_Dog_Simulation','Long_Dog_Simulation/RL Agent',obsInfo,actInfo);
Ts = 0.05;
Tf = 2;
rng(0)
statePath = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(280,'Name','CriticStateFC1')
    reluLayer('Name','CriticRelu1')
    fullyConnectedLayer(240,'Name','CriticStateFC')
    fullyConnectedLayer(200,'Name','CriticStateFC2')
    ];
actionPath = [
    featureInputLayer(numActions,'Normalization','none','Name','Action')
    fullyConnectedLayer(200,'Name','CriticActionFC1')];
commonPath = [
    additionLayer(2,'Name','add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1,'Name','CriticOutput')];
criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
%figure
%plot(criticNetwork)
criticOpts = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlQValueRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},criticOpts);
actorNetwork = [
    featureInputLayer(numObservations,'Normalization','none','Name','State')
    fullyConnectedLayer(200, 'Name','actorFC1')
    reluLayer('Name', 'actorRelu1')
    fullyConnectedLayer(120, 'Name','actorFC2')
    reluLayer('Name', 'ActorRelu2')
    fullyConnectedLayer(60, 'Name','actorFC3')
    reluLayer('Name', 'ActorRelu3')
    fullyConnectedLayer(numActions,'Name','ActionT')
    tanhLayer('Name','Action')
    ];
actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
actor = rlDeterministicActorRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'State'},'Action',{'Action'},actorOptions);
agentOptions = rlDDPGAgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.85;
agentOptions.MiniBatchSize = 300;
agentOptions.ExperienceBufferLength = 1e6;
agentOptions.TargetSmoothFactor = 1e-3;
agentOptions.NoiseOptions.MeanAttractionConstant = 0.15;
agentOptions.NoiseOptions.Variance = 0.1;
agent = rlDDPGAgent(actor,critic,agentOptions);
maxEpisodes = 8000;
maxSteps = floor(Tf/Ts);  
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxEpisodes,...
    'MaxStepsPerEpisode',maxSteps,...
    'ScoreAveragingWindowLength',250,...
    'Verbose',true,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',10000,...                   
    'SaveAgentCriteria','EpisodeReward',... 
    'SaveAgentValue',200);
doTraining = 0;

if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else 
    % Load the pretrained agent for the example.
    load('DOG1.mat', 'agent')
end

simOpts = rlSimulationOptions('MaxSteps',maxSteps,'StopOnError','on');
experiences = sim(env,agent,simOpts);
%save("DOG2.mat","agent")

