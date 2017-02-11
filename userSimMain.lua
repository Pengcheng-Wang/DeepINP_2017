local CIFileReader = require 'file_reader'
local CIUserSimulator = require 'UserSimulator'
local CIUserActsPredictor = require 'UserSimLearner/UserActsPredictor'
local CIUserScorePredictor = require 'UserSimLearner/UserScorePredictor'
local CIUserBehaviorGenerator = require 'UserSimLearner/UserBehaviorGenerator'
local CIUserBehaviorGenEvaluator = require 'UserSimLearner/UserBehaviorGenEvaluator'

opt = lapp[[
       --trType         (default "rl")           training type : sc (score) | ac (action) | bg (behavior generation) | rl (implement rlenvs API) | ev (evaluation of act/score prediction)
       -s,--save          (default "upplogs")      subdirectory to save logs
       -n,--ciunet       (default "")          reload pretrained CI user simulation network
       -m,--uppModel         (default "lstm")   type of model to train: moe | mlp | linear | lstm
       -f,--full                                use the full dataset
       -p,--plot                                plot while training
       -o,--optimization  (default "adam")       optimization: SGD | LBFGS | adam | rmsprop
       -r,--learningRate  (default 2e-4)        learning rate, for SGD only
       -b,--batchSize     (default 30)          batch size
       -m,--momentum      (default 0)           momentum, for SGD only
       -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
       --coefL1           (default 0)           L1 penalty on the weights
       --coefL2           (default 0)           L2 penalty on the weights
       -t,--threads       (default 4)           number of threads
       -g,--gpu        (default 0)          gpu device id, 0 for using cpu
       --prepro           (default "std")       input state feature preprocessing: rsc | std
       --lstmHd           (default 192)          lstm hidden layer size
       --lstmHist         (default 6)           lstm hist length
       --ubgDir           (default "ubgModel")  directory storing uap and usp models
       --uapFile          (default "uap.t7")          file storing userActsPredictor model
       --uspFile          (default "usp.t7")          file storing userScorePredictor model
       --actSmpLen        (default 6)           The sampling candidate list length for user action generation
       --ciuTType         (default "train")     Training or testing for use sim model train | test
       --actEvaScp        (default 1)           The action selection range in prediction evaluation calculation
    ]]

-- threads and default tensor type
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

-- Read CI trace and survey data files, and do validation
local fr = CIFileReader()
fr:evaluateTraceFile()
fr:evaluateSurveyData()

-- Construct CI user simulator model using real user data
local CIUserModel = CIUserSimulator(fr, opt)

if opt.trType == 'sc' then
    local CIUserScorePred = CIUserScorePredictor(CIUserModel, opt)
    for i=1, 2e5 do
        CIUserScorePred:trainOneEpoch()
    end
elseif opt.trType == 'ac' then
    local CIUserActsPred = CIUserActsPredictor(CIUserModel, opt)
    for i=1, 2e5 do
        CIUserActsPred:trainOneEpoch()
    end
elseif opt.trType == 'bg' then
    local CIUserActsPred = CIUserActsPredictor(CIUserModel, opt)
    local CIUserScorePred = CIUserScorePredictor(CIUserModel, opt)
    local CIUserBehaviorGen = CIUserBehaviorGenerator(CIUserModel, CIUserActsPred, CIUserScorePred, opt)
    local scoreStat = {0, 0}
    local totalTrajLength = 0
    local totalLengthEachType = {0, 0}
    local rnds = 10000
    for i=1, rnds do
        local sc, tl
        sc, tl = CIUserBehaviorGen:sampleOneTraj()
        scoreStat[sc] = scoreStat[sc] + 1
        totalTrajLength = totalTrajLength + tl
        totalLengthEachType[sc] = totalLengthEachType[sc] + tl
    end
    print('Score dist:', scoreStat, 'Avg length:', totalTrajLength/rnds, 'Avg length of each nlg type 1:', totalLengthEachType[1]/scoreStat[1],
        'Avg length of each nlg type 2:', totalLengthEachType[2]/scoreStat[2])
    local numTrans = 0
    for uid, urec in pairs(CIUserModel.realUserRLActs) do
        numTrans = numTrans + #urec - 1
    end
    print('Number of transitions in data set is', numTrans)
elseif opt.trType == 'rl' then
    local CIUserActsPred = CIUserActsPredictor(CIUserModel, opt)
    local CIUserScorePred = CIUserScorePredictor(CIUserModel, opt)
    local CIUserBehaviorGen = CIUserBehaviorGenerator(CIUserModel, CIUserActsPred, CIUserScorePred, opt)

    local numTrans = 0
    local numTransType = {0, 0, 0, 0}
    for uid, urec in pairs(CIUserModel.realUserRLActs) do
        numTrans = numTrans + #urec - 1
        for k,v in pairs(CIUserModel.realUserRLTypes[uid]) do
            if v ~= 0 then
                numTransType[v] = numTransType[v] + 1
            end
        end
    end
    print('Number of transitions in data set is', numTrans, 'type: ', numTransType)

    local gens = 10000
    local adpTotLen = 0
    local adpLenType = {0, 0, 0, 0}
    for i=1, gens do
        print('iter', i)
        local obv, score, term, adpType
        local adpCnt = 0
        term = false
        obv, adpType = CIUserBehaviorGen:start()
        print('^### Outside in main\n state:', obv, '\n type:', adpType)
        while not term do
            adpLenType[adpType] = adpLenType[adpType] + 1
            adpCnt = adpCnt + 1
            local rndAdpAct = torch.random(fr.ciAdpActRanges[adpType][1], fr.ciAdpActRanges[adpType][2])
--            print('^--- Adaptation type', adpType, 'Random act choice: ', rndAdpAct)
            score, obv, term, adpType = CIUserBehaviorGen:step(rndAdpAct)
--            print('^### Outside in main\n state:', obv, '\n type:', adpType, '\n score:', score, ',term:', term)
        end
        adpTotLen = adpTotLen + adpCnt
    end
    print('In user behaviro generation in', gens, 'times, avg len:', adpTotLen/gens, 'Adp types:', adpLenType)
elseif opt.trType == 'ev' then
    local CIUserActsPred = CIUserActsPredictor(CIUserModel, opt)
    local CIUserScorePred = CIUserScorePredictor(CIUserModel, opt)
    local CIUserBehaviorGen = CIUserBehaviorGenEvaluator(CIUserModel, CIUserActsPred, CIUserScorePred, opt)
end



--print('@@', fr.traceData['100-0028'])
--print('#', #fr.data)
--print('#', #fr.data, ',', fr.data[1], '@@', fr.data[55][1], fr.data[55][81])