local CIFileReader = require 'file_reader'
local CIUserSimulator = require 'UserSimulator'
local CIUserActsPredictor = require 'UserSimLearner/UserActsPredictor'
local CIUserScorePredictor = require 'UserSimLearner/UserScorePredictor'
local CIUserBehaviorGenerator = require 'UserSimLearner/UserBehaviorGenerator'

opt = lapp[[
       --trType         (default "sc")           training type : sc (score) | ac (action) | bg (behavior generation)
       -s,--save          (default "upplogs")      subdirectory to save logs
       -n,--network       (default "")          reload pretrained network
       -m,--uppModel         (default "lstm")   type of model tor train: moe | mlp | linear | lstm
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
       -g,--gpu_id        (default 0)          gpu device id, 0 for using cpu
       --prepro           (default "std")       input state feature preprocessing: rsc | std
       --lstmHd           (default 192)          lstm hidden layer size
       --lstmHist         (default 6)           lstm hist length
       --ubgDir           (default "ubgModel")  directory storing uap and usp models
       --uapFile          (default "uap.t7")          file storing userActsPredictor model
       --uspFile          (default "usp.t7")          file storing userScorePredictor model
    ]]

-- threads and default tensor type
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

-- Read CI trace and survey data files, and do validation
local fr = CIFileReader()
fr:evaluateTraceFile()
fr:evaluateSurveyData()

-- Construct CI user simulator model using real user data
local CIUserModel = CIUserSimulator(fr)

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
end



--print('@@', fr.traceData['100-0028'])
--print('#', #fr.data)
--print('#', #fr.data, ',', fr.data[1], '@@', fr.data[55][1], fr.data[55][81])