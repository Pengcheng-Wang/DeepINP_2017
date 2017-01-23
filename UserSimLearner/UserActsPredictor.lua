--
-- User: pwang8
-- Date: 1/23/17
-- Time: 12:43 PM
-- Implement a classification problem to predict user's next action in CI
-- user simulation model. Definition of user's state features and actions
-- are in the CI ijcai document.
--

require 'torch'
require 'nn'
require 'nnx'
require 'optim'
local nninit = require 'nninit'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIUserActsPredictor = classic.class('UserActsPredictor')


function CIUserActsPredictor:_init(CIUserSimulator)
    local opt = lapp[[
       -s,--save          (default "logs")      subdirectory to save logs
       -n,--network       (default "")          reload pretrained network
       -m,--uapModel         (default "mlp")   type of model tor train: moe | mlp | linear
       -f,--full                                use the full dataset
       -p,--plot                                plot while training
       -o,--optimization  (default "SGD")       optimization: SGD | LBFGS
       -r,--learningRate  (default 0.05)        learning rate, for SGD only
       -b,--batchSize     (default 10)          batch size
       -m,--momentum      (default 0)           momentum, for SGD only
       -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
       --coefL1           (default 0)           L1 penalty on the weights
       --coefL2           (default 0)           L2 penalty on the weights
       -t,--threads       (default 4)           number of threads
    ]]

    -- threads
    torch.setnumthreads(opt.threads)

    ----------------------------------------------------------------------
    -- define model to train
    -- on the 10-class classification problem
    --
    classes = {}
    for i=1, CIUserSimulator.CIFr.usrActInd_end do classes[i] = i end
    local inputFeatureNum = CIUserSimulator.realUserDataStates[1]:size()[1]

    if opt.network == '' then
        -- define model to train
        self.model = nn.Sequential()

        if opt.uapModel == 'moe' then
            ------------------------------------------------------------
            -- mixture of experts
            ------------------------------------------------------------
            experts = nn.ConcatTable()
            local numOfExp = 4
            for i = 1, numOfExp do
                local expert = nn.Sequential()
                expert:add(nn.Linear(inputFeatureNum, 32))
                expert:add(nn.ReLU())
                expert:add(nn.Linear(32, 24))
                expert:add(nn.ReLU())
                expert:add(nn.Linear(24, #classes))
                expert:add(nn.LogSoftMax())
                experts:add(expert)
            end

            gater = nn.Sequential()
            gater:add(nn.Linear(inputFeatureNum, 24))
            gater:add(nn.Tanh())
            gater:add(nn.Linear(24, numOfExp))
            gater:add(nn.SoftMax())

            trunk = nn.ConcatTable()
            trunk:add(gater)
            trunk:add(experts)

            self.model:add(trunk)
            self.model:add(nn.MixtureTable())
            ------------------------------------------------------------

        elseif opt.uapModel == 'mlp' then
            ------------------------------------------------------------
            -- regular 2-layer MLP
            ------------------------------------------------------------
            self.model:add(nn.Reshape(inputFeatureNum))
            self.model:add(nn.Linear(inputFeatureNum, 32))
            self.model:add(nn.ReLU())
            self.model:add(nn.Linear(32, 24))
            self.model:add(nn.ReLU())
            self.model:add(nn.Linear(24, #classes))
            self.model:add(nn.LogSoftMax())
            ------------------------------------------------------------

        elseif opt.uapModel == 'linear' then
            ------------------------------------------------------------
            -- simple linear model: logistic regression
            ------------------------------------------------------------
            self.model:add(nn.Reshape(inputFeatureNum))
            self.model:add(nn.Linear(inputFeatureNum, #classes))
            self.model:add(nn.LogSoftMax())
            ------------------------------------------------------------

        else
            print('Unknown model type')
            cmd:text()
            error()
        end

        -- params init
        local uapLinearLayers = self.model:findModules('nn.Linear')
        for l = 1, #uapLinearLayers do
            uapLinearLayers[l]:init('weight', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)}):init('bias', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)})
        end
    else
        print('<trainer> reloading previously trained network')
        self.model = torch.load(opt.network)
    end

    -- retrieve parameters and gradients
    self.uapParam, self.uapDParam = self.model:getParameters()

--    -- verbose
--    print(self.model)

    ----------------------------------------------------------------------
    -- loss function: negative log-likelihood
    --
    self.uapCriterion = nn.ClassNLLCriterion()
end


return CIUserActsPredictor
