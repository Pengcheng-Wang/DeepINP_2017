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
require 'rnn'
local nninit = require 'nninit'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIUserActsPredictor = classic.class('UserActsPredictor')

function CIUserActsPredictor:_init(CIUserSimulator)
    opt = lapp[[
       -s,--save          (default "uaplogs")      subdirectory to save logs
       -n,--network       (default "")          reload pretrained network
       -m,--uapModel         (default "mlp")   type of model tor train: moe | mlp | linear | lstm
       -f,--full                                use the full dataset
       -p,--plot                                plot while training
       -o,--optimization  (default "rmsprop")       optimization: SGD | LBFGS | adam | rmsprop
       -r,--learningRate  (default 2e-3)        learning rate, for SGD only
       -b,--batchSize     (default 30)          batch size
       -m,--momentum      (default 0)           momentum, for SGD only
       -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
       --coefL1           (default 0)           L1 penalty on the weights
       --coefL2           (default 0)           L2 penalty on the weights
       -t,--threads       (default 4)           number of threads
       -g,--gpu_id        (default 0)          gpu device id, 0 for using cpu
       --prepro           (default "std")       input state feature preprocessing: rsc | std
       --lstmHd           (default 16)          lstm hidden layer size
       --lstmHist         (default 3)           lstm hist length
    ]]

    -- threads
    torch.setnumthreads(opt.threads)
    -- batch size?
    if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
        error('LBFGS should not be used with small mini-batches; 1000 is recommended')
    end

    self.ciUserSimulator = CIUserSimulator
    ----------------------------------------------------------------------
    -- define model to train
    -- on the 10-class classification problem
    --
    classes = {}
    for i=1, CIUserSimulator.CIFr.usrActInd_end do classes[i] = i end
    self.inputFeatureNum = CIUserSimulator.realUserDataStates[1]:size()[1]

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
                expert:add(nn.Linear(self.inputFeatureNum, 32))
                expert:add(nn.ReLU())
                expert:add(nn.Linear(32, 24))
                expert:add(nn.ReLU())
                expert:add(nn.Linear(24, #classes))
                expert:add(nn.LogSoftMax())
                experts:add(expert)
            end

            gater = nn.Sequential()
            gater:add(nn.Linear(self.inputFeatureNum, 24))
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
            self.model:add(nn.Reshape(self.inputFeatureNum))
            self.model:add(nn.Linear(self.inputFeatureNum, 32))
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
            self.model:add(nn.Reshape(self.inputFeatureNum))
            self.model:add(nn.Linear(self.inputFeatureNum, #classes))
            self.model:add(nn.LogSoftMax())
            ------------------------------------------------------------

        elseif opt.uapModel == 'lstm' then
            ------------------------------------------------------------
            -- lstm
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            local lstm = nn.FastLSTM(self.inputFeatureNum, opt.lstmHd, opt.lstmHist) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
            lstm.i2g:init({'bias', {{3*opt.lstmHd+1, 4*opt.lstmHd}}}, nninit.constant, 1)
            lstm:remember('both')
            self.model:add(lstm)
            self.model:add(nn.NormStabilizer())
            self.model:add(nn.Linear(opt.lstmHd, #classes))
            self.model:add(nn.LogSoftMax())
            self.model = nn.Sequencer(self.model)
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

--    -- verbose
--    print(self.model)

    ----------------------------------------------------------------------
    -- loss function: negative log-likelihood
    --
    self.uapCriterion = nn.ClassNLLCriterion()
    if opt.uapModel == 'lstm' then
        self.uapCriterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
    end

    self.trainEpoch = 1
    -- this matrix records the current confusion across classes
    self.uapConfusion = optim.ConfusionMatrix(classes)

    -- log results to files
    self.uapTrainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
    self.uapTestLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

    ----------------------------------------------------------------------
    --- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
    ---
    if opt.gpu_id > 0 then
        local ok, cunn = pcall(require, 'cunn')
        local ok2, cutorch = pcall(require, 'cutorch')
        if not ok then print('package cunn not found!') end
        if not ok2 then print('package cutorch not found!') end
        if ok and ok2 then
            print('using CUDA on GPU ' .. opt.gpu_id .. '...')
            cutorch.setDevice(opt.gpu_id)
--            cutorch.manualSeed(opt.seed)
            --- set up cuda nn
            self.model = self.model:cuda()
            self.uapCriterion = self.uapCriterion:cuda()
        else
            print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
            print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
            print('Falling back on CPU mode')
            opt.gpu_id = 0 -- overwrite user setting
        end
    end

    ----------------------------------------------------------------------
    --- Prepare data for lstm
    ---
    self.rnnRealUserDataStates = {}
    self.rnnRealUserDataActs = {}
    if opt.uapModel == 'lstm' then
        local realUserDataEndLines = {}
        for i=1, #self.ciUserSimulator.realUserDataStartLines - 1 do
            realUserDataEndLines[i] = self.ciUserSimulator.realUserDataStartLines[i+1] - 1
        end
        realUserDataEndLines[#self.ciUserSimulator.realUserDataStartLines] = #self.ciUserSimulator.realUserDataStates

        local indSeqHead = 1
        local indSeqTail = opt.lstmHist
        local indUserSeq = 1    -- user id ptr. Use this to get the tail of each trajectory
        while indSeqTail <= #self.ciUserSimulator.realUserDataStates do
            if indSeqTail <= realUserDataEndLines[indUserSeq] then
                self.rnnRealUserDataStates[#self.rnnRealUserDataStates + 1] = {}
                self.rnnRealUserDataActs[#self.rnnRealUserDataActs + 1] = {}
                for i=1, opt.lstmHist do
                    self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = self.ciUserSimulator.realUserDataStates[indSeqHead+i-1]
                    self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i] = self.ciUserSimulator.realUserDataActs[indSeqHead+i-1]
                end
                indSeqHead = indSeqHead + 1
                indSeqTail = indSeqTail + 1
            else
                indUserSeq = indUserSeq + 1 -- next user's records
                indSeqHead = self.ciUserSimulator.realUserDataStartLines[indUserSeq]
                indSeqTail = indSeqHead + opt.lstmHist - 1
            end
        end
        -- There are in total 15509 sequences if histLen is 3. 14707 if histLen is 5. 15108 if histLen is 4. 15911 if histLen is 2.
    end


    -- retrieve parameters and gradients
    -- have to put these lines here below the gpu setting
    self.uapParam, self.uapDParam = self.model:getParameters()
end


-- training function
function CIUserActsPredictor:trainOneEpoch()
    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. self.trainEpoch .. ' [batchSize = ' .. opt.batchSize .. ']')
    local inputs
    local targets
    local t = 1
    local lstmLengthPerBatch = math.ceil(#self.ciUserSimulator.realUserDataStates / opt.batchSize)
    local lstmIter = 1  -- lstm iterate for each squence starts from this value
    local epochDone = false
    while not epochDone do
        if opt.uapModel ~= 'lstm' then
            -- create mini batch
            inputs = torch.Tensor(opt.batchSize, self.inputFeatureNum)
            targets = torch.Tensor(opt.batchSize)
            local k = 1
            for i = t, math.min(t+opt.batchSize-1, #self.ciUserSimulator.realUserDataStates) do
                -- load new sample
                local input = self.ciUserSimulator.realUserDataStates[i]    -- :clone() -- if preprocess is called, clone is not needed, I believe
                -- need do preprocess for input features
                input = self.ciUserSimulator:preprocessUserStateData(input, opt.prepro)
                local target = self.ciUserSimulator.realUserDataActs[i]
                inputs[k] = input
                targets[k] = target
                k = k + 1
            end

            -- at the end of dataset, if it could not be divided into full batch
            if k ~= opt.batchSize + 1 then
                while k <= opt.batchSize do
                    local randInd = torch.random(1, #self.ciUserSimulator.realUserDataStates)
                    inputs[k] = self.ciUserSimulator:preprocessUserStateData(self.ciUserSimulator.realUserDataStates[randInd], opt.prepro)
                    targets[k] = self.ciUserSimulator.realUserDataActs[randInd]
                    k = k + 1
                end
            end

            t = t + opt.batchSize
            if t > #self.ciUserSimulator.realUserDataStates then
                epochDone = true
            end

            if opt.gpu_id > 0 then
                inputs = inputs:cuda()
                targets = targets:cuda()
            end

        else
            -- lstm
            inputs = {}
            targets = {}
            local k
            for j = 1, opt.lstmHist do
                inputs[j] = torch.Tensor(opt.batchSize, self.inputFeatureNum)
                targets[j] = torch.Tensor(opt.batchSize)
                k = 1
                for i = lstmIter, math.min(lstmIter+opt.batchSize-1, #self.rnnRealUserDataStates) do
                    local input = self.rnnRealUserDataStates[i][j]
                    input = self.ciUserSimulator:preprocessUserStateData(input, opt.prepro)
                    local target = self.rnnRealUserDataActs[i][j]
                    inputs[j][k] = input
                    targets[j][k] = target
                    k = k + 1
                end
            end

            -- at the end of dataset, if it could not be divided into full batch
            if k ~= opt.batchSize + 1 then
                while k <= opt.batchSize do
                    local randInd = torch.random(1, #self.rnnRealUserDataStates)
                    for j = 1, opt.lstmHist do
                        local input = self.rnnRealUserDataStates[randInd][j]
                        input = self.ciUserSimulator:preprocessUserStateData(input, opt.prepro)
                        local target = self.rnnRealUserDataActs[randInd][j]
                        inputs[j][k] = input:clone()
                        targets[j][k] = target:clone()
                    end
                    k = k + 1
                end
            end

            lstmIter = lstmIter + opt.batchSize
            if lstmIter > #self.rnnRealUserDataStates then
                epochDone = true
            end

            if opt.gpu_id > 0 then
                for _,v in pairs(inputs) do
                    v = v:cuda()
                end
                for _,v in pairs(targets) do
                    v = v:cuda()
                end
            end

        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= self.uapParam then
                self.uapParam:copy(x)
            end

            -- reset gradients
            self.uapDParam:zero()

            -- evaluate function for complete mini batch
            local outputs = self.model:forward(inputs)
            local f = self.uapCriterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = self.uapCriterion:backward(outputs, targets)
            self.model:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
                -- locals:
                local norm,sign= torch.norm,torch.sign

                -- Loss:
                f = f + opt.coefL1 * norm(self.uapParam,1)
                f = f + opt.coefL2 * norm(self.uapParam,2)^2/2

                -- Gradients:
                self.uapDParam:add( sign(self.uapParam):mul(opt.coefL1) + self.uapParam:clone():mul(opt.coefL2) )
            end

            -- update self.uapConfusion
            if opt.uapModel == 'lstm' then
                for j = 1, opt.lstmHist do
                    for i = 1,opt.batchSize do
                        self.uapConfusion:add(outputs[j][i], targets[j][i])
                    end
                end
            else
                for i = 1,opt.batchSize do
                    self.uapConfusion:add(outputs[i], targets[i])
                end
            end

            -- return f and df/dX
            return f, self.uapDParam
        end

        self.model:training()
        if opt.uapModel == 'lstm' then
            self.model:forget()
        end

        -- optimize on current mini-batch
        if opt.optimization == 'LBFGS' then

            -- Perform LBFGS step:
            lbfgsState = lbfgsState or {
                maxIter = opt.maxIter,
                lineSearch = optim.lswolfe
            }
            optim.lbfgs(feval, self.uapParam, lbfgsState)

            -- disp report:
            print('LBFGS step')
            print(' - progress in batch: ' .. t .. '/' .. #self.ciUserSimulator.realUserDataStates)
            print(' - nb of iterations: ' .. lbfgsState.nIter)
            print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

        elseif opt.optimization == 'SGD' then

            -- Perform SGD step:
            sgdState = sgdState or {
                learningRate = opt.learningRate,
                momentum = opt.momentum,
                learningRateDecay = 5e-7
            }
            optim.sgd(feval, self.uapParam, sgdState)

            -- disp progress
            if opt.uapModel ~= 'lstm' then
                xlua.progress(t, #self.ciUserSimulator.realUserDataStates)
            else
                xlua.progress(lstmIter, #self.rnnRealUserDataStates)
            end


        elseif opt.optimization == 'adam' then

            -- Perform Adam step:
            adamState = adamState or {
                learningRate = opt.learningRate,
                learningRateDecay = 5e-7
            }
            optim.adam(feval, self.uapParam, adamState)

            -- disp progress
            if opt.uapModel ~= 'lstm' then
                xlua.progress(t, #self.ciUserSimulator.realUserDataStates)
            else
                xlua.progress(lstmIter, #self.rnnRealUserDataStates)
            end

        elseif opt.optimization == 'rmsprop' then

            -- Perform Adam step:
            rmspropState = rmspropState or {
                learningRate = opt.learningRate
            }
            optim.rmsprop(feval, self.uapParam, rmspropState)

            -- disp progress
            if opt.uapModel ~= 'lstm' then
                xlua.progress(t, #self.ciUserSimulator.realUserDataStates)
            else
                xlua.progress(lstmIter, #self.rnnRealUserDataStates)
            end

        else
            error('unknown optimization method')
        end
    end

    -- time taken
    time = sys.clock() - time
    time = time / #self.ciUserSimulator.realUserDataStates
    print("<trainer> time to learn 1 epoch = " .. (time*1000) .. 'ms')

    -- print self.uapConfusion matrix
--    print(self.uapConfusion)
    self.uapConfusion:updateValids()
    local confMtxStr = 'average row correct: ' .. (self.uapConfusion.averageValid*100) .. '% \n' ..
            'average rowUcol correct (VOC measure): ' .. (self.uapConfusion.averageUnionValid*100) .. '% \n' ..
            ' + global correct: ' .. (self.uapConfusion.totalValid*100) .. '%'
    print(confMtxStr)
    self.uapTrainLogger:add{['% mean class accuracy (train set)'] = self.uapConfusion.totalValid * 100}
    self.uapConfusion:zero()

    -- save/log current net
    local filename = paths.concat(opt.save, 'uap.t7')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, self.model)

    -- next epoch
    self.trainEpoch = self.trainEpoch + 1
end


return CIUserActsPredictor
