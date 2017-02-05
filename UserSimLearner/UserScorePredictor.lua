--
-- User: pwang8
-- Date: 1/25/17
-- Time: 11:06 AM
-- Implement a classification problem to predict user's final score (nlg)
-- in CI user simulation model.
-- Definition of user's state features and actions
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

local CIUserScorePredictor = classic.class('UserScorePredictor')

function CIUserScorePredictor:_init(CIUserSimulator, opt)

    -- batch size?
    if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
        error('LBFGS should not be used with small mini-batches; 1000 is recommended')
    end

    self.ciUserSimulator = CIUserSimulator
    self.opt = opt
    ----------------------------------------------------------------------
    -- define model to train
    -- on the 15-class classification problem
    --
    self.classes = {}
    for i=1,2 do self.classes[i] = i end
    self.inputFeatureNum = CIUserSimulator.realUserDataStates[1]:size()[1]

    if opt.ciunet == '' then
        -- define model to train
        self.model = nn.Sequential()

        if opt.uppModel == 'moe' then
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
                expert:add(nn.Linear(24, #self.classes))
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

        elseif opt.uppModel == 'mlp' then
            ------------------------------------------------------------
            -- regular 2-layer MLP
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            self.model:add(nn.Linear(self.inputFeatureNum, 32))
            self.model:add(nn.ReLU())
            self.model:add(nn.Linear(32, 24))
            self.model:add(nn.ReLU())
            self.model:add(nn.Linear(24, #self.classes))
            self.model:add(nn.LogSoftMax())
            ------------------------------------------------------------

        elseif opt.uppModel == 'linear' then
            ------------------------------------------------------------
            -- simple linear model: logistic regression
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            self.model:add(nn.Linear(self.inputFeatureNum, #self.classes))
            self.model:add(nn.LogSoftMax())
            ------------------------------------------------------------

        elseif opt.uppModel == 'lstm' then
            ------------------------------------------------------------
            -- lstm
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            local lstm = nn.FastLSTM(self.inputFeatureNum, opt.lstmHd, opt.lstmHist) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
            lstm.i2g:init({'bias', {{3*opt.lstmHd+1, 4*opt.lstmHd}}}, nninit.constant, 1)
            lstm:remember('both')
            self.model:add(lstm)
            self.model:add(nn.NormStabilizer())
            self.model:add(nn.Linear(opt.lstmHd, #self.classes))
            self.model:add(nn.LogSoftMax())
            self.model = nn.Sequencer(self.model)
            ------------------------------------------------------------

        else
            print('Unknown model type')
            cmd:text()
            error()
        end

        -- params init
        local uspLinearLayers = self.model:findModules('nn.Linear')
        for l = 1, #uspLinearLayers do
            uspLinearLayers[l]:init('weight', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)}):init('bias', nninit.kaiming, {dist = 'uniform', gain = 1/math.sqrt(3)})
        end
    else
        print('<trainer> reloading previously trained ciunet')
        self.model = torch.load(opt.ciunet)
    end

    --    -- verbose
    --    print(self.model)

    ----------------------------------------------------------------------
    -- loss function: negative log-likelihood
    --
    self.uspCriterion = nn.ClassNLLCriterion()
--    if opt.uppModel == 'lstm' then
--        self.uspCriterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
--    end
    -- I'm trying to use the last output from lstm as source for backprop, so use ClassNLLCriterion for lstm

    self.trainEpoch = 1
    -- this matrix records the current confusion across classes
    self.uspConfusion = optim.ConfusionMatrix(self.classes)

    -- log results to files
    self.uspTrainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
    self.uspTestLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

    ----------------------------------------------------------------------
    --- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
    ---
    if opt.gpu > 0 then
        local ok, cunn = pcall(require, 'cunn')
        local ok2, cutorch = pcall(require, 'cutorch')
        if not ok then print('package cunn not found!') end
        if not ok2 then print('package cutorch not found!') end
        if ok and ok2 then
            print('using CUDA on GPU ' .. opt.gpu .. '...')
            cutorch.setDevice(opt.gpu)
            --            cutorch.manualSeed(opt.seed)
            --- set up cuda nn
            self.model = self.model:cuda()
            self.uspCriterion = self.uspCriterion:cuda()
        else
            print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
            print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
            print('Falling back on CPU mode')
            opt.gpu = 0 -- overwrite user setting
        end
    end

    ----------------------------------------------------------------------
    --- Prepare data for lstm
    ---
    self.rnnRealUserDataStates = {}
    self.rnnRealUserDataRewards = {}
    self.rnnRealUserDataStarts = {}
    self.rnnRealUserDataEnds = {}
    self.rnnRealUserDataPad = torch.Tensor(#self.ciUserSimulator.realUserDataStartLines):fill(0)    -- indicating whether data has padding at head (should be padded)
    if opt.uppModel == 'lstm' then
        local indSeqHead = 1
        local indSeqTail = opt.lstmHist
        local indUserSeq = 1    -- user id ptr. Use this to get the tail of each trajectory
        while indSeqTail <= #self.ciUserSimulator.realUserDataStates do
            if self.rnnRealUserDataPad[indUserSeq] < 1 then
                for padi = opt.lstmHist-1, 1, -1 do
                    self.rnnRealUserDataStates[#self.rnnRealUserDataStates + 1] = {}
                    self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards + 1] = {}
                    for i=1, padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = torch.Tensor(self.ciUserSimulator.userStateFeatureCnt):fill(0) -- fill in all 0s states as dumb states
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i] = self.ciUserSimulator.realUserDataRewards[indSeqHead]  -- duplicate the 1st user action for padded states
                    end
                    for i=1, opt.lstmHist-padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i+padi] = self.ciUserSimulator.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i+padi] = self.ciUserSimulator.realUserDataRewards[indSeqHead+i-1]
                    end
                    if padi == opt.lstmHist-1 then
                        self.rnnRealUserDataStarts[#self.rnnRealUserDataStarts+1] = #self.rnnRealUserDataStates     -- This is the start of a user's record
                    end
                    if indSeqHead+(opt.lstmHist-padi)-1 == self.ciUserSimulator.realUserDataEndLines[indUserSeq] then
                        self.rnnRealUserDataPad[indUserSeq] = 1
                        break   -- if padding tail is going to outrange this user record's tail, break
                    end
                end
                self.rnnRealUserDataPad[indUserSeq] = 1
            else
                if indSeqTail <= self.ciUserSimulator.realUserDataEndLines[indUserSeq] then
                    self.rnnRealUserDataStates[#self.rnnRealUserDataStates + 1] = {}
                    self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards + 1] = {}
                    for i=1, opt.lstmHist do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = self.ciUserSimulator.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i] = self.ciUserSimulator.realUserDataRewards[indSeqHead+i-1]
                    end
                    indSeqHead = indSeqHead + 1
                    indSeqTail = indSeqTail + 1
                else
                    self.rnnRealUserDataEnds[#self.rnnRealUserDataEnds+1] = #self.rnnRealUserDataStates     -- This is the end of a user's record
                    indUserSeq = indUserSeq + 1 -- next user's records
                    indSeqHead = self.ciUserSimulator.realUserDataStartLines[indUserSeq]
                    indSeqTail = indSeqHead + opt.lstmHist - 1
                end
            end
        end
        self.rnnRealUserDataEnds[#self.rnnRealUserDataEnds+1] = #self.rnnRealUserDataStates     -- Set the end of the last user's record
        -- There are in total 15509 sequences if histLen is 3. 14707 if histLen is 5. 15108 if histLen is 4. 15911 if histLen is 2.
    end

    -- retrieve parameters and gradients
    -- have to put these lines here below the gpu setting
    self.uspParam, self.uspDParam = self.model:getParameters()
end


-- training function
function CIUserScorePredictor:trainOneEpoch()
    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. self.trainEpoch .. ' [batchSize = ' .. self.opt.batchSize .. ']')
    local inputs
    local targets
    local t = 1
    local lstmLengthPerBatch = math.ceil(#self.ciUserSimulator.realUserDataStates / self.opt.batchSize)
    local lstmIter = 1  -- lstm iterate for each squence starts from this value
    local epochDone = false
    while not epochDone do
        if self.opt.uppModel ~= 'lstm' then
            -- create mini batch
            inputs = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
            targets = torch.Tensor(self.opt.batchSize)
            local k = 1
            for i = t, math.min(t+self.opt.batchSize-1, #self.ciUserSimulator.realUserDataStates) do
                -- load new sample
                local input = self.ciUserSimulator.realUserDataStates[i]    -- :clone() -- if preprocess is called, clone is not needed, I believe
                -- need do preprocess for input features
                input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                local target = self.ciUserSimulator.realUserDataRewards[i]
                inputs[k] = input
                targets[k] = target
                k = k + 1
            end

            -- at the end of dataset, if it could not be divided into full batch
            if k ~= self.opt.batchSize + 1 then
                while k <= self.opt.batchSize do
                    local randInd = torch.random(1, #self.ciUserSimulator.realUserDataStates)
                    inputs[k] = self.ciUserSimulator:preprocessUserStateData(self.ciUserSimulator.realUserDataStates[randInd], self.opt.prepro)
                    targets[k] = self.ciUserSimulator.realUserDataRewards[randInd]
                    k = k + 1
                end
            end

            t = t + self.opt.batchSize
            if t > #self.ciUserSimulator.realUserDataStates then
                epochDone = true
            end

            if self.opt.gpu > 0 then
                inputs = inputs:cuda()
                targets = targets:cuda()
            end

        else
            -- lstm
            inputs = {}
            targets = {}
            local k
            for j = 1, self.opt.lstmHist do
                inputs[j] = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
                targets[j] = torch.Tensor(self.opt.batchSize)
                k = 1
                for i = lstmIter, math.min(lstmIter+self.opt.batchSize-1, #self.rnnRealUserDataStates) do
                    local input = self.rnnRealUserDataStates[i][j]
                    input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                    local target = self.rnnRealUserDataRewards[i][j]
                    inputs[j][k] = input
                    targets[j][k] = target
                    k = k + 1
                end
            end

            -- at the end of dataset, if it could not be divided into full batch
            if k ~= self.opt.batchSize + 1 then
                while k <= self.opt.batchSize do
                    local randInd = torch.random(1, #self.rnnRealUserDataStates)
                    for j = 1, self.opt.lstmHist do
                        local input = self.rnnRealUserDataStates[randInd][j]
                        input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                        local target = self.rnnRealUserDataRewards[randInd][j]
                        inputs[j][k] = input
                        targets[j][k] = target
                    end
                    k = k + 1
                end
            end

            lstmIter = lstmIter + self.opt.batchSize
            if lstmIter > #self.rnnRealUserDataStates then
                epochDone = true
            end

            if self.opt.gpu > 0 then
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
            if x ~= self.uspParam then
                self.uspParam:copy(x)
            end

            -- reset gradients
            self.uspDParam:zero()

            -- evaluate function for complete mini batch
            local outputs = self.model:forward(inputs)
            local f
            local df_do
            local df_do_finalStep
            local gradOutputsZeroed = {}

            if self.opt.uppModel == 'lstm' then
                f = self.uspCriterion:forward(outputs[self.opt.lstmHist], targets[self.opt.lstmHist])
                df_do_finalStep = self.uspCriterion:backward(outputs[self.opt.lstmHist], targets[self.opt.lstmHist])
                for step=1, self.opt.lstmHist do
                    gradOutputsZeroed[step] = torch.zeros(self.opt.batchSize, #self.classes)
                    if self.opt.gpu > 0 then
                        gradOutputsZeroed[step] = gradOutputsZeroed[step]:cuda()
                    end
                end
                gradOutputsZeroed[self.opt.lstmHist] = df_do_finalStep
                df_do = gradOutputsZeroed
            else
                f = self.uspCriterion:forward(outputs, targets)
                df_do = self.uspCriterion:backward(outputs, targets)
            end

            self.model:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if self.opt.coefL1 ~= 0 or self.opt.coefL2 ~= 0 then
                -- locals:
                local norm,sign= torch.norm,torch.sign

                -- Loss:
                f = f + self.opt.coefL1 * norm(self.uspParam,1)
                f = f + self.opt.coefL2 * norm(self.uspParam,2)^2/2

                -- Gradients:
                self.uspDParam:add( sign(self.uspParam):mul(self.opt.coefL1) + self.uspParam:clone():mul(self.opt.coefL2) )
            end

            -- update self.uspConfusion
            if self.opt.uppModel == 'lstm' then
--                for j = 1, self.opt.lstmHist do
--                    for i = 1,self.opt.batchSize do
--                        self.uspConfusion:add(outputs[j][i], targets[j][i])
--                    end
--                end
                for i = 1,self.opt.batchSize do
                    self.uspConfusion:add(outputs[self.opt.lstmHist][i], targets[self.opt.lstmHist][i])
                end
            else
                for i = 1,self.opt.batchSize do
                    self.uspConfusion:add(outputs[i], targets[i])
                end
            end

            -- return f and df/dX
            return f, self.uspDParam
        end

        self.model:training()
        if self.opt.uppModel == 'lstm' then
            self.model:forget()
        end

        -- optimize on current mini-batch
        if self.opt.optimization == 'LBFGS' then

            -- Perform LBFGS step:
            lbfgsState = lbfgsState or {
                maxIter = self.opt.maxIter,
                lineSearch = optim.lswolfe
            }
            optim.lbfgs(feval, self.uspParam, lbfgsState)

            -- disp report:
            print('LBFGS step')
            print(' - progress in batch: ' .. t .. '/' .. #self.ciUserSimulator.realUserDataStates)
            print(' - nb of iterations: ' .. lbfgsState.nIter)
            print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

        elseif self.opt.optimization == 'SGD' then

            -- Perform SGD step:
            sgdState = sgdState or {
                learningRate = self.opt.learningRate,
                momentum = self.opt.momentum,
                learningRateDecay = 5e-7
            }
            optim.sgd(feval, self.uspParam, sgdState)

            -- disp progress
            if self.opt.uppModel ~= 'lstm' then
                xlua.progress(t, #self.ciUserSimulator.realUserDataStates)
            else
                xlua.progress(lstmIter, #self.rnnRealUserDataStates)
            end


        elseif self.opt.optimization == 'adam' then

            -- Perform Adam step:
            adamState = adamState or {
                learningRate = self.opt.learningRate,
                learningRateDecay = 5e-7
            }
            optim.adam(feval, self.uspParam, adamState)

            -- disp progress
            if self.opt.uppModel ~= 'lstm' then
                xlua.progress(t, #self.ciUserSimulator.realUserDataStates)
            else
                xlua.progress(lstmIter, #self.rnnRealUserDataStates)
            end

        elseif self.opt.optimization == 'rmsprop' then

            -- Perform Adam step:
            rmspropState = rmspropState or {
                learningRate = self.opt.learningRate
            }
            optim.rmsprop(feval, self.uspParam, rmspropState)

            -- disp progress
            if self.opt.uppModel ~= 'lstm' then
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
--    time = time / #self.ciUserSimulator.realUserDataStates  -- This is not accurate, if lstm is used
    print("<trainer> time to learn 1 epoch = " .. (time*1000) .. 'ms')

    -- print self.uspConfusion matrix
    --    print(self.uspConfusion)
    self.uspConfusion:updateValids()
    local confMtxStr = 'average row correct: ' .. (self.uspConfusion.averageValid*100) .. '% \n' ..
            'average rowUcol correct (VOC measure): ' .. (self.uspConfusion.averageUnionValid*100) .. '% \n' ..
            ' + global correct: ' .. (self.uspConfusion.totalValid*100) .. '%'
    print(confMtxStr)
    self.uspTrainLogger:add{['% mean class accuracy (train set)'] = self.uspConfusion.totalValid * 100}


    -- save/log current net
    local filename = paths.concat(self.opt.save, 'usp.t7')
    os.execute('mkdir -p ' .. sys.dirname(filename))
--    if paths.filep(filename) then
--        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
--    end
    print('<trainer> saving ciunet to '..filename)
    torch.save(filename, self.model)

    if self.trainEpoch % 20 == 0 then
        filename = paths.concat(self.opt.save, string.format('%d', self.trainEpoch)..'_'..string.format('%.2f', self.uspConfusion.totalValid*100)..'usp.t7')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('<trainer> saving periodly trained ciunet to '..filename)
        torch.save(filename, self.model)
    end

    self.uspConfusion:zero()
    -- next epoch
    self.trainEpoch = self.trainEpoch + 1
end


return CIUserScorePredictor

