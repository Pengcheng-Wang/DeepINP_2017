--
-- Created by IntelliJ IDEA.
-- User: pwang8
-- Date: 5/6/17
-- Time: 5:23 PM
-- This script is modified from UserActsPredictor.lua and UserScorePredictor.lua
-- This script creates a NN model which combines user action and score prediction.
--
-- Todo: pwang8. May 6. Need to add an opt field to indicate whether using separate Action/Score models or unique model
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

local CIUserActScorePredictor = classic.class('UserActScorePredictor')

function CIUserActScorePredictor:_init(CIUserSimulator, opt)

    -- batch size?
    if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
        error('LBFGS should not be used with small mini-batches; 1000 is recommended')
    end

    self.ciUserSimulator = CIUserSimulator
    self.opt = opt
    ----------------------------------------------------------------------
    -- define model to train
    -- on the 15-class user action classification problem
    -- and 2-class user outcome(score) classification problem
    --
    classesActs = {}
    classesScores = {}
    for i=1, CIUserSimulator.CIFr.usrActInd_end do classesActs[i] = i end   -- set action classes
    for i=1,2 do classesScores[i] = i end   -- set score(outcome) classes
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

                -- The following code creates two output modules, with one module matches
                -- to user action prediction, and the other matches to user outcome(score) prediction
                mulOutConcatTab = nn.ConcatTable()
                actSeqNN = nn.Sequential()
                actSeqNN.add(nn.Linear(24, #classesActs))
                actSeqNN.add(nn.LogSoftMax())
                scoreSeqNN = nn.Sequential()
                scoreSeqNN.add(nn.Linear(24, #classesScores))
                scoreSeqNN.add(nn.LogSoftMax())
                mulOutConcatTab.add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
                mulOutConcatTab.add(scoreSeqNN) -- {act, outcome(score)}

                expert:add(mulOutConcatTab)
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

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            mulOutConcatTab = nn.ConcatTable()
            actSeqNN = nn.Sequential()
            actSeqNN.add(nn.Linear(24, #classesActs))
            actSeqNN.add(nn.LogSoftMax())
            scoreSeqNN = nn.Sequential()
            scoreSeqNN.add(nn.Linear(24, #classesScores))
            scoreSeqNN.add(nn.LogSoftMax())
            mulOutConcatTab.add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab.add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            ------------------------------------------------------------

        elseif opt.uppModel == 'linear' then
            ------------------------------------------------------------
            -- simple linear model: logistic regression
            ------------------------------------------------------------
            -- Attention: this implementation with ConcatTable does not have
            -- any differences from separate act/score prediction implementation.
            -- Because the two modules have no shared params at all.
            -- So, should not try to use it.
            self.model:add(nn.Reshape(self.inputFeatureNum))

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            mulOutConcatTab = nn.ConcatTable()
            actSeqNN = nn.Sequential()
            actSeqNN.add(nn.Linear(self.inputFeatureNum, #classesActs))
            actSeqNN.add(nn.LogSoftMax())
            scoreSeqNN = nn.Sequential()
            scoreSeqNN.add(nn.Linear(self.inputFeatureNum, #classesScores))
            scoreSeqNN.add(nn.LogSoftMax())
            mulOutConcatTab.add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab.add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
            ------------------------------------------------------------

        elseif opt.uppModel == 'lstm' then
            ------------------------------------------------------------
            -- lstm
            ------------------------------------------------------------
            self.model:add(nn.Reshape(self.inputFeatureNum))
            local lstm = nn.FastLSTM(self.inputFeatureNum, opt.lstmHd, opt.lstmBackProp) -- the 3rd param, [rho], the maximum amount of backpropagation steps to take back in time, default value is 9999
            lstm.i2g:init({'bias', {{3*opt.lstmHd+1, 4*opt.lstmHd}}}, nninit.constant, 1)
            lstm:remember('both')
            self.model:add(lstm)
            self.model:add(nn.NormStabilizer())

            -- The following code creates two output modules, with one module matches
            -- to user action prediction, and the other matches to user outcome(score) prediction
            mulOutConcatTab = nn.ConcatTable()
            actSeqNN = nn.Sequential()
            actSeqNN.add(nn.Linear(opt.lstmHd, #classesActs))
            actSeqNN.add(nn.LogSoftMax())
            scoreSeqNN = nn.Sequential()
            scoreSeqNN.add(nn.Linear(opt.lstmHd, #classesScores))
            scoreSeqNN.add(nn.LogSoftMax())
            mulOutConcatTab.add(actSeqNN)   -- should pay attention to the sequence of action and outcome prediction table
            mulOutConcatTab.add(scoreSeqNN) -- {act, outcome(score)}

            self.model:add(mulOutConcatTab)
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
        print('<trainer> reloading previously trained ciunet')
        self.model = torch.load(opt.ciunet)
    end

    --    -- verbose
    --    print(self.model)

    ----------------------------------------------------------------------
    -- loss function: negative log-likelihood
    --
    self.uaspPrlCriterion = nn.ParallelCriterion()
    self.uaspPrlCriterion.add(nn.ClassNLLCriterion())   -- action prediction loss function
    self.uaspPrlCriterion.add(nn.ClassNLLCriterion())   -- score (outcome) prediction loss function
--    self.uapCriterion = nn.ClassNLLCriterion()
--    self.uspCriterion = nn.ClassNLLCriterion()
    if opt.uppModel == 'lstm' then
        self.uaspPrlCriterion = nn.SequencerCriterion(self.uaspPrlCriterion)
    end

    self.trainEpoch = 1
    -- these matrices records the current confusion across classesActs and classesScores
    self.uapConfusion = optim.ConfusionMatrix(classesActs)
    self.uspConfusion = optim.ConfusionMatrix(classesScores)

    -- log results to files
    self.uaspTrainLogger = optim.Logger(paths.concat(opt.save, 'uaspTrain.log'))

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
            self.uapCriterion = self.uapCriterion:cuda()
            self.uspConfusion = self.uspConfusion:cuda()
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
    self.rnnRealUserDataActs = {}
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
                    self.rnnRealUserDataActs[#self.rnnRealUserDataActs + 1] = {}
                    self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards + 1] = {}
                    for i=1, padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = torch.Tensor(self.ciUserSimulator.userStateFeatureCnt):fill(0)
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i] = self.ciUserSimulator.realUserDataActs[indSeqHead]  -- duplicate the 1st user action for padded states
                        self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards][i] = self.ciUserSimulator.realUserDataRewards[indSeqHead]
                    end
                    for i=1, opt.lstmHist-padi do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i+padi] = self.ciUserSimulator.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i+padi] = self.ciUserSimulator.realUserDataActs[indSeqHead+i-1]
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
                    self.rnnRealUserDataActs[#self.rnnRealUserDataActs + 1] = {}
                    self.rnnRealUserDataRewards[#self.rnnRealUserDataRewards + 1] = {}
                    for i=1, opt.lstmHist do
                        self.rnnRealUserDataStates[#self.rnnRealUserDataStates][i] = self.ciUserSimulator.realUserDataStates[indSeqHead+i-1]
                        self.rnnRealUserDataActs[#self.rnnRealUserDataActs][i] = self.ciUserSimulator.realUserDataActs[indSeqHead+i-1]
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
    self.uaspParam, self.uaspDParam = self.model:getParameters()
end


-- training function
function CIUserActScorePredictor:trainOneEpoch()
    -- local vars
    local time = sys.clock()

    -- do one epoch
    print('<trainer> on training set:')
    print("<trainer> online epoch # " .. self.trainEpoch .. ' [batchSize = ' .. self.opt.batchSize .. ']')
    local inputs
    local targetsActScore
    local targetsAct
    local targetsScore
    local t = 1
    local lstmIter = 1  -- lstm iterate for each squence starts from this value
    local epochDone = false
    while not epochDone do
        if self.opt.uppModel ~= 'lstm' then
            -- create mini batch
            inputs = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
            targetsAct = torch.Tensor(self.opt.batchSize)
            targetsScore = torch.Tensor(self.opt.batchSize)
            local k = 1
            for i = t, math.min(t+self.opt.batchSize-1, #self.ciUserSimulator.realUserDataStates) do
                -- load new sample
                local input = self.ciUserSimulator.realUserDataStates[i]    -- :clone() -- if preprocess is called, clone is not needed, I believe
                -- need do preprocess for input features
                input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                local singleTargetAct = self.ciUserSimulator.realUserDataActs[i]
                local singleTargetScore = self.ciUserSimulator.realUserDataRewards[i]
                inputs[k] = input
                targetsAct[k] = singleTargetAct
                targetsScore[k] = singleTargetScore
                k = k + 1
            end

            -- at the end of dataset, if it could not be divided into full batch
            if k ~= self.opt.batchSize + 1 then
                while k <= self.opt.batchSize do
                    local randInd = torch.random(1, #self.ciUserSimulator.realUserDataStates)
                    inputs[k] = self.ciUserSimulator:preprocessUserStateData(self.ciUserSimulator.realUserDataStates[randInd], self.opt.prepro)
                    targets[k] = self.ciUserSimulator.realUserDataActs[randInd]
                    targetsAct[k] = self.ciUserSimulator.realUserDataActs[randInd]
                    targetsScore[k] = self.ciUserSimulator.realUserDataRewards[randInd]
                    k = k + 1
                end
            end

            t = t + self.opt.batchSize
            if t > #self.ciUserSimulator.realUserDataStates then
                epochDone = true
            end

            if self.opt.gpu > 0 then
                inputs = inputs:cuda()
                targetsAct = targetsAct:cuda()
                targetsScore = targetsScore:cuda()
            end

            targetsActScore = {targetsAct, targetsScore}

        else
            -- lstm
            inputs = {}
            targetsAct = {}
            targetsScore = {}
            local k
            for j = 1, self.opt.lstmHist do
                inputs[j] = torch.Tensor(self.opt.batchSize, self.inputFeatureNum)
                targetsAct[j] = torch.Tensor(self.opt.batchSize)
                targetsScore[j] = torch.Tensor(self.opt.batchSize)
                k = 1
                for i = lstmIter, math.min(lstmIter+self.opt.batchSize-1, #self.rnnRealUserDataStates) do
                    local input = self.rnnRealUserDataStates[i][j]
                    input = self.ciUserSimulator:preprocessUserStateData(input, self.opt.prepro)
                    local singleTargetAct = self.rnnRealUserDataActs[i][j]
                    local singleTargetScore = self.rnnRealUserDataRewards[i][j]
                    inputs[j][k] = input
                    targetsAct[j][k] = singleTargetAct
                    targetsScore[j][k] = singleTargetScore
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
                        local singleTargetAct = self.rnnRealUserDataActs[randInd][j]
                        local singleTargetScore = self.rnnRealUserDataRewards[randInd][j]
                        inputs[j][k] = input
                        targetsAct[j][k] = singleTargetAct
                        targetsScore[j][k] = singleTargetScore
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
                for _,v in pairs(targetsAct) do
                    v = v:cuda()
                end
                for _,v in pairs(targetsScore) do
                    v = v:cuda()
                end
            end

            targetsActScore = {targetsAct, targetsScore}

        end

        -- create closure to evaluate f(X) and df/dX
        local feval = function(x)
            -- just in case:
            collectgarbage()

            -- get new parameters
            if x ~= self.uaspParam then
                self.uaspParam:copy(x)
            end

            -- reset gradients
            self.uaspDParam:zero()

            -- evaluate function for complete mini batch
            local outputs = self.model:forward(inputs)
            local f = self.uapCriterion:forward(outputs, targets)

            -- estimate df/dW
            local df_do = self.uapCriterion:backward(outputs, targets)
            self.model:backward(inputs, df_do)

            -- penalties (L1 and L2):
            if self.opt.coefL1 ~= 0 or self.opt.coefL2 ~= 0 then
                -- locals:
                local norm,sign= torch.norm,torch.sign

                -- Loss:
                f = f + self.opt.coefL1 * norm(self.uaspParam,1)
                f = f + self.opt.coefL2 * norm(self.uaspParam,2)^2/2

                -- Gradients:
                self.uaspDParam:add( sign(self.uaspParam):mul(self.opt.coefL1) + self.uaspParam:clone():mul(self.opt.coefL2) )
            end

            -- update self.uapConfusion
            if self.opt.uppModel == 'lstm' then
                for j = 1, self.opt.lstmHist do
                    for i = 1,self.opt.batchSize do
                        self.uapConfusion:add(outputs[j][i], targets[j][i])
                    end
                end
            else
                for i = 1,self.opt.batchSize do
                    self.uapConfusion:add(outputs[i], targets[i])
                end
            end

            -- return f and df/dX
            return f, self.uaspDParam
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
            optim.lbfgs(feval, self.uaspParam, lbfgsState)

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
            optim.sgd(feval, self.uaspParam, sgdState)

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
            optim.adam(feval, self.uaspParam, adamState)

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
            optim.rmsprop(feval, self.uaspParam, rmspropState)

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
    --    time = time / #self.ciUserSimulator.realUserDataStates
    print("<trainer> time to learn 1 epoch = " .. (time*1000) .. 'ms')

    -- print self.uapConfusion matrix
    --    print(self.uapConfusion)
    self.uapConfusion:updateValids()
    local confMtxStr = 'average row correct: ' .. (self.uapConfusion.averageValid*100) .. '% \n' ..
            'average rowUcol correct (VOC measure): ' .. (self.uapConfusion.averageUnionValid*100) .. '% \n' ..
            ' + global correct: ' .. (self.uapConfusion.totalValid*100) .. '%'
    print(confMtxStr)
    self.uaspTrainLogger:add{['% mean class accuracy (train set)'] = self.uapConfusion.totalValid * 100}


    -- save/log current net
    local filename = paths.concat(self.opt.save, 'uap.t7')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    --    if paths.filep(filename) then
    --        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    --    end
    print('<trainer> saving ciunet to '..filename)
    torch.save(filename, self.model)

    if self.trainEpoch % 20 == 0 then
        filename = paths.concat(self.opt.save, string.format('%d', self.trainEpoch)..'_'..string.format('%.2f', self.uapConfusion.totalValid*100)..'uap.t7')
        os.execute('mkdir -p ' .. sys.dirname(filename))
        print('<trainer> saving periodly trained ciunet to '..filename)
        torch.save(filename, self.model)
    end

    self.uapConfusion:zero()
    -- next epoch
    self.trainEpoch = self.trainEpoch + 1
end


return CIUserActScorePredictor


