--
-- Created by IntelliJ IDEA.
-- User: pwang8
-- Date: 1/31/17
-- Time: 11:08 PM
-- This script aims at creating one script implementing the rlenvs APIs.
-- This script is modified based on UserBehaviorGenerator.lua
--

require 'torch'
require 'nn'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIFileReader = require 'file_reader'
local CIUserSimulator = require 'UserSimulator'
local CIUserActsPredictor = require 'UserSimLearner/UserActsPredictor'
local CIUserScorePredictor = require 'UserSimLearner/UserScorePredictor'

local CIUserSimEnv = classic.class('CIUserSimEnv')

function CIUserSimEnv:_init(opt)

    -- Read CI trace and survey data files, and do validation
    local fr = CIFileReader()
    fr:evaluateTraceFile()
    fr:evaluateSurveyData()

    -- Construct CI user simulator model using real user data
    local CIUserModel = CIUserSimulator(fr)
    local CIUserActsPred = CIUserActsPredictor(CIUserModel, opt)
    local CIUserScorePred = CIUserScorePredictor(CIUserModel, opt)

    self.userActsPred = torch.load(paths.concat(opt.ubgDir , opt.uapFile))
    self.userScorePred = torch.load(paths.concat(opt.ubgDir , opt.uspFile))
    self.userActsPred:evaluate()
    self.userScorePred:evaluate()
    self.CIUSim = CIUserModel
    self.CIUap = CIUserActsPred
    self.CIUsp = CIUserScorePred
    self.opt = opt

    --- Data generation related variables
    self.realDataStartsCnt = #self.CIUap.rnnRealUserDataStarts     -- count the number of real users
    self.rndStartInd = 1            -- Attention: self.CIUap.rnnRealUserDataStarts contains data only from training set.

    self.curRnnStatesRaw = self.CIUap.rnnRealUserDataStates[self.CIUap.rnnRealUserDataStarts[self.rndStartInd]]     -- sample the 1st state
    self.curRnnUserAct = self.CIUap.rnnRealUserDataActs[self.CIUap.rnnRealUserDataStarts[self.rndStartInd]][self.opt.lstmHist] -- sample the 1st action (at last time step)
    self.adpTriggered = false
    self.adpType = 0  -- valid type value should range from 1 to 4
    self.rlStateRaw = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d
    self.rlStatePrep = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d
    self.nextSingleStepStateRaw = nil

    self.tabRnnStateRaw = {}   -- raw state value, used for updating future states according to current actions. This is state for user act/score prediction nn, not for rl
    for j=1, self.opt.lstmHist do
        local sinStepUserState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
        sinStepUserState[1] = self.curRnnStatesRaw[j]
        self.tabRnnStateRaw[j] = sinStepUserState:clone()
    end
    self.tabRnnStatePrep = {}
    self.timeStepCnt = 1

end

------------------------------------------------
--- Create APIs following the format of rlenvs
--  All the following code is used by the RL script

--  1 state returned, of type 'int', of dimensionality 1 x self.size x self.size, between 0 and 1
function CIUserSimEnv:getStateSpec()
    return {'real', {1, 1, self.CIUSim.userStateFeatureCnt}, {0, 3}}    -- not sure about threshold of values, not guaranteed
end

-- 1 action required, of type 'int', of dimensionality 1, between 1 and 10
function CIUserSimEnv:getActionSpec()
    return {'int', 1, {1, 10}}
end


--- This function calculates and sets self.curRnnUserAct, which is the predicted
--  user action according to current tabRnnStatePrep value
function CIUserSimEnv:_calcUserAct()
    -- Pick an action using the action prediction model
    self.userActsPred:forget()
    local nll_acts = self.userActsPred:forward(self.tabRnnStatePrep)[self.opt.lstmHist]:squeeze() -- get act choice output for last time step

    --        print('Action choice likelihood Next time step:\n', torch.exp(nll_acts))
    local lpy   -- log likelihood value
    local lps   -- sorted index in desendent-order
    lpy, lps = torch.sort(nll_acts, 1, true)
    lpy = torch.exp(lpy)
    lpy = torch.cumsum(lpy)
    local actSampleLen = self.opt.actSmpLen
    lpy = torch.div(lpy, lpy[actSampleLen])
    local greedySmpThres = 0.6

    if self.timeStepCnt == 2 then
        greedySmpThres = 0.1
    elseif self.timeStepCnt == 3 then
        greedySmpThres = 0.3
    elseif self.timeStepCnt == 4 then
        greedySmpThres = 0.5
    end

    self.curRnnUserAct = lps[1]  -- the action result given by the action predictor
    if torch.uniform() > greedySmpThres then
        -- sample according to classifier output
        local rndActPick = torch.uniform()
        for i=1, actSampleLen do
            if rndActPick <= lpy[i] then
                self.curRnnUserAct = lps[i]
                break
            end
        end
    end

    return self.curRnnUserAct
end


function CIUserSimEnv:_updateRnnStatePrep()
    -- states after preprocessing
    for j=1, self.opt.lstmHist do
        local prepSinStepState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
        prepSinStepState[1] = self.CIUSim:preprocessUserStateData(self.tabRnnStateRaw[j][1], self.opt.prepro)
        self.tabRnnStatePrep[j] = prepSinStepState:clone()
    end
end


function CIUserSimEnv:start()
    local valid = false

    while not valid do
        --- randomly select one human user's record whose 1st action cannot be ending action
        repeat
            self.rndStartInd = torch.random(1, self.realDataStartsCnt)
        until self.CIUap.rnnRealUserDataActs[self.CIUap.rnnRealUserDataStarts[self.rndStartInd]][self.opt.lstmHist] ~= self.CIUSim.CIFr.usrActInd_end

        --- Get this user's state record at the 1st time stpe. This process means we sample
        --  user's 1st action and survey data from human user's records. Then we use our prediction
        --  model to estimate user's future ations.
        self.curRnnStatesRaw = self.CIUap.rnnRealUserDataStates[self.CIUap.rnnRealUserDataStarts[self.rndStartInd]]     -- sample the 1st state
        self.curRnnUserAct = self.CIUap.rnnRealUserDataActs[self.CIUap.rnnRealUserDataStarts[self.rndStartInd]][self.opt.lstmHist] -- sample the 1st action (at last time step)
        self.adpTriggered = false
        self.adpType = 0  -- valid type value should range from 1 to 4
        self.rlStateRaw = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d
        self.rlStatePrep = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d

        self.tabRnnStateRaw = {}   -- raw state value, used for updating future states according to current actions. This is state for user act/score prediction nn, not for rl
        for j=1, self.opt.lstmHist do
            self.sinStepUserState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
            self.sinStepUserState[1] = self.curRnnStatesRaw[j]
            self.tabRnnStateRaw[j] = self.sinStepUserState:clone()
        end
        self.tabRnnStatePrep = {}
        self.timeStepCnt = 1

        --        print(self.timeStepCnt, 'time step state:') for k,v in ipairs(self.tabRnnStateRaw) do print(k,v) end
        --        print(self.timeStepCnt, 'time step act:', self.curRnnUserAct)

        -- When user ap/sp state and action were given, check if adaptation could be triggered
        self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.tabRnnStateRaw[self.opt.lstmHist], self.curRnnUserAct)

        while not self.adpTriggered and self.curRnnUserAct ~= self.CIUSim.CIFr.usrActInd_end do
            -- apply user's action onto raw state representation
            -- This is the state representation for next single time step
            self.nextSingleStepStateRaw = self.tabRnnStateRaw[self.opt.lstmHist]:clone()
            self.CIUSim:applyUserActOnState(self.nextSingleStepStateRaw, self.curRnnUserAct)
            -- print('--- Next single step rnn state raw', self.nextSingleStepStateRaw)

            -- reconstruct rnn state table for next time step
            for j=1, self.opt.lstmHist-1 do
                self.tabRnnStateRaw[j] = self.tabRnnStateRaw[j+1]:clone()
            end
            self.tabRnnStateRaw[self.opt.lstmHist] = self.nextSingleStepStateRaw:clone()

            self.timeStepCnt = self.timeStepCnt + 1
            self:_updateRnnStatePrep()
            self:_calcUserAct()

            --            print(self.timeStepCnt, 'time step state:') for k,v in ipairs(self.tabRnnStateRaw) do print(k,v) end
            --            print(self.timeStepCnt, 'time step act:', self.curRnnUserAct)

            -- When user ap/sp state and action were given, check if adaptation could be triggered
            self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.tabRnnStateRaw[self.opt.lstmHist], self.curRnnUserAct)

        end -- end of while

        -- Attention: we guarantee that the ending user action will not trigger adaptation
        if self.adpTriggered then
            --            print('--- Adp triggered')

            self.rlStateRaw[1][1] = self.tabRnnStateRaw[self.opt.lstmHist][1] -- copy the last time step RAW state representation. Clone() is not needed.

            -- Need to add the user action's effect on rl state
            self.CIUSim:applyUserActOnState(self.rlStateRaw, self.curRnnUserAct)

            -- Need to add the user action's effect on rl state
            self.rlStatePrep[1][1] = self.CIUSim:preprocessUserStateData(self.rlStateRaw[1][1], self.opt.prepro)   -- do preprocessing before sending back to RL
            --            print('--- After apply user act, rl state:', self.rlStateRaw[1][1])
            --            print('--- Prep rl state', self.rlStatePrep[1][1])
            -- Should get action choice from the RL agent here

            valid = true    -- not necessary
            return self.rlStatePrep, self.adpType

        else    -- self.curRnnUserAct == self.CIUSim.CIFr.usrActInd_end
        --            print('Regenerate user behavior trajectory from start!')
        valid = false   -- not necessary
        end

    end
end


function CIUserSimEnv:step(adpAct)
    assert(adpAct >= self.CIUSim.CIFr.ciAdpActRanges[self.adpType][1] and adpAct <= self.CIUSim.CIFr.ciAdpActRanges[self.adpType][2])

    self.nextSingleStepStateRaw = self.tabRnnStateRaw[self.opt.lstmHist]:clone()
    self.CIUSim:applyAdpActOnState(self.nextSingleStepStateRaw, self.adpType, adpAct)

    repeat

        self.CIUSim:applyUserActOnState(self.nextSingleStepStateRaw, self.curRnnUserAct)
        -- reconstruct rnn state table for next time step
        for j=1, self.opt.lstmHist-1 do
            self.tabRnnStateRaw[j] = self.tabRnnStateRaw[j+1]:clone()
        end
        self.tabRnnStateRaw[self.opt.lstmHist] = self.nextSingleStepStateRaw:clone()

        self.timeStepCnt = self.timeStepCnt + 1
        self:_updateRnnStatePrep()
        self:_calcUserAct()

        --        print(self.timeStepCnt, 'time step state:') for k,v in ipairs(self.tabRnnStateRaw) do print(k,v) end
        --        print(self.timeStepCnt, 'time step act:', self.curRnnUserAct)

        -- When user ap/sp state and action were given, check if adaptation could be triggered
        self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.tabRnnStateRaw[self.opt.lstmHist], self.curRnnUserAct)

        self.nextSingleStepStateRaw = self.tabRnnStateRaw[self.opt.lstmHist]:clone()

    until self.adpTriggered or self.curRnnUserAct == self.CIUSim.CIFr.usrActInd_end

    -- Attention: we guarantee that the ending user action will not trigger adaptation
    if self.adpTriggered then
        --        print('--- Adp triggered')

        self.rlStateRaw[1][1] = self.tabRnnStateRaw[self.opt.lstmHist][1] -- copy the last time step RAW state representation. Clone() is not needed.

        -- Need to add the user action's effect on rl state
        self.CIUSim:applyUserActOnState(self.rlStateRaw, self.curRnnUserAct)

        -- Need to add the user action's effect on rl state
        self.rlStatePrep[1][1] = self.CIUSim:preprocessUserStateData(self.rlStateRaw[1][1], self.opt.prepro)   -- do preprocessing before sending back to RL
        --        print('--- After apply user act, rl state:', self.rlStateRaw[1][1])
        --        print('--- Prep rl state', self.rlStatePrep[1][1])
        -- Should get action choice from the RL agent here

        return 0, self.rlStatePrep, false, self.adpType

    else    -- self.curRnnUserAct == self.CIUSim.CIFr.usrActInd_end
    self.rlStateRaw[1][1] = self.tabRnnStateRaw[self.opt.lstmHist][1] -- copy the last time step RAW state representation. Clone() is not needed.
    -- Does not need to apply an ending user action. It will not change state representation.
    -- Need to add the user action's effect on rl state
    self.rlStatePrep[1][1] = self.CIUSim:preprocessUserStateData(self.rlStateRaw[1][1], self.opt.prepro)   -- do preprocessing before sending back to RL

    local nll_rewards = self.userScorePred:forward(self.tabRnnStatePrep)
    lp, rin = torch.max(nll_rewards[self.opt.lstmHist]:squeeze(), 1)
    --        print('Predicted reward:', rin[1], torch.exp(nll_rewards[self.opt.lstmHist]:squeeze()))
    --        print('--====== End')
    local score = 1
    if rin[1] == 2 then score = -1 end

    return score, self.rlStatePrep, true, 0
    end

end

--- Set up the trianing mode for this rl environment
function CIUserSimEnv:training()
end

--- Set up the evaluate mode for this rl environment
function CIUserSimEnv:evaluate()
end

--- Returns (RGB) display of screen, Fake function for CIUserSimEnv
function CIUserSimEnv:getDisplay()
    return torch.repeatTensor(torch.div(self.rlStateRaw, 50), 3, 1, 1)
end

--- RGB screen of size self.size x self.size
function CIUserSimEnv:getDisplaySpec()
    return {'real', {3, 1, self.CIUSim.userStateFeatureCnt}, {0, 1}}
end

return CIUserSimEnv

