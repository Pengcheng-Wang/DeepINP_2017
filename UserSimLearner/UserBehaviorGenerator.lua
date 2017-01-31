--
-- User: pwang8
-- Date: 1/26/17
-- Time: 3:21 PM
-- Generate simulated user interaction data using the predicted
-- user action and user nlg score.
-- Right now, this script is only guaranteed to work correctly
-- with lstm generated user action and score.
--

require 'torch'
require 'nn'
--require 'nnx'
--require 'optim'
--require 'rnn'
--local nninit = require 'nninit'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIUserBehaviorPredictor = classic.class('UserBehaviorPredictor')

function CIUserBehaviorPredictor:_init(CIUserSimulator, CIUserActsPred, CIUserScorePred, opt)
    print('#', paths.concat(opt.ubgDir , opt.uapFile))
    self.userActsPred = torch.load(paths.concat(opt.ubgDir , opt.uapFile))
    self.userScorePred = torch.load(paths.concat(opt.ubgDir , opt.uspFile))
    self.userActsPred:evaluate()
    self.userScorePred:evaluate()
    self.CIUSim = CIUserSimulator
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
    for j=1, opt.lstmHist do
        local sinStepUserState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
        sinStepUserState[1] = self.curRnnStatesRaw[j]
        self.tabRnnStateRaw[j] = sinStepUserState:clone()
    end
    self.tabRnnStatePrep = {}
    self.timeStepCnt = 1

end

--- This function generates one trajectory from the simulated user model
--- The 1st action, together with user's survey data, right now is sample
--- from real data. The 2nd to 4th action is sampled from action predictoin
--- model, with propotion to the predicted probability. All other actions are
--- directly sampled from simulated model from the highest ranked action.
function CIUserBehaviorPredictor:sampleOneTraj()
    -- Right now, only lstm based action/score prediction model is supported
    local realDataStartsCnt = #self.CIUap.rnnRealUserDataStarts     -- count the number of real users
    local rndStartInd
    --- randomly select one human user's record whose 1st action cannot be ending action
    repeat
        rndStartInd = torch.random(1, realDataStartsCnt)
    until self.CIUap.rnnRealUserDataActs[self.CIUap.rnnRealUserDataStarts[rndStartInd]][self.opt.lstmHist] ~= self.CIUSim.CIFr.usrActInd_end

    --- Get this user's state record at the 1st time stpe. This process means we sample
    --  user's 1st action and survey data from human user's records. Then we use our prediction
    --  model to estimate user's future ations.
    local curRnnStatesRaw = self.CIUap.rnnRealUserDataStates[self.CIUap.rnnRealUserDataStarts[rndStartInd]]     -- sample the 1st state
    local curRnnUserAct = self.CIUap.rnnRealUserDataActs[self.CIUap.rnnRealUserDataStarts[rndStartInd]][self.opt.lstmHist] -- sample the 1st action (at last time step)
    local adpTriggered = false
    local adpType = 0  -- valid type value should range from 1 to 4
    local rlStateRaw = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d
    local rlStatePrep = torch.Tensor(1, 1, self.CIUSim.userStateFeatureCnt):fill(0)   -- this state should be 3d
    local nextSingleStepStateRaw

    local tabRnnStateRaw = {}   -- raw state value, used for updating future states according to current actions. This is state for user act/score prediction nn, not for rl
    for j=1, opt.lstmHist do
        local sinStepUserState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
        sinStepUserState[1] = curRnnStatesRaw[j]
        tabRnnStateRaw[j] = sinStepUserState:clone()
    end
    local tabRnnStatePrep = {}
    local timeStepCnt = 1

    repeat

        -- This is the state representation for next single time step
        nextSingleStepStateRaw = tabRnnStateRaw[self.opt.lstmHist]:clone()

--        print(timeStepCnt, 'time step state:') for k,v in ipairs(tabRnnStateRaw) do print(k,v) end
        print(timeStepCnt, 'time step act:', curRnnUserAct)

        -- When user ap/sp state and action were given, check if adaptation could be triggered
        adpTriggered, adpType = self.CIUSim:isAdpTriggered(tabRnnStateRaw[self.opt.lstmHist], curRnnUserAct)

        -- Attention: the state value for RL is not the same as it is for user action prediction
        if adpTriggered then
            print('--- Adp triggered')
            rlStateRaw[1][1] = tabRnnStateRaw[self.opt.lstmHist][1] -- copy the last time step RAW state representation. Clone() is not needed.

            -- Need to add the user action's effect on rl state
            self.CIUSim:applyUserActOnState(rlStateRaw, curRnnUserAct)
            rlStatePrep[1][1] = self.CIUSim:preprocessUserStateData(rlStateRaw[1][1], self.opt.prepro)   -- do preprocessing before sending back to RL
--            print('--- After apply user act, rl state:', rlStateRaw[1][1])
--            print('--- Prep rl state', rlStatePrep[1][1])
            -- Should get action choice from the RL agent here
            -- Right now, generate a fake RL-adp action
            local rndAdpAct = torch.random(self.CIUSim.CIFr.ciAdpActRanges[adpType][1], self.CIUSim.CIFr.ciAdpActRanges[adpType][2])
            print('--- Adaptation triggered for type', adpType, 'Random act choice: ', rndAdpAct)

            -- Apply rl adp action onto user's single time step state
            self.CIUSim:applyAdpActOnState(nextSingleStepStateRaw, adpType, rndAdpAct)
        end

        -- apply user's action onto raw state representation
        self.CIUSim:applyUserActOnState(nextSingleStepStateRaw, curRnnUserAct)
--        print('--- Next single step rnn state raw', nextSingleStepStateRaw)

        timeStepCnt = timeStepCnt + 1

        -- reconstruct rnn state table for next time step
        for j=1, opt.lstmHist-1 do
            tabRnnStateRaw[j] = tabRnnStateRaw[j+1]:clone()
        end
        tabRnnStateRaw[opt.lstmHist] = nextSingleStepStateRaw:clone()

        -- states after preprocessing
        for j=1, opt.lstmHist do
            local prepSinStepState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
            prepSinStepState[1] = self.CIUSim:preprocessUserStateData(tabRnnStateRaw[j][1], self.opt.prepro)
            tabRnnStatePrep[j] = prepSinStepState:clone()
        end

        -- Pick an action using the action prediction model
        self.userActsPred:forget()
        local nll_acts = self.userActsPred:forward(tabRnnStatePrep)[self.opt.lstmHist]:squeeze() -- get act choice output for last time step

--        print('Action choice likelihood Next time step:\n', torch.exp(nll_acts))
        local lpy   -- log likelihood value
        local lps   -- sorted index in desendent-order
        lpy, lps = torch.sort(nll_acts, 1, true)
        lpy = torch.exp(lpy)
        lpy = torch.cumsum(lpy)
        local actSampleLen = self.opt.actSmpLen
        lpy = torch.div(lpy, lpy[actSampleLen])
        local greedySmpThres = 0.75

        if timeStepCnt == 2 then
            greedySmpThres = 0.1
        elseif timeStepCnt == 3 then
            greedySmpThres = 0.3
        elseif timeStepCnt == 4 then
            greedySmpThres = 0.5
        end

        curRnnUserAct = lps[1]  -- the action result given by the action predictor
        if torch.uniform() > greedySmpThres then
            -- sample according to classifier output
            local rndActPick = torch.uniform()
            for i=1, actSampleLen do
                if rndActPick <= lpy[i] then
                    curRnnUserAct = lps[i]
                    break
                end
            end
        end
--        print('Choose action for next step:', curRnnUserAct)

    until curRnnUserAct == self.CIUSim.CIFr.usrActInd_end

--    print(timeStepCnt, 'time step state:') for k,v in ipairs(tabRnnStateRaw) do print(k,v) end
    print(timeStepCnt, 'time step act:', curRnnUserAct)

    -- Predict this student's score
    local nll_rewards = self.userScorePred:forward(tabRnnStatePrep)
    lp, rin = torch.max(nll_rewards[opt.lstmHist]:squeeze(), 1)
    print('Predicted reward:', rin[1], torch.exp(nll_rewards[opt.lstmHist]:squeeze()))

    return rin[1], timeStepCnt   -- return the predicted nlg classification
end


------------------------------------------------
--- Create APIs following the format of rlenvs
--  All the following code is used by the RL script


--  1 state returned, of type 'int', of dimensionality 1 x self.size x self.size, between 0 and 1
function CIUserBehaviorPredictor:getStateSpec()
    return {'int', {1, 1, self.CIUSim.userStateFeatureCnt}, {0, 10}}    -- not sure about threshold of values, not guaranteed
end


function CIUserBehaviorPredictor:start()
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
        for j=1, opt.lstmHist do
            self.sinStepUserState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
            self.sinStepUserState[1] = self.curRnnStatesRaw[j]
            self.tabRnnStateRaw[j] = self.sinStepUserState:clone()
        end
        self.tabRnnStatePrep = {}
        self.timeStepCnt = 1

        print(self.timeStepCnt, 'time step state:') for k,v in ipairs(self.tabRnnStateRaw) do print(k,v) end
        print(self.timeStepCnt, 'time step act:', self.curRnnUserAct)

        -- When user ap/sp state and action were given, check if adaptation could be triggered
        self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.tabRnnStateRaw[self.opt.lstmHist], self.curRnnUserAct)

        while not self.adpTriggered and self.curRnnUserAct ~= self.CIUSim.CIFr.usrActInd_end do
            -- apply user's action onto raw state representation
            -- This is the state representation for next single time step
            self.nextSingleStepStateRaw = self.tabRnnStateRaw[self.opt.lstmHist]:clone()
            self.CIUSim:applyUserActOnState(self.nextSingleStepStateRaw, self.curRnnUserAct)
            -- print('--- Next single step rnn state raw', self.nextSingleStepStateRaw)

            -- reconstruct rnn state table for next time step
            for j=1, opt.lstmHist-1 do
                self.tabRnnStateRaw[j] = self.tabRnnStateRaw[j+1]:clone()
            end
            self.tabRnnStateRaw[opt.lstmHist] = self.nextSingleStepStateRaw:clone()

            -- states after preprocessing
            for j=1, opt.lstmHist do
                local prepSinStepState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
                prepSinStepState[1] = self.CIUSim:preprocessUserStateData(self.tabRnnStateRaw[j][1], self.opt.prepro)
                self.tabRnnStatePrep[j] = prepSinStepState:clone()
            end

            self.timeStepCnt = self.timeStepCnt + 1

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
            local greedySmpThres = 0.75

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

            print(self.timeStepCnt, 'time step state:') for k,v in ipairs(self.tabRnnStateRaw) do print(k,v) end
            print(self.timeStepCnt, 'time step act:', self.curRnnUserAct)

            -- When user ap/sp state and action were given, check if adaptation could be triggered
            self.adpTriggered, self.adpType = self.CIUSim:isAdpTriggered(self.tabRnnStateRaw[self.opt.lstmHist], self.curRnnUserAct)

        end -- end of while

        -- Attention: we guarantee that the ending user action will not trigger adaptation
        if self.adpTriggered then
            print('--- Adp triggered')
            -- apply user's action onto raw state representation
            self.nextSingleStepStateRaw = self.tabRnnStateRaw[self.opt.lstmHist]:clone()
            self.CIUSim:applyUserActOnState(self.nextSingleStepStateRaw, self.curRnnUserAct)
            -- reconstruct rnn state table for next time step
            for j=1, opt.lstmHist-1 do
                self.tabRnnStateRaw[j] = self.tabRnnStateRaw[j+1]:clone()
            end
            self.tabRnnStateRaw[opt.lstmHist] = self.nextSingleStepStateRaw:clone()

            -- states after preprocessing
            for j=1, opt.lstmHist do
                local prepSinStepState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
                prepSinStepState[1] = self.CIUSim:preprocessUserStateData(self.tabRnnStateRaw[j][1], self.opt.prepro)
                self.tabRnnStatePrep[j] = prepSinStepState:clone()
            end

            self.rlStateRaw[1][1] = self.tabRnnStateRaw[self.opt.lstmHist][1] -- copy the last time step RAW state representation. Clone() is not needed.

            -- Need to add the user action's effect on rl state
            self.rlStatePrep[1][1] = self.tabRnnStatePrep[self.opt.lstmHist][1]   -- preprocess before sending back to RL
            print('--- After apply user act, rl state:', self.rlStateRaw[1][1])
            print('--- Prep rl state', self.rlStatePrep[1][1])
            -- Should get action choice from the RL agent here

            valid = true    -- not necessary
            return self.rlStatePrep, self.adpType
--            -- Apply rl adp action onto user's single time step state
--            self.CIUSim:applyAdpActOnState(self.nextSingleStepStateRaw, self.adpType, rndAdpAct)

        else    -- self.curRnnUserAct == self.CIUSim.CIFr.usrActInd_end
            print('Regenerate user behavior trajectory from start!')
            valid = false   -- not necessary
        end

    end
end


function CIUserBehaviorPredictor:step(adpAct)
    assert(adpAct >= self.CIUSim.CIFr.ciAdpActRanges[self.adpType][1] and adpAct <= self.CIUSim.CIFr.ciAdpActRanges[self.adpType][2])
    -- Todo:pwang8. Implement this step()
end

return CIUserBehaviorPredictor
