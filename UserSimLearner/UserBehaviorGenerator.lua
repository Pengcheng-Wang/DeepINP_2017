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
    self.CIUSim = CIUserSimulator
    self.CIUap = CIUserActsPred
    self.CIUsp = CIUserScorePred
    self.opt = opt

    if opt.uppModel == 'lstm' then
        local userStates = torch.Tensor(CIUserSimulator.userStateFeatureCnt):fill(0)
        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 1] = 1
        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 2] = 1
        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 3] = 9

        local prepStates = CIUserSimulator:preprocessUserStateData(torch.Tensor(CIUserSimulator.userStateFeatureCnt):fill(0), opt.prepro)
        local inPrepStates = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
        inPrepStates[1] = prepStates:clone()
        local tabPrepStates = {}
        for i=1, 5 do
            tabPrepStates[i] = inPrepStates:clone()
        end
        local lstTimeState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
        lstTimeState[1] = CIUserSimulator:preprocessUserStateData(userStates, opt.prepro)
        tabPrepStates[6] = lstTimeState

        local nll_acts = self.userActsPred:forward(tabPrepStates)
        print('@@@', nll_acts[6]:squeeze(), '\n$#$', torch.exp(nll_acts[6]:squeeze()))
        lp, ain = torch.max(nll_acts[6]:squeeze(), 1)
        print('##', lp, ain)

        local countScope = 0 --4
        local st = torch.Tensor(15):fill(0)
        for k, v in ipairs(CIUserSimulator.realUserDataStartLines) do
            st[CIUserSimulator.realUserDataActs[v+countScope]] = st[CIUserSimulator.realUserDataActs[v+countScope]] +1 -- check act dist at each x-th time step
        end
        print('Act count', st)

        self.userActsPred:forget()
        self.userScorePred:forget()
        local tltCnt = 0
        local crcActCnt = 0
        local crcRewCnt = 0
        local userInd = 1
        local earlyTotAct = torch.Tensor(opt.lstmHist+81):fill(1e-6)
        local earlyCrcAct = torch.Tensor(opt.lstmHist+81):fill(0)
        local firstActDist = torch.Tensor(15):fill(0)
--        for i=1, #CIUserActsPred.rnnRealUserDataStates do
--            local userState = CIUserActsPred.rnnRealUserDataStates[i]
--            local userAct = CIUserActsPred.rnnRealUserDataActs[i]
--            local userRew = CIUserScorePred.rnnRealUserDataRewards[i]
--
--            local tabState = {}
--            for j=1, opt.lstmHist do
--                local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
--                prepUserState[1] = CIUserSimulator:preprocessUserStateData(userState[j], opt.prepro)
--                tabState[j] = prepUserState:clone()
--            end
--
--            local nll_acts = self.userActsPred:forward(tabState)
--            lp, ain = torch.max(nll_acts[opt.lstmHist]:squeeze(), 1)
--            if i == CIUserActsPred.rnnRealUserDataStarts[userInd]+countScope then firstActDist[ain[1]] = firstActDist[ain[1]]+1 end   -- check act dist at each x-th time step
----            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end
--            lpy, lps = torch.sort(nll_acts[opt.lstmHist]:squeeze(), 1, true)
--            local crtExt = false
--            local smpLen = 1
--            for l=1, smpLen do
--                if lps[l] == userAct[opt.lstmHist] then
--                    crcActCnt = crcActCnt + 1
--                    crtExt = true
--                end
--            end
--
--            local indDiff = i - CIUserActsPred.rnnRealUserDataStarts[userInd]
--            if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserActsPred.rnnRealUserDataEnds[userInd] then
--                if crtExt then  --ain[1] == userAct[opt.lstmHist] then
--                    earlyCrcAct[indDiff+1] = earlyCrcAct[indDiff+1] + 1
--                end
--            earlyTotAct[indDiff+1] = earlyTotAct[indDiff+1] + 1
--            end
--            if i == CIUserActsPred.rnnRealUserDataEnds[userInd] then
--                userInd = userInd+1
--            end
--
--            local nll_rewards = self.userScorePred:forward(tabState)
--            lp, rin = torch.max(nll_rewards[opt.lstmHist]:squeeze(), 1)
--            if userAct[opt.lstmHist] == CIUserSimulator.CIFr.usrActInd_end and rin[1] == userRew[opt.lstmHist] then crcRewCnt = crcRewCnt + 1 end
--
--            tltCnt = tltCnt + 1
--
----            if userAct[opt.lstmHist] == CIUserSimulator.CIFr.usrActInd_end then
--                self.userActsPred:forget()
--                self.userScorePred:forget()
----            end
--
--        end

--        print('1st act: ', firstActDist)
--        print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserActsPred.rnnRealUserDataEnds, torch.cdiv(earlyCrcAct, earlyTotAct))

    else
--        print('This part of the evaluation code does not have to be correct. I have not carefully checked it.')
--        local userStates = torch.Tensor(CIUserSimulator.userStateFeatureCnt):fill(0)
--        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 1] = 0
--        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 2] = 5
--        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 3] = 9
--
--        local prepStates = CIUserSimulator:preprocessUserStateData(userStates, opt.prepro)
--        local inPrepStates = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
--        inPrepStates[1] = prepStates:clone()
--
--        local nll_acts = self.userActsPred:forward(inPrepStates)
--        lp, ain = torch.max(nll_acts[1]:squeeze(), 1)
--        print('##', lp, ain, nll_acts[1]:squeeze())
--
--        local st = torch.Tensor(15):fill(0)
--        for k, v in ipairs(CIUserSimulator.realUserDataStartLines) do
--            st[CIUserSimulator.realUserDataActs[v]] = st[CIUserSimulator.realUserDataActs[v]] +1
--        end
--        print('Act count\n', st)
--
--        self.userActsPred:forget()
----        self.userScorePred:forget()
--        local tltCnt = 0
--        local crcActCnt = 0
--        local crcRewCnt = 0
--        local userInd = 1
--        local earlyTotAct = 0
--        local earlyCrcAct = 0
--        for i=1, #CIUserSimulator.realUserDataStates do
--            local userState = CIUserSimulator:preprocessUserStateData(CIUserSimulator.realUserDataStates[i], opt.prepro)
--            local userAct = CIUserSimulator.realUserDataActs[i]
--            local userRew = CIUserSimulator.realUserDataRewards[i]
--
--            local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
--            prepUserState[1] = userState:clone()
--
--            local nll_acts = self.userActsPred:forward(prepUserState)
--            lp, ain = torch.max(nll_acts[1]:squeeze(), 1)
--            if ain[1] == userAct then crcActCnt = crcActCnt + 1 end
--
--            if i == CIUserSimulator.realUserDataEndLines[userInd] then
--                userInd = userInd+1
--            elseif i == CIUserSimulator.realUserDataStartLines[userInd] and i <= CIUserSimulator.realUserDataEndLines[userInd] then
--                if ain[1] == userAct then
--                    earlyCrcAct = earlyCrcAct + 1
--                end
--                earlyTotAct = earlyTotAct + 1
--            end
--
----            local nll_rewards = self.userScorePred:forward(prepUserState)
----            lp, rin = torch.max(nll_rewards[1]:squeeze(), 1)
----            if userAct == CIUserSimulator.CIFr.usrActInd_end and rin[1] == userRew then crcRewCnt = crcRewCnt + 1 end
--
--            tltCnt = tltCnt + 1
--
--            if userAct == CIUserSimulator.CIFr.usrActInd_end then
--                self.userActsPred:forget()
----                self.userScorePred:forget()
--            end
--
--        end
--
--        print('###', crcActCnt/tltCnt, crcRewCnt/402, earlyCrcAct/earlyTotAct)
--
    end
end

--- This function generates one trajectory from the simulated user model
--- The 1st action, together with user's survey data, right now is sample
--- from real data. The 2nd to 4th action is sampled from action predictoin
--- model, with propotion to the predicted probability. All other actions are
--- directly sampled from simulated model from the highest ranked action.
function CIUserBehaviorPredictor:sampleOneTraj()
    -- Right now, only lstm based action/score prediction model is supported
--    print('@#@#@', self.CIUap.rnnRealUserDataStarts, '@@', self.CIUSim.realUserDataStartLines)
    local realDataStartsCnt = #self.CIUap.rnnRealUserDataStarts
    local rndStartInd
    --- randomly select one human user's record whose 1st action cannot be ending action
    repeat
        rndStartInd = torch.random(1, realDataStartsCnt)
    until self.CIUap.rnnRealUserDataActs[self.CIUap.rnnRealUserDataStarts[rndStartInd]][self.opt.lstmHist] ~= self.CIUSim.CIFr.usrActInd_end

    --- Get this user's state record at the 1st time stpe. This process means we sampel
    --  user's 1st action and survey data from human user's records. Then we use our prediction
    --  model to estimate user's future ations.
    local curRnnStatesRaw = self.CIUap.rnnRealUserDataStates[self.CIUap.rnnRealUserDataStarts[rndStartInd]]     -- sample the 1st state
    local curRnnAct = self.CIUap.rnnRealUserDataActs[self.CIUap.rnnRealUserDataStarts[rndStartInd]][self.opt.lstmHist] -- sample the 1st action (at last time step)
    local adpTriggered = false
    local adpType = 0  -- should range from 1 to 4

    local tabRnnStateRaw = {}   -- raw state value, used for updating future states according to current actions
    for j=1, opt.lstmHist do
        local sinStepUserState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
        sinStepUserState[1] = curRnnStatesRaw[j]
        tabRnnStateRaw[j] = sinStepUserState:clone()
    end

    adpTriggered, adpType = self.CIUSim:isAdpTriggered(tabRnnStateRaw[self.opt.lstmHist], curRnnAct)    -- Todo: pwang8. Should fake some adp action later for testing
print(adpTriggered, adpType) os.exit()
    local tabRnnStatePrep = {}  -- states after preprocessing
    for j=1, opt.lstmHist do
        local prepSinStepState = torch.Tensor(1, self.CIUSim.userStateFeatureCnt)
        prepSinStepState[1] = self.CIUSim:preprocessUserStateData(tabRnnStateRaw[j][1], self.opt.prepro)
        tabRnnStatePrep[j] = prepSinStepState:clone()
    end

    local nll_acts = self.userActsPred:forward(tabRnnStatePrep)[self.opt.lstmHist]:squeeze() -- get act choice output for last time step
    local timeStepCnt = 2
    local curPredAct

lpy, lps = torch.sort(nll_acts[opt.lstmHist]:squeeze(), 1, true)

    repeat

    until true

--    local curRnnStatesRaw = self.CIUap.rnnRealUserDataStates[self.CIUap.rnnRealUserDataStarts[rndStartInd]+1]
--    for k,v in ipairs(curRnnStatesRaw) do print('#@#', k, 'sta:', v) end
--    local curRnnActs = self.CIUap.rnnRealUserDataActs[self.CIUap.rnnRealUserDataStarts[rndStartInd]+1]
--    for k,v in ipairs(curRnnActs) do print('^^&', k, 'act:', v) end
end

return CIUserBehaviorPredictor
