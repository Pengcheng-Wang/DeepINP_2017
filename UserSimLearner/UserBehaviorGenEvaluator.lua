--
-- User: pwang8
-- Date: 2/5/17
-- Time: 4:59 PM
-- This file is used for testing the performance of user action/score predictors' performance on test set.
-- And this script is modified from a pervious version of UserBehaviorGenerator.lua
--

require 'torch'
require 'nn'
local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIUserBehaviorGenEvaluator = classic.class('UserBehaviorGenEvaluator')

function CIUserBehaviorGenEvaluator:_init(CIUserSimulator, CIUserActsPred, CIUserScorePred, opt)
    -- The user action/score predictors for evaluation should be pre-trained, and loaded from files
    -- Also, the CIUserSimulator, CIUserActsPred, CIUserScorePred should be initialized using
    -- the test set.
    print('#', paths.concat(opt.ubgDir , opt.uapFile))
    self.userActsPred = torch.load(paths.concat(opt.ubgDir , opt.uapFile))
    self.userScorePred = torch.load(paths.concat(opt.ubgDir , opt.uspFile))
    self.userActsPred:evaluate()
    self.userScorePred:evaluate()
    self.CIUSim = CIUserSimulator
    self.CIUap = CIUserActsPred
    self.CIUsp = CIUserScorePred
    self.opt = opt

    if opt.uppModel == 'lstm' then
--        local userStates = torch.Tensor(CIUserSimulator.userStateFeatureCnt):fill(0)
--        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 1] = 1
--        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 2] = 0.6
--        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 3] = 0.3
--
--        local prepStates = CIUserSimulator:preprocessUserStateData(torch.Tensor(CIUserSimulator.userStateFeatureCnt):fill(0), opt.prepro)
--        local inPrepStates = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
--        inPrepStates[1] = prepStates:clone()
--        local tabPrepStates = {}
--        for i=1, 5 do
--            tabPrepStates[i] = inPrepStates:clone()
--        end
--        local lstTimeState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
--        lstTimeState[1] = CIUserSimulator:preprocessUserStateData(userStates, opt.prepro)
--        tabPrepStates[6] = lstTimeState
--
--        local nll_acts = self.userActsPred:forward(tabPrepStates)
--        print('@@@', nll_acts[6]:squeeze(), '\n$#$', torch.exp(nll_acts[6]:squeeze()))
--        lp, ain = torch.max(nll_acts[6]:squeeze(), 1)
--        print('##', lp, ain)

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
        for i=1, #CIUserActsPred.rnnRealUserDataStates do
            local userState = CIUserActsPred.rnnRealUserDataStates[i]
            local userAct = CIUserActsPred.rnnRealUserDataActs[i]
            local userRew = CIUserScorePred.rnnRealUserDataRewards[i]

            local tabState = {}
            for j=1, opt.lstmHist do
                local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
                prepUserState[1] = CIUserSimulator:preprocessUserStateData(userState[j], opt.prepro)
                tabState[j] = prepUserState:clone()
            end

            local nll_acts = self.userActsPred:forward(tabState)
            lp, ain = torch.max(nll_acts[opt.lstmHist]:squeeze(), 1)
            if i == CIUserActsPred.rnnRealUserDataStarts[userInd]+countScope then firstActDist[ain[1]] = firstActDist[ain[1]]+1 end   -- check act dist at each x-th time step
--            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end
            lpy, lps = torch.sort(nll_acts[opt.lstmHist]:squeeze(), 1, true)
            local crtExt = false
            local smpLen = self.opt.actEvaScp    -- This is the range of selected actions to check for correct prediction
            for l=1, smpLen do
                if lps[l] == userAct[opt.lstmHist] then
                    crcActCnt = crcActCnt + 1
                    crtExt = true
                end
            end

            local indDiff = i - CIUserActsPred.rnnRealUserDataStarts[userInd]
            if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserActsPred.rnnRealUserDataEnds[userInd] then
                if crtExt then  --ain[1] == userAct[opt.lstmHist] then
                    earlyCrcAct[indDiff+1] = earlyCrcAct[indDiff+1] + 1 -- This is at the indDiff's time step, what is prob of correct act prediction
                end
            earlyTotAct[indDiff+1] = earlyTotAct[indDiff+1] + 1
            end
            if i == CIUserActsPred.rnnRealUserDataEnds[userInd] then
                userInd = userInd+1
            end

            local nll_rewards = self.userScorePred:forward(tabState)
            lp, rin = torch.max(nll_rewards[opt.lstmHist]:squeeze(), 1)
            if userAct[opt.lstmHist] == CIUserSimulator.CIFr.usrActInd_end and rin[1] == userRew[opt.lstmHist] then crcRewCnt = crcRewCnt + 1 end

            tltCnt = tltCnt + 1

--            if userAct[opt.lstmHist] == CIUserSimulator.CIFr.usrActInd_end then
                self.userActsPred:forget()
                self.userScorePred:forget()
--            end

        end

        print('1st act: ', firstActDist)
        print('Following stats are act pred accuracy, reward pred accuracy, act pred accu of each time step')
        print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserActsPred.rnnRealUserDataEnds, torch.cdiv(earlyCrcAct, earlyTotAct))

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

return CIUserBehaviorGenEvaluator

