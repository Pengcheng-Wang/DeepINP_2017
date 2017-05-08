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

function CIUserBehaviorGenEvaluator:_init(CIUserSimulator, CIUserActsPred, CIUserScorePred, CIUserActScorePred, opt)

    local tltCnt = 0
    local crcActCnt = 0
    local crcRewCnt = 0
    local userInd = 1
    local earlyTotAct = torch.Tensor(opt.lstmHist+81):fill(1e-6)
    local earlyCrcAct = torch.Tensor(opt.lstmHist+81):fill(0)
    local firstActDist = torch.Tensor(CIUserSimulator.CIFr.usrActInd_end):fill(0)   -- 15 user actions

    local countScope=0  -- This param is used to calculate action distribution at countScope time step

    if opt.sharedLayer < 1 then

        -- Bipartitle Act/Score prediction model
        -- The user action/score predictors for evaluation should be pre-trained, and loaded from files
        -- Also, the CIUserSimulator, CIUserActsPred, CIUserScorePred should be initialized using
        -- the test set.
        print('User Act Score Bipartitle model: #', paths.concat(opt.ubgDir , opt.uapFile))
        self.userActsPred = torch.load(paths.concat(opt.ubgDir , opt.uapFile))
        self.userScorePred = torch.load(paths.concat(opt.ubgDir , opt.uspFile))
        self.userActsPred:evaluate()
        self.userScorePred:evaluate()
        self.CIUSim = CIUserSimulator
        self.CIUap = CIUserActsPred
        self.CIUsp = CIUserScorePred
        self.opt = opt

        if opt.uppModel == 'lstm' then
            -- sharedLayer == 0 and lstm model
--            self._actionDistributionCalc(CIUserSimulator, countScope)

            self.userActsPred:forget()
            self.userScorePred:forget()

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
                        break
                    end
                end

                local indDiff = i - CIUserActsPred.rnnRealUserDataStarts[userInd]
                if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserActsPred.rnnRealUserDataEnds[userInd] then
                    if crtExt then
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

                self.userActsPred:forget()
                self.userScorePred:forget()

            end

--            print('1st act: ', firstActDist)
            print('Following stats are act pred accuracy, reward pred accuracy, act pred accu of each time step')
            print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserActsPred.rnnRealUserDataEnds, torch.cdiv(earlyCrcAct, earlyTotAct))

        else
            -- sharedLayer == 0 and not lstm models
            for i=1, #CIUserSimulator.realUserDataStates do
                local userState = CIUserSimulator:preprocessUserStateData(CIUserSimulator.realUserDataStates[i], opt.prepro)
                local userAct = CIUserSimulator.realUserDataActs[i]
                local userRew = CIUserSimulator.realUserDataRewards[i]

                local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
                prepUserState[1] = userState:clone()

                local nll_acts = self.userActsPred:forward(prepUserState)
                lp, ain = torch.max(nll_acts[1]:squeeze(), 1)

                if i == CIUserSimulator.realUserDataStartLines[userInd]+countScope then firstActDist[ain[1]] = firstActDist[ain[1]]+1 end   -- check act dist at each x-th time step
                --            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end
                lpy, lps = torch.sort(nll_acts[1]:squeeze(), 1, true)
                local crtExt = false
                local smpLen = self.opt.actEvaScp    -- This is the range of selected actions to check for correct prediction
                for l=1, smpLen do
                    if lps[l] == userAct then
                        crcActCnt = crcActCnt + 1
                        crtExt = true
                        break
                    end
                end

                local indDiff = i - CIUserSimulator.realUserDataStartLines[userInd]
                if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserSimulator.realUserDataEndLines[userInd] then
                    if crtExt then
                        earlyCrcAct[indDiff+1] = earlyCrcAct[indDiff+1] + 1 -- This is at the indDiff's time step, what is prob of correct act prediction
                    end
                    earlyTotAct[indDiff+1] = earlyTotAct[indDiff+1] + 1
                end
                if i == CIUserSimulator.realUserDataEndLines[userInd] then
                    userInd = userInd+1
                end

                local nll_rewards = self.userScorePred:forward(prepUserState)
                lp, rin = torch.max(nll_rewards[1]:squeeze(), 1)
                if userAct == CIUserSimulator.CIFr.usrActInd_end and rin[1] == userRew then crcRewCnt = crcRewCnt + 1 end

                tltCnt = tltCnt + 1

            end

--            print('1st act: ', firstActDist)
            print('Following stats are act pred accuracy, reward pred accuracy, act pred accu of each time step')
            print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserSimulator.realUserDataEndLines, torch.cdiv(earlyCrcAct, earlyTotAct))
        end

    else
        -- opt.sharedLayer == 1
        -- Unique Act/Score prediction model with shared lower layers
        -- The user action/score predictors for evaluation should be pre-trained, and loaded from files
        -- Also, the CIUserSimulator, CIUserActsPred, CIUserScorePred should be initialized using
        -- the test set.
        print('User Act Score shared-layer model: #', paths.concat(opt.ubgDir , opt.uapFile))
        self.userActScorePred = torch.load(paths.concat(opt.ubgDir , opt.uapFile))
        self.userActScorePred:evaluate()
        self.CIUSim = CIUserSimulator
        self.CIUasp = CIUserActScorePred
        self.opt = opt

        if opt.uppModel == 'lstm' then
            -- sharedLayer == 1 and lstm model

            --            self._actionDistributionCalc(CIUserSimulator, countScope)

            self.userActScorePred:forget()

            for i=1, #CIUserActScorePred.rnnRealUserDataStates do
                local userState = CIUserActScorePred.rnnRealUserDataStates[i]
                local userAct = CIUserActScorePred.rnnRealUserDataActs[i]
                local userRew = CIUserActScorePred.rnnRealUserDataRewards[i]

                local tabState = {}
                for j=1, opt.lstmHist do
                    local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
                    prepUserState[1] = CIUserSimulator:preprocessUserStateData(userState[j], opt.prepro)
                    tabState[j] = prepUserState:clone()
                end

                local nll_acts = self.userActScorePred:forward(tabState)
                lp, ain = torch.max(nll_acts[opt.lstmHist][1]:squeeze(), 1)     -- then 2nd [1] index is for action prediction from the shared act/score prediction outcome
                if i == CIUserActScorePred.rnnRealUserDataStarts[userInd]+countScope then firstActDist[ain[1]] = firstActDist[ain[1]]+1 end   -- check act dist at each x-th time step
                --            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end
                lpy, lps = torch.sort(nll_acts[opt.lstmHist][1]:squeeze(), 1, true)
                local crtExt = false
                local smpLen = self.opt.actEvaScp    -- This is the range of selected actions to check for correct prediction
                for l=1, smpLen do
                    if lps[l] == userAct[opt.lstmHist] then
                        crcActCnt = crcActCnt + 1
                        crtExt = true
                        break
                    end
                end

                local indDiff = i - CIUserActScorePred.rnnRealUserDataStarts[userInd]
                if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserActScorePred.rnnRealUserDataEnds[userInd] then
                    if crtExt then
                        earlyCrcAct[indDiff+1] = earlyCrcAct[indDiff+1] + 1 -- This is at the indDiff's time step, what is prob of correct act prediction
                    end
                    earlyTotAct[indDiff+1] = earlyTotAct[indDiff+1] + 1
                end
                if i == CIUserActScorePred.rnnRealUserDataEnds[userInd] then
                    userInd = userInd+1
                end

                -- The predicted reward is the 2nd output of nll_acts in 2nd dim
                lp, rin = torch.max(nll_acts[opt.lstmHist][2]:squeeze(), 1)
                if userAct[opt.lstmHist] == CIUserSimulator.CIFr.usrActInd_end and rin[1] == userRew[opt.lstmHist] then crcRewCnt = crcRewCnt + 1 end

                tltCnt = tltCnt + 1

                self.userActScorePred:forget()

            end

            --            print('1st act: ', firstActDist)
            print('Following stats are act pred accuracy, reward pred accuracy, act pred accu of each time step')
            print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserActScorePred.rnnRealUserDataEnds, torch.cdiv(earlyCrcAct, earlyTotAct))

        else
            -- SharedLayer == 1, and not lstm models
            for i=1, #CIUserSimulator.realUserDataStates do
                local userState = CIUserSimulator:preprocessUserStateData(CIUserSimulator.realUserDataStates[i], opt.prepro)
                local userAct = CIUserSimulator.realUserDataActs[i]
                local userRew = CIUserSimulator.realUserDataRewards[i]

                local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
                prepUserState[1] = userState:clone()

                local nll_acts = self.userActScorePred:forward(prepUserState)

                -- Here, if moe is used with shared lower layers, it is the problem that,
                -- due to limitation of MixtureTable module, we have to join tables together as
                -- a single tensor as output of the whole user action and score prediction model.
                -- So, to guarantee the compatability, we need split the tensor into two tables here,
                -- for act prediction and score prediction respectively.
                if opt.uppModel == 'moe' then
                    nll_acts = nll_acts:split(CIUserSimulator.CIFr.usrActInd_end, 2)  -- We assume 1st dim is batch index. Act pred is the 1st set of output, having dim of 15. Score dim 2.
                end

                lp, ain = torch.max(nll_acts[1][1]:squeeze(), 1)

                if i == CIUserSimulator.realUserDataStartLines[userInd]+countScope then firstActDist[ain[1]] = firstActDist[ain[1]]+1 end   -- check act dist at each x-th time step
                --            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end
                lpy, lps = torch.sort(nll_acts[1][1]:squeeze(), 1, true)
                local crtExt = false
                local smpLen = self.opt.actEvaScp    -- This is the range of selected actions to check for correct prediction
                for l=1, smpLen do
                    if lps[l] == userAct then
                        crcActCnt = crcActCnt + 1
                        crtExt = true
                        break
                    end
                end

                local indDiff = i - CIUserSimulator.realUserDataStartLines[userInd]
                if indDiff >= 0 and indDiff <= opt.lstmHist+80 and i <= CIUserSimulator.realUserDataEndLines[userInd] then
                    if crtExt then
                        earlyCrcAct[indDiff+1] = earlyCrcAct[indDiff+1] + 1 -- This is at the indDiff's time step, what is prob of correct act prediction
                    end
                    earlyTotAct[indDiff+1] = earlyTotAct[indDiff+1] + 1
                end
                if i == CIUserSimulator.realUserDataEndLines[userInd] then
                    userInd = userInd+1
                end

                -- The predicted reward is the 2nd output of nll_acts in 2nd dim
                lp, rin = torch.max(nll_acts[2][1]:squeeze(), 1)
                if userAct == CIUserSimulator.CIFr.usrActInd_end and rin[1] == userRew then crcRewCnt = crcRewCnt + 1 end

                tltCnt = tltCnt + 1

            end

            --            print('1st act: ', firstActDist)
            print('Following stats are act pred accuracy, reward pred accuracy, act pred accu of each time step')
            print('###', crcActCnt/tltCnt, crcRewCnt/#CIUserSimulator.realUserDataEndLines, torch.cdiv(earlyCrcAct, earlyTotAct))
        end

    end
end

function CIUserBehaviorGenEvaluator:_actionDistributionCalc(CIUserSimulator, cntScope)
    local countScope = cntScope
    local st = torch.Tensor(CIUserSimulator.CIFr.usrActInd_end):fill(0)     -- tensor dim is 15 (user action types)
    for k, v in ipairs(CIUserSimulator.realUserDataStartLines) do
        st[CIUserSimulator.realUserDataActs[v+countScope]] = st[CIUserSimulator.realUserDataActs[v+countScope]] +1 -- check act dist at each x-th time step
    end
    print('Act count at time step ', countScope+1, ' is:', st)
end

return CIUserBehaviorGenEvaluator

