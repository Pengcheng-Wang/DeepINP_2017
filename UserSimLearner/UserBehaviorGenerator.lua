--
-- User: pwang8
-- Date: 1/26/17
-- Time: 3:21 PM
-- Generate simulated user interaction data using the predicted
-- user action and user nlg score
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

    if opt.uppModel == 'lstm' then
        local userStates = torch.Tensor(CIUserSimulator.userStateFeatureCnt):fill(0)
        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 1] = 0
        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 2] = 5
        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 3] = 9

        local prepStates = CIUserSimulator:preprocessUserStateData(torch.Tensor(CIUserSimulator.userStateFeatureCnt):fill(0), opt.prepro)
        local inPrepStates = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
        inPrepStates[1] = prepStates:clone()
        local tabPrepStates = {}
        for i=1, 5 do
            tabPrepStates[i] = CIUserSimulator:preprocessUserStateData(userStates, opt.prepro)
        end
        tabPrepStates[6] = inPrepStates

        local nll_acts = self.userActsPred:forward(tabPrepStates)
        lp, ain = torch.max(nll_acts[6]:squeeze(), 1)
        print('##', lp, ain)

        local st = torch.Tensor(15):fill(0)
        for k, v in ipairs(CIUserSimulator.realUserDataStartLines) do
            st[CIUserSimulator.realUserDataActs[v]] = st[CIUserSimulator.realUserDataActs[v]] +1
        end
        print('Act count', st)

        self.userActsPred:forget()
        self.userScorePred:forget()
        local tltCnt = 0
        local crcActCnt = 0
        local crcRewCnt = 0
        local userInd = 1
        local earlyTotAct = 0
        local earlyCrcAct = 0
        for i=1, #CIUserSimulator.realUserDataStates do
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
            if ain[1] == userAct[opt.lstmHist] then crcActCnt = crcActCnt + 1 end

            if i == CIUserSimulator.realUserDataEndLines[userInd] then
                userInd = userInd+1
            elseif i == CIUserSimulator.realUserDataStartLines[userInd] and i <= CIUserSimulator.realUserDataEndLines[userInd] then
                if ain[1] == userAct[opt.lstmHist] then
                    earlyCrcAct = earlyCrcAct + 1
                end
                earlyTotAct = earlyTotAct + 1
            end

            local nll_rewards = self.userScorePred:forward(tabState)
            lp, rin = torch.max(nll_rewards[opt.lstmHist]:squeeze(), 1)
            if userAct[opt.lstmHist] == CIUserSimulator.CIFr.usrActInd_end and rin[1] == userRew[opt.lstmHist] then crcRewCnt = crcRewCnt + 1 end

            tltCnt = tltCnt + 1

            if userAct[opt.lstmHist] == CIUserSimulator.CIFr.usrActInd_end then
                self.userActsPred:forget()
                self.userScorePred:forget()
            end

        end

        print('###', crcActCnt/tltCnt, crcRewCnt/402, earlyCrcAct/earlyTotAct)

        os.exit()
    else
        print('ttt')
        local userStates = torch.Tensor(CIUserSimulator.userStateFeatureCnt):fill(0)
        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 1] = 0
        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 2] = 5
        userStates[CIUserSimulator.CIFr.userStateGamePlayFeatureCnt + 3] = 9

        local prepStates = CIUserSimulator:preprocessUserStateData(userStates, opt.prepro)
        local inPrepStates = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
        inPrepStates[1] = prepStates:clone()

        local nll_acts = self.userActsPred:forward(inPrepStates)
        lp, ain = torch.max(nll_acts[1]:squeeze(), 1)
        print('##', lp, ain, nll_acts[1]:squeeze())

        local st = torch.Tensor(15):fill(0)
        for k, v in ipairs(CIUserSimulator.realUserDataStartLines) do
            st[CIUserSimulator.realUserDataActs[v]] = st[CIUserSimulator.realUserDataActs[v]] +1
        end
        print('Act count\n', st)

        self.userActsPred:forget()
--        self.userScorePred:forget()
        local tltCnt = 0
        local crcActCnt = 0
        local crcRewCnt = 0
        local userInd = 1
        local earlyTotAct = 0
        local earlyCrcAct = 0
        for i=1, #CIUserSimulator.realUserDataStates do
            local userState = CIUserSimulator:preprocessUserStateData(CIUserSimulator.realUserDataStates[i], opt.prepro)
            local userAct = CIUserSimulator.realUserDataActs[i]
            local userRew = CIUserSimulator.realUserDataRewards[i]

            local prepUserState = torch.Tensor(1, CIUserSimulator.userStateFeatureCnt)
            prepUserState[1] = userState:clone()

            local nll_acts = self.userActsPred:forward(prepUserState)
            lp, ain = torch.max(nll_acts[1]:squeeze(), 1)
            if ain[1] == userAct then crcActCnt = crcActCnt + 1 end

            if i == CIUserSimulator.realUserDataEndLines[userInd] then
                userInd = userInd+1
            elseif i == CIUserSimulator.realUserDataStartLines[userInd] and i <= CIUserSimulator.realUserDataEndLines[userInd] then
                if ain[1] == userAct then
                    earlyCrcAct = earlyCrcAct + 1
                end
                earlyTotAct = earlyTotAct + 1
            end

--            local nll_rewards = self.userScorePred:forward(prepUserState)
--            lp, rin = torch.max(nll_rewards[1]:squeeze(), 1)
--            if userAct == CIUserSimulator.CIFr.usrActInd_end and rin[1] == userRew then crcRewCnt = crcRewCnt + 1 end

            tltCnt = tltCnt + 1

            if userAct == CIUserSimulator.CIFr.usrActInd_end then
                self.userActsPred:forget()
--                self.userScorePred:forget()
            end

        end

        print('###', crcActCnt/tltCnt, crcRewCnt/402, earlyCrcAct/earlyTotAct)

        os.exit()
    end
end

return CIUserBehaviorPredictor
