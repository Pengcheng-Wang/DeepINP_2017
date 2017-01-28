--
-- User: pwang8
-- Date: 1/22/17
-- Time: 3:44 PM
-- Using real interaction data to create user simulation model
--

local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation
local TableSet = require 'MyMisc.TableSetMisc'

local CIUserSimulator = classic.class('UserSimulator')

function CIUserSimulator:_init(CIFileReader)
    self.realUserDataStates = {}
    self.realUserDataActs = {}
    self.realUserDataRewards = {}
    self.CIFr = CIFileReader    -- a ref to the file reader
    self.realUserDataStartLines = {}    -- this table stores the starting line of each real human user's interation
    self.realUserDataEndLines = {}

    self.userStateFeatureCnt = CIFileReader.userStateGamePlayFeatureCnt + CIFileReader.userStateSurveyFeatureCnt    -- 18+3 now

    for userId, userRcd in pairs(CIFileReader.traceData) do

        -- set up initial user state before taking actions
        self.realUserDataStates[#self.realUserDataStates + 1] = torch.Tensor(self.userStateFeatureCnt):fill(0)
        self.realUserDataStartLines[#self.realUserDataStartLines + 1] = #self.realUserDataStates -- Stores start lines for each user interaction
        for i=1, CIFileReader.userStateSurveyFeatureCnt do
            -- set up survey features, which are behind game play features in the state feature tensor
            self.realUserDataStates[#self.realUserDataStates][CIFileReader.userStateGamePlayFeatureCnt+i] = CIFileReader.surveyData[userId][i]
        end

        for time, act in ipairs(userRcd) do
            self.realUserDataActs[#self.realUserDataStates] = act
--            print('#', userId, self.realUserDataStates[#self.realUserDataStates], ',', self.realUserDataActs[#self.realUserDataStates])

            if CIFileReader.surveyData[userId][CIFileReader.userStateSurveyFeatureCnt+1] > 0 then
                self.realUserDataRewards[#self.realUserDataStates] = 1
            else
                self.realUserDataRewards[#self.realUserDataStates] = 2     -- This is (binary) reward class label, not reward value
            end

            if act == CIFileReader.usrActInd_end then
--                print('@@ End action reached')
            else
                -- set the next time step state set
                self.realUserDataStates[#self.realUserDataStates + 1] = self.realUserDataStates[#self.realUserDataStates]:clone()

                if act == CIFileReader.usrActInd_askTeresaSymp then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_TeresaSymp] =
                        (4 - CIFileReader.AdpTeresaSymptomAct[userId][time]) / 3.0  -- (act1--1.0, act3--0.33). So y=(4-x)/3
                elseif act == CIFileReader.usrActInd_askBryceSymp then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_BryceSymp] =
                        (3 - CIFileReader.AdpBryceSymptomAct[userId][time]) / 2.0  -- (act1--1.0, act2--0.5). So y=(3-x)/2
                elseif act == CIFileReader.usrActInd_talkQuentin and
                        self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_KimLetQuentinRevealActOne] < 1 and
                        self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_talkQuentin] < 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_PresentQuiz] =
                            (2 - CIFileReader.AdpPresentQuizAct[userId][time])  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x
                elseif act == CIFileReader.usrActInd_talkRobert and
                        self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_talkRobert] < 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_PresentQuiz] =
                        (2 - CIFileReader.AdpPresentQuizAct[userId][time])  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x
                elseif act == CIFileReader.usrActInd_talkFord and
                        self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_talkFord] < 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_PresentQuiz] =
                        (2 - CIFileReader.AdpPresentQuizAct[userId][time])  -- act1-quiz-1.0, act2-no_quiz-0. y=2-x
                elseif act == CIFileReader.usrActInd_submitWorksheet then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrStateFeatureInd_WorksheetLevel] =
                        (CIFileReader.AdpWorksheetLevelAct[userId][time] / 3.0)  -- act1-0.33, act3-1. y=x/3
                end

                -- Add 1 to corresponding state features
                self.realUserDataStates[#self.realUserDataStates][act] = self.realUserDataStates[#self.realUserDataStates][act] + 1

                -- For indices 12, 13, 14, state feature values can only be 0 or 1
                if self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_BryceRevealActOne] > 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_BryceRevealActOne] = 1
                end
                if self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_QuentinRevealActOne] > 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_QuentinRevealActOne] = 1
                end
                if self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_KimLetQuentinRevealActOne] > 1 then
                    self.realUserDataStates[#self.realUserDataStates][CIFileReader.usrActInd_KimLetQuentinRevealActOne] = 1
                end

            end

        end

    end
    print('Human user actions number: ', #self.realUserDataStates, #self.realUserDataActs)

    for i=1, #self.realUserDataStartLines - 1 do
        self.realUserDataEndLines[i] = self.realUserDataStartLines[i+1] - 1
    end
    self.realUserDataEndLines[#self.realUserDataStartLines] = #self.realUserDataStates

    self.stateFeatureRescaleFactor = torch.Tensor(self.userStateFeatureCnt):fill(1)
    self.stateFeatureMeanEachFeature = torch.Tensor(self.userStateFeatureCnt):fill(0)
    self.stateFeatureStdEachFeature = torch.Tensor(self.userStateFeatureCnt):fill(1)
    -- Calculate user state feature value rescale factors
    self:_calcRealUserStateFeatureRescaleFactor()
    collectgarbage()

    -- The shortest length record has a user actoin sequence length of 2. User id is 100-0466
--    -- calc min length
--    local minlen = 9999
--    for i=1,#self.realUserDataStartLines-1 do
--        if minlen > self.realUserDataStartLines[i+1] - self.realUserDataStartLines[i] then
--            minlen = self.realUserDataStartLines[i+1] - self.realUserDataStartLines[i]
--        end
--    end
--    if minlen > #self.realUserDataStates - self.realUserDataStartLines[#self.realUserDataStartLines] then
--        minlen = #self.realUserDataStates - self.realUserDataStartLines[#self.realUserDataStartLines]
--    end
--    print('$$$$$ min traj length is', minlen) os.exit()
    -- 273 students with postive nlg, 39 with 0 nlg, 90 with negative nlg. 67.9%
end

--- Calculate the observed largest state feature value for each game play feature,
--- and use it to rescale feature value later
function CIUserSimulator:_calcRealUserStateFeatureRescaleFactor()
    local allUserDataStates = torch.Tensor(#self.realUserDataStates, self.userStateFeatureCnt)
    local allInd = 1
    for _,v in pairs(self.realUserDataStates) do
        for i=1, self.CIFr.userStateGamePlayFeatureCnt do
            if self.stateFeatureRescaleFactor[i] < v[i] then
                self.stateFeatureRescaleFactor[i] = v[i]
            end
        end
        allUserDataStates[allInd] = v:clone()
        allInd = allInd + 1
    end
    self.stateFeatureMeanEachFeature = torch.mean(allUserDataStates, 1):squeeze()
    self.stateFeatureStdEachFeature = torch.std(allUserDataStates, 1):squeeze()
--    print('@@', self.stateFeatureMeanEachFeature, '#', self.stateFeatureStdEachFeature)
--    print('##', self.stateFeatureRescaleFactor)
    -- For the 402 CI data, this stateFeatureRescaleFactor vector is
    -- {44 ,20 ,3 ,9 ,7 ,9 ,7 ,10 ,39 ,43 ,10 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1}
    -- Note: when real/simulated user data is used in ML algorithms,
    -- the raw feature values should be divided by this tensor for rescaling.
    -- torch.cdiv(x, self.stateFeatureRescaleFactor)
    -- This is the stateFeatureMeanEachFeature:
    -- {5.0325, 1.3671, 0.7945, 0.9831, 1.0969, 1.6381, 0.9424, 0.8351, 2.9051,
    -- 5.3980, 0.6525, 0.2551, 0.2581, 0.1637, 0.4778, 0.5266, 0.2398, 0.4211, 0.4642 ,0.6148 ,0.3487}
    -- This is the stateFeatureStdEachFeature:
    -- {4.7120, 2.4729, 0.5952, 0.9774, 1.0633, 1.4782, 0.8478, 0.8897, 5.1906, 6.7046,
    -- 1.1095, 0.4359, 0.4376, 0.3700, 0.3904, 0.4136, 0.3589, 0.4938, 0.4987, 0.2325, 0.1198}
end

--- Right now, this preprocessing is rescaling
function CIUserSimulator:preprocessUserStateData(obvUserData, ppType)
    if ppType == 'rsc' then
        return torch.cdiv(obvUserData, self.stateFeatureRescaleFactor)
    elseif ppType == 'std' then
        local subMean = torch.add(obvUserData, -1, self.stateFeatureMeanEachFeature)
        return torch.cdiv(subMean, self.stateFeatureStdEachFeature)
    else
        print('!!!Error. Unrecognized preprocessing in UserSimulator.', ppType)
    end

end

--- Check if narrative adaptation point will be triggered
--  Notice: the curState should be raw state values, not preprocessed values
function CIUserSimulator:isAdpTriggered(curState, act)
    -- curState should be a 1d or 2d tensor. If it is
    -- 2d, I assume the 1st dim is batch dimension
    assert(curState:dim() == 1 or curState:dim() == 2)
    local stateRef = curState
    if curState:dim() == 2 then stateRef = curState[1] end

    if act == self.CIFr.usrActInd_askTeresaSymp then
        return true, self.CIFr.ciAdp_TeresaSymp
    elseif act == self.CIFr.usrActInd_askBryceSymp then
        return true, self.CIFr.ciAdp_BryceSymp
    elseif act == self.CIFr.usrActInd_talkQuentin and
            curState[self.CIFr.usrActInd_KimLetQuentinRevealActOne] < 1 and
            curState[self.CIFr.usrActInd_talkQuentin] < 1 then
        return true, self.CIFr.ciAdp_PresentQuiz
    elseif act == self.CIFr.usrActInd_talkRobert and
            curState[self.CIFr.usrActInd_talkRobert] < 1 then
        return true, self.CIFr.ciAdp_PresentQuiz
    elseif act == self.CIFr.usrActInd_talkFord and
            curState[self.CIFr.usrActInd_talkFord] < 1 then
        return true, self.CIFr.ciAdp_PresentQuiz
    elseif act == self.CIFr.usrActInd_submitWorksheet then
        return true, self.CIFr.ciAdp_WorksheetLevel
    end

    return false, 0
end

return CIUserSimulator
