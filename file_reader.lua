--
-- User: pwang8
-- Date: 1/15/17
-- Time: 4:35 PM
-- This file is created to read trace and survey files of the CI narrative adaptation experiments.
-- The original purpose of this file is to create a lua version of file reader as the original
-- python file reader used in previous experiments. (The whole thing has to be done!)
--

local _ = require 'moses'
local class = require 'classic'
require 'classic.torch' -- Enables serialisation

local CIFileReader = classic.class('CIFileReader')

local posterReadMinTime = 2.0
local bookReadMinTime = 5.0

local usrActInd_posterRead = 1
local usrActInd_bookRead = 2
local usrActInd_talkKim = 3
local usrActInd_askTeresaSymp = 4
local usrActInd_askBryceSymp = 5
local usrActInd_talkQuentin = 6
local usrActInd_talkRobert = 7
local usrActInd_talkFord = 8
local usrActInd_quiz = 9
local usrActInd_testObject = 10
local usrActInd_submitWorksheet = 11
local usrActInd_BryceRevealActOne = 12
local usrActInd_QuentinRevealActOne = 13
local usrActInd_end = 14

-- Creates a DQN agent
function CIFileReader:_init(opt)
    self.traceFilePath = 'data/training-log-corpus.log'  --'data/training-survey-corpus.csv'
    -- Read data from CSV to tensor
    local traceFile = io.open(self.traceFilePath, 'r')
--    local header = traceFile:read()

    self.traceData = {}
    self.AdpTeresaSymptomAct = {}
    self.AdpBryceSymptomAct = {}
    self.AdpPresentQuizAct = {}
    self.AdpWorksheetLevelAct = {}
    self.talkCntQuentin = {}
    self.talkCntRobert = {}
    self.talkCntFord = {}

    local curId = ''
    local searchNextAdpPresentQuiz = false
    local talkRobFordQuentinLine = 0
    local talkCntRobFordQuentin = nil

    local i = 0     -- line number in trace file
    for line in traceFile:lines('*l') do
        i = i + 1
        local oneLine = line:split('|')
        if curId ~= oneLine[1] then -- new ID observed
            if searchNextAdpPresentQuiz then
                print('Error in trace file, line', i, '. searchNextAdpPresentQuiz is true at starting lines')
                os.exit()
            end
            curId = oneLine[1]
            self.traceData[curId] = {}  -- Assume the first line in each user's trace does not have necessary info
            self.AdpTeresaSymptomAct[curId] = {}
            self.AdpBryceSymptomAct[curId] = {}
            self.AdpPresentQuizAct[curId] = {}
            self.AdpWorksheetLevelAct[curId] = {}
            self.talkCntQuentin[curId] = 0
            self.talkCntRobert[curId] = 0
            self.talkCntFord[curId] = 0
            self.talkCntKim[curId] = 0
        else
            if searchNextAdpPresentQuiz then
                if i > talkRobFordQuentinLine + 3 then
                    print('Error in trace file, line', i, '. select-present-quiz should be in 3 rows from the triggering talk')
                    os.exit()
                elseif oneLine[2] == 'DIALOG' and string.sub(oneLine[6], 1, 7) == 'Kimwant' then
                    print('### Delete the talk log on line', talkRobFordQuentinLine)
                    self.traceData[curId][#self.traceData[curId]] = nil     -- delete the most recent record
                    talkCntRobFordQuentin[curId] = talkCntRobFordQuentin[curId] - 1     -- decrease count
                    searchNextAdpPresentQuiz = false
                elseif oneLine[2] == 'ADAPTATION' and oneLine[4] == 'select-present-quiz' then
                    self.AdpPresentQuizAct[curId][#self.traceData[curId]] = tonumber(string.sub(oneLine[6], -1, -1))
                    searchNextAdpPresentQuiz = false
                end
            elseif oneLine[2] == 'LOOKEND' and oneLine[4]:split('-')[1] == 'poster' and tonumber(oneLine[5]:split('-')[2]) > posterReadMinTime then
                print('#@# poster', i, tonumber(oneLine[5]:split('-')[2]))
                self.traceData[curId][#self.traceData[curId] + 1] = usrActInd_posterRead   -- poster reading action index
            elseif oneLine[2] == 'BOOKREAD' and tonumber(oneLine[7]) > bookReadMinTime then
                print('@!@ book', i, tonumber(oneLine[7]))
                self.traceData[curId][#self.traceData[curId] + 1] = usrActInd_bookRead   -- book reading action index
            elseif oneLine[2] == 'DIALOG' and oneLine[5] == 'kim' and string.sub(oneLine[6], 1, string.len('Pathogen')) == 'Pathogen' then
                print('@@@ talk kim', i)
                self.traceData[curId][#self.traceData[curId] + 1] = usrActInd_talkKim   -- talk with (ask about pathogen) Kim
            elseif oneLine[2] == 'ADAPTATION' and oneLine[4] == 'select-teresa-symptoms-level' then
                print('### ask Teresa symptom', i)
                self.traceData[curId][#self.traceData[curId] + 1] = usrActInd_askTeresaSymp   -- Ask Teresa about her symptoms
                self.AdpTeresaSymptomAct[curId][#self.traceData[curId]] = tonumber(string.sub(oneLine[6], -1, -1))
            elseif oneLine[2] == 'ADAPTATION' and oneLine[4] == 'select-bryce-symptoms-level' then
                print('### ask Bryce symptoms', i)
                self.traceData[curId][#self.traceData[curId] + 1] = usrActInd_askBryceSymp   -- Ask Bryce about his symptoms
                self.AdpBryceSymptomAct[curId][#self.traceData[curId]] = tonumber(string.sub(oneLine[6], -1, -1))
            elseif oneLine[2] == 'TALK' and oneLine[5] == 'cur-action-talk-quentin' then
                print('### talk to quentin', i)
                if self.talkCntQuentin[curId] == 0 then -- select-prent-quiz will be triggered when talking with Quentin for 1st time
                    print('###### first talk with quentin')
                    searchNextAdpPresentQuiz = true
                end
                self.traceData[curId][#self.traceData[curId] + 1] = usrActInd_talkQuentin   -- Talk with Quentin
                self.talkCntQuentin[curId] = self.talkCntQuentin[curId] + 1
                talkCntRobFordQuentin = self.talkCntQuentin
                talkRobFordQuentinLine = i
            elseif oneLine[2] == 'TALK' and oneLine[5] == 'cur-action-talk-robert' then
                print('### talk to robert', i)
                if self.talkCntRobert[curId] == 0 then -- select-prent-quiz will be triggered when talking with Quentin for 1st time
                    print('###### first talk with robert')
                    searchNextAdpPresentQuiz = true
                end
                self.traceData[curId][#self.traceData[curId] + 1] = usrActInd_talkRobert   -- Talk with Robert
                self.talkCntRobert[curId] = self.talkCntRobert[curId] + 1
                talkCntRobFordQuentin = self.talkCntRobert
                talkRobFordQuentinLine = i
            end
        end
        if i == 1500 then
            break
        end
    end

    print('@@@', self.traceData)
    print('### Teresa adp', self.AdpTeresaSymptomAct)
    print('### Bryce adp', self.AdpBryceSymptomAct)
    print('### PresentQ adp', self.AdpPresentQuizAct)

    traceFile:close()
end

return CIFileReader

