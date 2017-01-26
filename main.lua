local CIFileReader = require 'file_reader'
local CIUserSimulator = require 'UserSimulator'
local CIUserActsPredictor = require 'UserSimLearner/UserActsPredictor'
local CIUserScorePredictor = require 'UserSimLearner/UserScorePredictor'

torch.setdefaulttensortype('torch.FloatTensor') -- Todo: pwang8. Change this settig to coordinate with the main setup setting.

opt = lapp[[
       --trType         (default "sc")           training type : sc (score) | ac (action)
    ]]

-- Read CI trace and survey data files, and do validation
local fr = CIFileReader()
fr:evaluateTraceFile()
fr:evaluateSurveyData()

-- Construct CI user simulator model using real user data
local CIUserModel = CIUserSimulator(fr)

if opt.trType == 'sc' then
    local CIUserScorePred = CIUserScorePredictor(CIUserModel)
    for i=1, 2e5 do
        CIUserScorePred:trainOneEpoch()
    end
elseif opt.trType == 'ac' then
    local CIUserActsPred = CIUserActsPredictor(CIUserModel)
    for i=1, 2e5 do
        CIUserActsPred:trainOneEpoch()
    end
end



--print('@@', fr.traceData['100-0028'])
--print('#', #fr.data)
--print('#', #fr.data, ',', fr.data[1], '@@', fr.data[55][1], fr.data[55][81])