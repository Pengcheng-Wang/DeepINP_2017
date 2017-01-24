local CIFileReader = require 'file_reader'
local CIUserSimulator = require 'UserSimulator'
local CIUserActsPredictor = require 'UserSimLearner/UserActsPredictor'

torch.setdefaulttensortype('torch.FloatTensor') -- Todo: pwang8. Change this settig to coordinate with the main setup setting.

-- Read CI trace and survey data files, and do validation
local fr = CIFileReader()
fr:evaluateTraceFile()
fr:evaluateSurveyData()

-- Construct CI user simulator model using real user data
local CIUserModel = CIUserSimulator(fr)
local CIUserActsPred = CIUserActsPredictor(CIUserModel)

for i=1, 100 do
    CIUserActsPred:trainOneEpoch()
end


--print('@@', fr.traceData['100-0028'])
--print('#', #fr.data)
--print('#', #fr.data, ',', fr.data[1], '@@', fr.data[55][1], fr.data[55][81])