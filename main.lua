local CIFileReader = require 'file_reader'
local CIUserSimulator = require 'UserSimulator'

torch.setdefaulttensortype('torch.FloatTensor') -- Todo: pwang8. Change this settig to coordinate with the main setup setting.

-- Read CI trace and survey data files, and do validation
local fr = CIFileReader()
fr:evaluateTraceFile()
fr:evaluateSurveyData()

-- Construct CI user simulator model using real user data
local CIUserModel = CIUserSimulator(fr)

--print('@@', fr.traceData['100-0028'])
--print('#', #fr.data)
--print('#', #fr.data, ',', fr.data[1], '@@', fr.data[55][1], fr.data[55][81])