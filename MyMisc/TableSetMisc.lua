--
-- User: pwang8
-- Date: 1/21/17
-- Time: 8:29 PM
-- Places to put setContains function
--
local class = require 'classic'
local TableSetMisc = classic.class('TableSetMisc')

function TableSetMisc.static.addToSet(set, key)
    set[key] = true
end
function TableSetMisc.static.removeFromSet(set, key)
    set[key] = nil
end
function TableSetMisc.static.setContains(set, key)
    return set[key] ~= nil
end

return TableSetMisc