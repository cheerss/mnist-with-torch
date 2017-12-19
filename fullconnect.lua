require 'cutorch'
require 'cunn'

fullconnect = {}

function fullconnect:net()
    local net = nn.Sequential()
    net:add(nn.Linear(28 * 28, 10))
    --net:add(nn.Sigmoid())
    --net:add(nn.Linear(300, 10))
    net:add(nn.LogSoftMax())
    return net
end
