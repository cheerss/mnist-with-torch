require 'cutorch'
require 'cunn'

convolution = {}

function convolution:net()
    local res = nn.Sequential()
    res:add(nn.SpatialConvolution(1, 3, 3, 3, 2, 2))
    res:add(nn.Reshape(169 * 3, true))
    res:add(nn.Linear(13 * 13 * 3, 10))
    res:add(nn.LogSoftMax())
    return res
end
