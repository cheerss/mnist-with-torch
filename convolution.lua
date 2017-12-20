require 'torch'
require 'nn'

convolution = {}

function convolution:net()
    local res = nn.Sequential()
    res:add(nn.SpatialConvolution(1, 3, 2, 2, 1, 1))
    res:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    res:add(nn.Reshape(169 * 3, true))
    res:add(nn.Linear(13 * 13 * 3, 10))
    res:add(nn.LogSoftMax())
    return res
end
