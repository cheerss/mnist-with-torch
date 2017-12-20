require 'torch'
require 'nn'
require 'optim'
require 'dataset'
require 'fullconnect'
require 'convolution'

cmd = torch.CmdLine()
cmd:option('-gpu', false, 'whether to use GPU')
cmd:option('-batchSize', 500, 'batch size')
cmd:option('-network', 'fullconnect', 'the type of the network, could be "fullconnect" or "conv"')
cmd:option('-pretrained', 0, 'the path of a pretrained moedel, have not been used yet')
cmd:option('-epoch', 35, 'epoch')
cmd:option('-learningRate', 1e-1, 'the learning rate')
cmd:option('-learningRateDecay', 1e-7, 'the learning rate decay')
cmd:option('-momentum', 0, 'the learning rate decay')
cmd:option('-weightDecay', 0, 'the learning rate decay')
cmd:option('-trainSize', 60000, 'the number of train images, should <60000')
cmd:option('-testSize', 10000, 'the number of test imgages, should <10000')
params = cmd:parse(arg)

if params.gpu then
    require 'cutorch'
    require 'cunn'
    gpu = params.gpu
end

format = params.network
all = params.trainSize
batch_size = params.batchSize
batch = all / batch_size
test_num = params.testSize
config = {
    learningRate = params.learningRate,
    weightDecay = 0,
    momentum = 0,
    learningRateDecay = 1e-7
}
all_images = torch.Tensor(dataset.nread_images("train-images-idx3-ubyte", 0, all, format))
all_images:reshape(all_images, all, all_images:size(1) / all)
all_labels = torch.Tensor(dataset.nread_labels("train-labels-idx1-ubyte", 0, all)) + 1

if true then
    if params.network == 'fullconnect' then
        net = fullconnect:net()
    else
        net = convolution:net()
        all_images:reshape(all_images, all, 1, 28, 28)
    end
    criterion = nn.ClassNLLCriterion()
    if gpu then
        all_images = all_images:cuda()
        all_labels = all_labels:cuda()
        net = net:cuda()
        criterion = criterion:cuda()
    end
    x, dl_dx = net:getParameters()
    epoch = params.epoch
    iterations = epoch * batch
    local last = os.clock()
    local now = 0
    local counter = 0
    for i = 1, iterations do
        function feval(params)
            shuffle_indecies = torch.randperm(all):long()
            shuffle_indecies = shuffle_indecies[{{1, batch_size}}]
            local start_index = 1 + counter * batch_size
            local end_index = math.min(all, start_index + batch_size)
            if end_index == all then
                counter = 0
            else
                counter = counter + 1
            end
            local train_images = all_images:index(1, shuffle_indecies)
            local train_labels = all_labels:index(1, shuffle_indecies)
            dl_dx:zero()
            local output = net:forward(train_images)
            local err = criterion:forward(output, train_labels)
            local gradOut = criterion:backward(output, train_labels)
            net:backward(train_images, gradOut)
            return err, dl_dx
        end
        _, loss = optim.sgd(feval, x, config)
        if i % 10 == 0 then
            now = os.clock()
            print(string.format("time: %.6f, iterations %d, current error: %f", now - last,  i, loss[1]))
            last = now
        end
    end
else
end

function model_precision(net, dataset, labels, n)
    local precision = 0
    local output = net:forward(dataset)
    local _, ans = torch.max(output, 2)
    for i  = 1, n do
        if ans[i][1] == labels[i] then
            precision = precision + 1
        end
    end
    precision = precision / n
    return precision
end

images = all_images
labels = all_labels
train_precision = model_precision(net, images, labels, all)
print(string.format("train_precision: %f", train_precision))

test_images = torch.Tensor(dataset.nread_images("t10k-images-idx3-ubyte", 0, test_num, format))
test_images:reshape(test_images, test_num, test_images:size(1) / test_num)
test_labels = torch.Tensor(dataset.nread_labels("t10k-labels-idx1-ubyte", 0, test_num)) + 1

if gpu then
    test_images = test_images:cuda()
    test_labels = test_labels:cuda()
end
if params.network == 'conv' then
    test_images:reshape(test_images, test_num, 1, 28, 28)
end
test_precision = model_precision(net, test_images, test_labels, test_num)
print(string.format("test_precision: %f", test_precision))
