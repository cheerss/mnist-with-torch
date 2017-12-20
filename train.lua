require 'torch'
require 'nn'
require 'optim'
require 'dataset'
require 'fullconnect'
require 'convolution'

-- read file

format = 'fullconnect'
normalize = nn.BatchNormalization(28 * 28)

all = 60000
batch_size = 3
batch = all / batch_size
test_num = 10000
all_images = torch.Tensor(dataset.nread_images("train-images-idx3-ubyte", 0, all, format))
all_images:reshape(all_images, all, all_images:size(1) / all)
all_labels = torch.Tensor(dataset.nread_labels("train-labels-idx1-ubyte", 0, all)) + 1

shuffle_indecies = torch.randperm(all):long()
all_images = all_images:index(1, shuffle_indecies):squeeze()
all_labels = all_labels:index(1, shuffle_indecies):squeeze()

if true then
    net = fullconnect:net()
    criterion = nn.ClassNLLCriterion()
    --net = torch.load("inter-model/iter-20000.model")
    x, dl_dx = net:getParameters()
    config = {
        learningRate = 1e-3,
        weightDecay = 0,
        momentum = 0,
        learningRateDecay = 1e-7
    }
    epoch = 2
    iterations = epoch * batch
    local last = os.clock()
    local now = 0
    local counter = 0
    function feval(params)
        local start_index = 1 + counter * batch_size
        local end_index = math.min(all, start_index + batch_size)
        if end_index == all then
            counter = 0
        else
            counter = counter + 1
        end
        local train_images = all_images[{{start_index, end_index}}]
        --train_images = normalize:forward(train_images)
        local train_labels = all_labels[{{start_index, end_index}}]
        dl_dx:zero()
        local output = net:forward(train_images)
        local err = criterion:forward(output, train_labels)
        local gradOut = criterion:backward(output, train_labels)
        net:backward(train_images, gradOut)
        return err, dl_dx
    end

    for i = 1, iterations do
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
    for i  = 1, n do
        local output = net:forward(dataset[i])
        local _, ans = torch.max(output, 1)
        if ans[1] == labels[i] then
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
test_labels = torch.Tensor(dataset.nread_labels("t10k-labels-idx1-ubyte", 0, test_num))

test_images = torch.Tensor(test_images)
test_images = normalize:forward(test_images)
test_labels = torch.Tensor(test_labels) + 1
test_precision = model_precision(net, test_images, test_labels, test_num)
if gpu then
    test_images = test_images:cuda()
    test_labels = test_labels:cuda()
end

print(string.format("test_precision: %f", test_precision))


