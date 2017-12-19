gpu = false
shuffle = false
if gpu then
    require 'cutorch'
    require 'cunn'
else 
    require 'torch'
    require 'nn'
end
require 'optim'
require 'dataset'
require 'fullconnect'
require 'convolution'

-- read file

format = 'fullconnect'
normalize = nn.BatchNormalization(28 * 28)
if gpu then
    normalize = normalize:cuda()
end

all = 60000
batch_size = 3
batch = all / batch_size
test_num = 10000
all_images = torch.Tensor(dataset.read_images("train-images-idx3-ubyte", 0, all, format))
all_labels = torch.Tensor(dataset.read_labels("train-labels-idx1-ubyte", 0, all)) + 1

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
        learningRateDecay = 1e-7
    }
    epoch = 2
    iterations = epoch * batch
    for i = 1, iterations do
        local train_images = all_images[{{1, i % batch * batch_size}}]
        --train_images = normalize:forward(train_images)
        local train_labels = all_labels[{{1, i % batch * batch_size}}]

        function feval(params)
            dl_dx:zero()
            local output = net:forward(train_images)
            local err = criterion:forward(output, train_labels)
            local gradOut = criterion:backward(output, train_labels)
            net:backward(train_images, gradOut)
            return err, dl_dx
        end
        _, loss = optim.sgd(feval, x, config)
        if i % 10 == 0 then
            print(string.format("iterations %d, current error: %f", i, loss[1]))
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

test_images = torch.Tensor(dataset.read_images("t10k-images-idx3-ubyte", 0, test_num, format))
test_labels = torch.Tensor(dataset.read_labels("t10k-labels-idx1-ubyte", 0, test_num))

test_images = torch.Tensor(test_images)
test_images = normalize:forward(test_images)
test_labels = torch.Tensor(test_labels) + 1
test_precision = model_precision(net, test_images, test_labels, test_num)
if gpu then
    test_images = test_images:cuda()
    test_labels = test_labels:cuda()
end

print(string.format("test_precision: %f", test_precision))


