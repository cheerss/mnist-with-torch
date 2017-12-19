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
all_labels = torch.Tensor(dataset.read_labels("train-labels-idx1-ubyte", 0, all))

shuffle_indecies = torch.randperm(all):long()
all_images = all_images:index(1, shuffle_indecies):squeeze()
all_labels = all_labels:index(1, shuffle_indecies):squeeze()


if true then
    if format == 'conv' then
        net = convolution:net()
    else
        net = fullconnect:net()
    end

    criterion = nn.ClassNLLCriterion()
    if gpu then
        criterion = criterion:cuda()
        net:cuda()
    end
    --net = torch.load("inter-model/iter-20000.model")
    x, dl_dx = net:getParameters()
    config = {
        learningRate = 1e-3,
        learningRateDecay = 1e-7
    }
    epoch = 2
    iterations = epoch * batch
    counter = 0
    for i = 1, iterations do
        local p1 = os.clock()
        if shuffle then
            shuffle_indecies = torch.randperm(all):long()
            shuffl_indecies = shuffle_indecies[{{1, batch_size}}]
            train_images = all_images:index(1, shuffle_indecies):squeeze()
        else
            train_images = all_images[{{1, i % batch * batch_size}}]
        end
        if gpu then
            train_images = train_images:cuda()
        end
        local p2 = os.clock()
        if format ~= 'conv' then
            --train_images = normalize:forward(train_images)
        end
        local p3 = os.clock()
        if shuffle then
            train_labels = all_labels:index(1, shuffle_indecies):squeeze()
        else
            train_labels = all_labels[{{1, i % batch * batch_size}}]
        end

        train_labels = torch.Tensor(train_labels) + 1
        if gpu then
            train_labels = train_labels:cuda()
        end
        local p4 = os.clock()
        function feval(params)
            dl_dx:zero()
            if format == 'conv' then
                train_images = torch.reshape(train_images, train_images:size(1), 1, train_images:size(2), train_images:size(3))
            end
            local p5 = os.clock()
            local output = net:forward(train_images)
            local p6 = os.clock()
            local err = criterion:forward(output, train_labels)
            local p7 = os.clock()
            local gradOut = criterion:backward(output, train_labels)
            local p8 = os.clock()
            net:backward(train_images, gradOut)
            local p9 = os.clock()
            --[[
            print(string.format("net forward time: %.4f", p6 - p5))
            print(string.format("criterion forward time: %.4f", p7 - p6))
            print(string.format("cirterion backward time: %.4f", p8 - p7))
            print(string.format("net backward time: %.4f", p9 - p8))
            ]]
            return err, dl_dx
        end
        _, loss = optim.sgd(feval, x, config)
        --[[
        print(string.format("copy time: %.4f", p2 - p1))
        print(string.format("normalize time: %.4f", p3 - p2))
        print(string.format("copy label time: %.4f", p4 - p3))
        ]]
        if i % 10 == 0 then
        --    print(dl_dx)
            print(string.format("iterations %d, current error: %f", i, loss[1]))
        end
        if i % 1000 == 0 then
            torch.save(string.format("inter-model/iter-%d.model", i), net)
        end
    end
else
    net = torch.load("inter-model/iter-4000.model")
    if gpu then
        net = net:cuda()
    end
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
labels = all_labels + 1
if gpu then
    images = images:cuda()
    labels = labels:cuda()
end
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


