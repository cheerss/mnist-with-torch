require 'dataset'
require 'cutorch'
require 'cunn'

normalize = nn.BatchNormalization(28 *28):cuda()
train_images = dataset.read_images("train-images-idx3-ubyte", 0, 100)
train_images = torch.Tensor(train_images):cuda()
print(train_images:size())
train_images = normalize:forward(train_images)
image3 = train_images[3]
for i = 1, 28 do
    for j = 1, 28 do
        io.write(string.format("%.2f ", image3[(i-1)*28 + j]))
    end
    io.write("\n")
end

