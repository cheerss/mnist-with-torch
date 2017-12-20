require 'torch'
require 'paths'
dataset = {}

function download_model()
    local addr = "http://yann.lecun.com/exdb/mnist/"
    local files = {}
    files[1] = "train-images-idx3-ubyte"
    files[2] = "train-labels-idx1-ubyte"
    files[3] = "t10k-images-idx3-ubyte"
    files[4] = "t10k-labels-idx1-ubyte"
    for i = 1, 4 do
        if not paths.filep(files[i]) then
            os.execute('wget ' .. addr .. files[i] .. '.gz')
            os.execute('gzip -d ' .. files[i] .. '.gz')
        end
    end
end

dataset.nread_labels = function(filename, start, count, format)
    download_model()
    local file = torch.DiskFile(filename, "r")
    file:binary()
    file:bigEndianEncoding()

    local magic = file:readInt()
    local all_count = file:readInt()
    file:seek(9 + start)
    local res = file:readByte(count)
    return torch.totable(res)
end

dataset.nread_images = function(filename, start, count, format)
    download_model()
    local  file = torch.DiskFile(filename, "r")
    file:binary()
    file:bigEndianEncoding()
    local magic =  file:readInt()
    local all_count = file:readInt()
    local width = file:readInt(); local height = file:readInt()
    file:seek(17 + start * height * width)
    local res = file:readByte(count * height * width)
    return torch.totable(res)
end
