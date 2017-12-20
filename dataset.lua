require 'torch'
dataset = {}

dataset.read_labels = function(filename, start,  count)
    local  file = io.open(filename, "r")
    local num =  file:read(4)
    local magic = string.unpack(">i4", num)
    num = file:read(4)
    local all = string.unpack(">i4", num)
    if filename == "train-labels-idx1-ubyte" and magic ~= 2049 and all ~= 60000 then
        error("error magic or count")
    end
    local res = {}
    file:seek("cur", start)
    for i=1, count do
        num = file:read(1)
        ubyte = string.unpack(">I1", num)
        res[#res + 1] = ubyte
    end
    file:close()
    return res
end

dataset.nread_labels = function(filename, start, count, format)
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


dataset.read_images = function(filename, start, count, format)
    local file = io.open(filename, "r")
    local num = file:read(4)
    local magic = string.unpack(">i4", num)
    num = file:read(4)
    local all = string.unpack(">i4", num)
    local width = file:read(4)
    local height = file:read(4)
    width = string.unpack(">i4", width); height = string.unpack(">i4", height)

    local res = {}
    file:seek("cur", start * width * height)
    if format == 'fullconnect' then
        for i = 1, count do
            image = {}
            for i = 1, width * height do
                ubyte = file:read(1)
                ubyte = string.unpack(">I1", ubyte)
                image[#image + 1] = ubyte
            end
            res[#res + 1] = image
        end
    else
         for i = 1, count do
            image = {}
            for i = 1, height do
                row = {}
                for i = 1, width do
                    ubyte = file:read(1)
                    ubyte = string.unpack(">I1", ubyte)
                    row[#row + 1] = ubyte
                end
                image[#image + 1] = row
            end
            res[#res + 1] = image
        end
    end
    file:close()
    return res
end

