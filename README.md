# mnist-with-torch
train mnist with lua-torch

## Requirement
- LUA 5.3
- torch7

## Usage

```th train.lua -gpu -network fullconnect -cudnn -learningRate 1e-1```, test presicion could 97% or so(4200 iterations)

or

```th train.lua -gpu -network conv -cudnn -learningRate 1e-3```, test presicion could be 95% or so(4200 iterations)
