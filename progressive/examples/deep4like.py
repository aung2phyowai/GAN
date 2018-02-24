import braindecode
from torch import nn
from GAN.progressive.layers import Reshape
from GAN.progressive.modules import ProgressiveGenerator,ProgressiveGeneratorBlock,
                                    ProgressiveDiscriminator,ProgressiveDiscriminatorBlock
from GAN.GAN.modules import LayerNorm

input_size = 972
n_chans = 1

def create_disc_blocks():
    blocks = []
    tmp_block = ProgressiveGeneratorBlock(
                              nn.Sequential(nn.Conv1d(25,25,9,padding=4),
                              nn.LayerNorm(25),
                              nn.LeakyReLU(0.2),
                              nn.Conv1d(25,25,3,stride=3)),
                              nn.LayerNorm(25),
                              nn.LeakyReLU(0.2),
                              nn.Sequential(nn.Conv2d(1,25,(9,n_chans),padding=4),
                              Reshape([[0],[1],[2]])
                              )
    block.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
                              nn.Sequential(nn.Conv1d(25,50,9,padding=4),
                              nn.LayerNorm(50),
                              nn.LeakyReLU(0.2),
                              nn.Conv1d(50,50,3,stride=3)),
                              nn.LayerNorm(50),
                              nn.LeakyReLU(0.2),
                              nn.Sequential(nn.Conv2d(1,50,(9,n_chans),padding=4),
                              Reshape([[0],[1],[2]])
                              )
    block.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
                              nn.Sequential(nn.Conv1d(50,100,9,padding=4),
                              nn.LayerNorm(100),
                              nn.LeakyReLU(0.2),
                              nn.Conv1d(100,100,3,stride=3)),
                              nn.LayerNorm(100),
                              nn.LeakyReLU(0.2),
                              nn.Sequential(nn.Conv2d(1,100,(9,n_chans),padding=4),
                              Reshape([[0],[1],[2]])
                              )
    block.append(tmp_block)
    tmp_block = ProgressiveGeneratorBlock(
                              nn.Sequential(nn.Conv1d(100,200,9,padding=4),
                              nn.LayerNorm(200),
                              nn.LeakyReLU(0.2),
                              nn.Conv1d(200,200,3,stride=3),
                              nn.LayerNorm(200),
                              nn.LeakyReLU(0.2),
                              Reshape([[0],-1]),
                              nn.Linear(200*12,1)),
                              nn.Sequential(nn.Conv2d(1,200,(9,n_chans),padding=4),
                              Reshape([[0],[1],[2]])
                              )
    block.append(tmp_block)
