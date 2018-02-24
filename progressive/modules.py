from torch import nn

class ProgressiveGenerator(nn.Module):
    def __init__(self,*blocks):
        self.blocks = blocks
        self.cur_block = 0

    def forward(self,input,alpha=1.):
        tmp = 0
        upsample = False
        for i in range(0,self.cur_block+1):
            input = self.blocks[i](input,last=(i==self.cur_block))
            if alpha<1. and i==self.cur_block-1:
                tmp = input
                upsample = True

        if upsample:
            tmp = nn.functional.upsample(tmp,input.size())
            input = alpha*input+(1.-alpha)*tmp
        return input

class ProgressiveDiscriminator(nn.Module):
    def __init__(self,*blocks):
        self.blocks = blocks
        self.cur_block = len(self.blocks)-1

    def forward(self,input):
        for i in range(self.cur_block,len(self.blocks)+1):
            input = self.blocks[i](input,first=(i==self.cur_block))
        return input

class ProgressiveGeneratorBlock(nn.Module):
    def __init__(self,intermediate_sequence,out_sequence):
        self.intermediate_sequence = intermediate_sequence
        self.out_sequence = out_sequence

    def forward(self,input,last=False):
        out = self.intermediate_sequence(input)
        if last:
            out = self.out_sequence(out)
        return out

class ProgressiveDiscriminatorBlock(nn.Module):
    def __init__(self,intermediate_sequence,in_sequence):
        self.intermediate_sequence = intermediate_sequence
        self.in_sequence = in_sequence

    def forward(self,input,first=False):
        if first:
            input = self.in_sequence(input)
        out = self.intermediate_sequence(input)
        return out
