import torch
import torch.nn as nn
from entmax import sparsemax

_EPSILON = 1e-6

def _vector_norms(v:torch.Tensor)->torch.Tensor:
    """ Computes the vector norms
    Args:
        v: The vector from which there must be calculated the norms

    Returns:
            A tensor containing the norms of input vector v
    """

    squared_norms = torch.sum(v * v, dim=1, keepdim=True)
    return torch.sqrt(squared_norms + _EPSILON)

class MLP(nn.Module):
    '''
    Multi-layer perceptron class
    '''
    def __init__(self, input_size:int, hidden_size:int, output_size:int):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x:torch.Tensor):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        return output

class MemoryWrapLayer(nn.Module):

    def __init__(self, encoder_output_dim:int, output_dim:int):
        super(MemoryWrapLayer, self).__init__()

        final_input_dim = encoder_output_dim*2 
        self.fc = MLP(final_input_dim,final_input_dim*2,output_dim)

        
    def forward(self, encoder_output:torch.Tensor, memory_set:torch.Tensor, return_weights:bool=False)->torch.Tensor:
        """Forward call of MemoryWrap.
        Args:
            input: A tensor of dimensions [b,dim] where dim is the dimension required by the encoder
            memory_set: Memory set. A tensor of dimension [m,dim] where m is the number of examples in memory
            parsed_memory: a flag to indicate if the memory set is already parsed by the encoder. It is useful
            to reduce the testing time if you fix the memory or if you parse the whole training set.
        Returns:
            A tuple `(output, content-weight)` where `output`
            is the output tensor, `content_weights` is a tensor containing the
            read weights for sample in memory. If return_weights is False, then
            only `output` is returned.
        """

        encoder_norm = encoder_output / _vector_norms(encoder_output)
        memory_norm = memory_set / _vector_norms(memory_set)
        sim = torch.mm(encoder_norm,memory_norm.transpose(0,1))
        content_weights = sparsemax(sim,dim=1)
        memory_vector = torch.matmul(content_weights,memory_set)
        final_input = torch.cat([encoder_output,memory_vector],1)
        output = self.fc(final_input)

        if return_weights:
            return output, content_weights
        else: 
            return output

class BaselineMemory(nn.Module):

    def __init__(self, encoder_output_dim:int, output_dim:int):
        super(BaselineMemory, self).__init__()

        final_input_dim = encoder_output_dim
        self.fc = MLP(final_input_dim,final_input_dim*2,output_dim)

        
    def forward(self, encoder_output:torch.Tensor, memory_set:torch.Tensor, return_weights:bool=False)->torch.Tensor:
        """Forward call of MemoryWrap.
        Args:
            input: A tensor of dimensions [b,dim] where dim is the dimension required by the encoder
            memory_set: Memory set. A tensor of dimension [m,dim] where m is the number of examples in memory
            parsed_memory: a flag to indicate if the memory set is already parsed by the encoder
        Returns:
            A tuple `(output, content-weight)` where `output`
            is the output tensor, `content_weights` is a tensor containing the
            read weights for sample in memory. If return_weights is False, then
            only `output` is returned.
        """
        encoder_norm = encoder_output / _vector_norms(encoder_output)
        memory_norm = memory_set / _vector_norms(memory_set)
        sim = torch.mm(encoder_norm,memory_norm.transpose(0,1))
        content_weights = sparsemax(sim,dim=1)
        memory_vector = torch.matmul(content_weights,memory_set)
        output = self.fc(memory_vector)

        if return_weights:
            return output, content_weights
        else: 
            return output