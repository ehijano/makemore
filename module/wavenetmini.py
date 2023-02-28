import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

torch.manual_seed(42)

def get_training_data(words, char_index_map, block_size = 3):
    x, y = [], []
    for w in words:
        context = [0 for i in range(block_size)]
        for ch in w + '.':
            idx = char_index_map[ch]
            x.append(context)
            y.append(idx)
            context = context[1:] + [idx]

    return torch.tensor(x), torch.tensor(y)


class Linear:

    def __init__(self, fan_in, fan_out, bias = True, gain = 5/3):
        self.weight = torch.randn((fan_in, fan_out)) * gain / (fan_in)**0.5
        self.bias = torch.zeros(fan_out) if bias else None

    def __call__(self, x):
        self.out = x @ self.weight
        if self.bias is not None:
            self.out += self.bias
        return self.out
    
    def parameters(self):
        return [self.weight] + ([] if self.bias is None else [self.bias])
    

class BatchNorm1D:
    def __init__(self, dim, eps = 1e-5, momentum = 0.1):
        self.eps = eps
        self.dim = dim
        self.momentum = momentum
        self.training = False

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):

        if self.training:
            dims = tuple([i for i in range(x.ndim - 1)])
            xmean = x.mean(dims, keepdim = True) # batch mean
            xvar = x.var(dims, keepdim = True) # batch std
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out  = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * xmean
                self.running_var = (1-self.momentum) * self.running_var + self.momentum * xvar

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Tanh():
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []

class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.weight = torch.randn((num_embeddings, embedding_dim))

    def __call__(self, IX):
        self.out = self.weight[IX]
        if len(self.out.shape) == 2 :
            self.out = self.out.view(1, self.out.shape[0], self.out.shape[1])
        return self.out
    
    def parameters(self):
        return [self.weight]
    
class Flatten:
    def __call__(self, x):
        self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self):
        return []
    
class FlattenConsecutive:
    def __init__(self, n):
        self.n = n

    def __call__(self, x):
        batch, block, embedding = x.shape
        x = x.view(batch, block // self.n , embedding * self.n) # Instead of block//self.n, -1 should also work.

        if x.shape[1] == 1:
            x = x.squeeze(1)

        self.out = x
        return self.out
    
    def parameters(self):
        return []


class Sequential:
    def __init__(self, layers):
        self.layers = layers

        # last layer less confident:
        with torch.no_grad():
            self.layers[-1].weight += 0.1
    
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        self.out = x
        return self.out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# x = self.C[x_mini].view(x_mini.shape[0], self.block_size*self.emb_dim)
# Now:
# x = flatten(embedding(x_mini))


class WaveNetMini:
    def __init__(self, char_index_map, chars, block_size = 3, emb_dim = 2, hidden_layer_size = 100, neighbour_number = 2):
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.hidden_layer_size = hidden_layer_size
        self.char_index_map = char_index_map
        self.chars = chars
        self.vocab_size = len(chars)
        self.neighbour_number = neighbour_number

        self.reset_params()

    def reset_params(self):
        
        
        """self.sequential = Sequential(
        [
            Embedding(self.vocab_size, self.emb_dim)
            , FlattenConsecutive(self.neighbour_number)
            , Linear( self.emb_dim * self.neighbour_number , self.hidden_layer_size, bias = False ), BatchNorm1D(self.hidden_layer_size), Tanh()
            , FlattenConsecutive(self.neighbour_number)
            , Linear( self.hidden_layer_size * self.neighbour_number , self.hidden_layer_size, bias = False ), BatchNorm1D(self.hidden_layer_size), Tanh()
            , FlattenConsecutive(self.neighbour_number)
            , Linear( self.hidden_layer_size * self.neighbour_number , self.hidden_layer_size, bias = False ), BatchNorm1D(self.hidden_layer_size), Tanh()
            , Linear( self.hidden_layer_size, self.vocab_size)
        ]
        )"""
        

        layers = [Embedding(self.vocab_size, self.emb_dim)
            , FlattenConsecutive(self.neighbour_number)
            , Linear( self.emb_dim * self.neighbour_number , self.hidden_layer_size, bias = False )]

        batch_size = self.block_size // self.neighbour_number

        while(batch_size > 1):
            layers = layers + [
                FlattenConsecutive(self.neighbour_number)
            , Linear( self.hidden_layer_size * self.neighbour_number , self.hidden_layer_size, bias = False )
            , BatchNorm1D(self.hidden_layer_size), Tanh()]

            batch_size = batch_size // self.neighbour_number
        
        layers = layers + [Linear( self.hidden_layer_size, self.vocab_size)]

        self.sequential = Sequential(layers)


        self.parameters = self.sequential.parameters()

        for p in self.parameters:
            p.requires_grad = True

    def set_training(self, training):
        for l in self.sequential.layers:
            if isinstance(l, BatchNorm1D):
                l.training = training
    
    def train(self, X, Y, episodes = 100, lr = 0.1, minibatch = 32, ls = None):
        self.set_training(True)

        history = []

        for ep in range(episodes):

            # Run training only on random minibatches. Doesnt use the exact gradient. Only the one computed from the minibatch
            ix = torch.randint(0, X.shape[0], (minibatch,))
            x_mini, y_mini = X[ix], Y[ix]

            # Forward
            x = self.sequential(x_mini)

            self.loss = F.cross_entropy(x, y_mini) # loss = F.cross_entropy(logits, Y)

            history.append(self.loss.item())

            # Backward
            #for l in self.layers:
            #    l.out.retain_grad()  # Remove after done plotting!
            for p in self.parameters:
                p.grad = None
            self.loss.backward()

            if ls is not None:
                lr = ls(ep)

            # update
            for p in self.parameters:
                p.data += - lr * p.grad

        self.set_training(False)
        return history
    
    @torch.no_grad()
    def predict(self,context_init):
        context = [c for c in context_init]
        string = ''.join(context)
        done = False

        while not done:

            #x = self.C[[self.char_index_map[c] for c in context]].view(-1, self.block_size*self.emb_dim) # emb = C[X]
            x =  [self.char_index_map[c] for c in context]

            x = self.sequential(x)

            p = F.softmax(x, dim=1)


            #print(p)


            # Sample from proba
            random_idx = torch.multinomial(p, num_samples=1, replacement=True).item()

            character = self.chars[random_idx]
            string = string + character
            context = context[1:] + [character]

            #print(string)

            if character == '.':
                done = True
        

        return string
    
    @torch.no_grad()
    def split_loss(self, xs, ys):
        #x = self.C[xs].view(-1, self.block_size*self.emb_dim) # emb = C[X]
        x = self.sequential(xs)
        loss = F.cross_entropy( x , ys) 
        return loss.item()


