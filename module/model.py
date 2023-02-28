import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

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
        self.training = True

        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):

        if self.training:
            xmean = x.mean(0, keepdim = True) # batch mean
            xvar = x.std(0, keepdim = True) # batch std
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x-xmean)/torch.sqrt(xvar + self.eps)
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


class Model:
    def __init__(self, char_index_map, chars, block_size = 3, emb_dim = 2, hidden_layer_size = 100):
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.hidden_layer_size = hidden_layer_size
        self.char_index_map = char_index_map
        self.chars = chars
        self.vocab_size = len(chars)

        self.g = torch.Generator().manual_seed(2147483647)

        self.reset_params()

    def reset_params(self):

        self.C = torch.randn((self.vocab_size, self.emb_dim), generator = self.g)

        self.layers = [
            Linear( self.emb_dim * self.block_size , self.hidden_layer_size ), BatchNorm1D(self.hidden_layer_size), Tanh()
            , Linear( self.hidden_layer_size , self.hidden_layer_size ), BatchNorm1D(self.hidden_layer_size), Tanh()
            , Linear( self.hidden_layer_size , self.hidden_layer_size ), BatchNorm1D(self.hidden_layer_size), Tanh()
            , Linear( self.hidden_layer_size , self.hidden_layer_size ), BatchNorm1D(self.hidden_layer_size), Tanh()
            , Linear( self.hidden_layer_size, self.vocab_size)
        ]
        
        self.parameters = []
        for l in self.layers:
            self.parameters = self.parameters + l.parameters()
            for p in l.parameters():
                p.requires_grad = True

    def find_lr(self, X, Y, episodes = 10, minibatch = 32): # X here should be the validation set
        number_cases = 1000
        lrs_exponent = torch.linspace(-3, 0, number_cases)
        lrs = 10**lrs_exponent

        results = []

        for i in range(number_cases):
            self.reset_params()

            self.train(X, Y, episodes = episodes, lr = lrs[i])

            results.append(self.loss.item())

        return lrs_exponent, results
    
    def train(self, X, Y, episodes = 100, lr = 0.1, minibatch = 32):

        history = []

        for _ in range(episodes):

            # Run training only on random minibatches. Doesnt use the exact gradient. Only the one computed from the minibatch
            ix = torch.randint(0, X.shape[0], (minibatch,))
            x_mini, y_mini = X[ix], Y[ix]

            # Forward
            x = self.C[x_mini].view(x_mini.shape[0], self.block_size*self.emb_dim) # minibatch, block, emb -> minibatch, block * embed
            print(x.shape)
            for layer in self.layers:
                x = layer(x)
            self.loss = F.cross_entropy(x, y_mini) # loss = F.cross_entropy(logits, Y)

            history.append(self.loss.item())

            # Backward
            for l in self.layers:
                l.out.retain_grad()  # Remove after done plotting!
            for p in self.parameters:
                p.grad = None
            self.loss.backward()

            # update
            for p in self.parameters:
                p.data += - lr * p.grad

        return history
    
    @torch.no_grad()
    def predict(self,context_init):
        context = [c for c in context_init]
        string = ''.join(context)
        done = False

        while not done:

            x = self.C[[self.char_index_map[c] for c in context]].view(-1, self.block_size*self.emb_dim) # emb = C[X]
            for layer in self.layers:
                x = layer(x)

            p = F.softmax(x, dim=1)

            # Sample from proba
            random_idx = torch.multinomial(p, num_samples=1, replacement=True, generator=self.g).item()

            character = self.chars[random_idx]
            string = string + character
            context = context[1:] + [character]

            #print(string)

            if character == '.':
                done = True
        

        return string
    
    @torch.no_grad()
    def split_loss(self, xs, ys):
        x = self.C[xs].view(-1, self.block_size*self.emb_dim) # emb = C[X]
        for layer in self.layers:
            x = layer(x)
        loss = F.cross_entropy( x , ys) 
        return loss.item()


    def visualize_layers(self, layer_type):
        plt.figure( figsize = (20, 5) )
        legends = []
        for i, layer in enumerate(self.layers[:-1]):
            if layer_type == 'linear' and isinstance(layer, Linear):
                t = layer.out.grad
                print(f'Layer {i}: gradient mean: {t.mean():.4f}, std: {t.std():.4f}')
                hy, hx = torch.histogram(t, density=True)
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'Linear layer {i}')
        plt.legend(legends)
        plt.title('Gradient distributions')
        plt.show()


    def visualize_activations(self, layer_type):
        plt.figure( figsize = (20, 5) )
        legends = []
        for i, layer in enumerate(self.layers[:-1]):
            if layer_type == 'tanh' and isinstance(layer, Tanh):
                t = layer.out
                hy, hx = torch.histogram(t, density=True)
                print(f'Tanh Layer {i}: mean {t.mean():.4f} std {t.std():.4f} saturated {100*(t.abs() > 0.97).float().mean():.4f}%')
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'Activation layer {i}')
        plt.legend(legends)
        plt.title('Activations distributions')
        plt.show()


    def visualize_weight_gradients(self):

        legends = []
        for i, p in enumerate(self.parameters):
            t = p.grad
            if p.ndim == 2:
                hy, hx = torch.histogram(t, density=True)
                print(f'Weight {i}: dims {tuple(p.shape)} mean {t.mean():.4f} std {t.std():.4f} grad/data std ratio {t.std()/p.std():.4f}%')
                plt.plot(hx[:-1].detach(), hy.detach())
                legends.append(f'Parameter {i}')

        plt.legend(legends)
        plt.title('Weight Gradient distributions')
        plt.show()


