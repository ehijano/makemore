import torch
import torch.nn.functional as F

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


class Model:
    def __init__(self, char_index_map, chars, block_size = 3, emb_dim = 2, layer_out_size = 100):
        self.block_size = block_size
        self.emb_dim = emb_dim
        self.layer_out_size = layer_out_size
        self.char_index_map = char_index_map
        self.chars = chars

        self.g = torch.Generator().manual_seed(2147483647)

        self.reset_params()

    def reset_params(self):

        self.C = torch.randn((27, self.emb_dim), generator = self.g)
        # Layer 1
        self.W1 = torch.randn((self.block_size*self.emb_dim , self.layer_out_size), generator = self.g)
        self.b1 = torch.randn(self.layer_out_size, generator = self.g)
        # Layer 2
        self.W2 = torch.randn((self.layer_out_size, 27), generator = self.g)
        self.b2 = torch.randn(27, generator = self.g)

        for p in [self.C, self.W1, self.b1, self.W2, self.b2]:
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
        # Run training only on random minibatches. Doesnt use the exact gradient. Only the one computed from the minibatch
        ix = torch.randint(0, X.shape[0], (minibatch,))
        x, y = X[ix], Y[ix]

        history = []

        for _ in range(episodes):
            # Forward
            emb = self.C[x] # emb = C[X]
            h = torch.tanh( emb.view(-1, self.block_size*self.emb_dim) @ self.W1 + self.b1 )
            logits = h @ self.W2 + self.b2
            self.loss = F.cross_entropy(logits, y) # loss = F.cross_entropy(logits, Y)

            history.append(self.loss.item())

            # Backward
            self.loss.backward()

            # update
            for p in [self.C, self.W1, self.b1, self.W2, self.b2]:
                p.data += - lr * p.grad

        return history
    
    def predict(self,context_init):
        context = [c for c in context_init]
        string = ''.join(context)
        done = False

        while not done:
            emb = self.C[[self.char_index_map[c] for c in context]] # emb = C[X]
            h = torch.tanh( emb.view(-1, self.block_size*self.emb_dim) @ self.W1 + self.b1 )
            logits = h @ self.W2 + self.b2
            #logcounts = logits.exp()
            #p = logcounts / logcounts.sum(1, keepdims = True)
            p = F.softmax(logits, dim=1)

            # Sample from proba
            random_idx = torch.multinomial(p, num_samples=1, replacement=True).item()

            character = self.chars[random_idx]
            string = string + character
            context = context[1:] + [character]

            #print(string)

            if character == '.':
                done = True
        

        return string