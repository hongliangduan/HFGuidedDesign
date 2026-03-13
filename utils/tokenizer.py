import torch

class Tokenizer:
    def __init__(self):
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'  
        self.un_acids = ['B', 'Z', 'X', 'J', 'O', 'U'] 

        self.vocab = list(self.amino_acids) + self.un_acids

        self.K = len(self.vocab)
        self.vocab_size = self.K  

        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

    def tokenize(self, sequence: str): 
        return [self.token_to_id.get(token) for token in sequence]
    
    def untokenize(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().tolist()  
        return ''.join([self.id_to_token[idx] for idx in token_ids])

    def one_hot(self, token_ids, pad_id):
        if isinstance(token_ids, list):
            token_ids = torch.LongTensor(token_ids)
        elif isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.long()
        else:
            raise TypeError(f"Unsupported type: {type(token_ids)}")
        
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
            squeeze_first = True
        else:
            squeeze_first = False
        
        B, L = token_ids.shape
        device = token_ids.device
        one_hot = torch.zeros(B, L, self.K, device=device, dtype=torch.float32)
        

        mask = (token_ids != pad_id)
        
        safe_token_ids = token_ids.clone()
        safe_token_ids[~mask] = 0 
        
        one_hot.scatter_(2, safe_token_ids.unsqueeze(2), 1.)
        
        one_hot *= mask.unsqueeze(2).float()
        
        if squeeze_first:
            one_hot = one_hot.squeeze(0)
        
        return one_hot
    
if __name__ == "__main__":
    tokenizer = Tokenizer()
    seq = "ACDXYZ"
    ids = tokenizer.tokenize(seq)
    print("Token IDs:", ids)
    print("Untokenized:", tokenizer.untokenize(ids))
    print("One-hot:\n", tokenizer.one_hot(ids))