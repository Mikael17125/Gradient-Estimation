import torch
from models import CustomCLIP
from utils import compute_accuracy

class SPSA:
    def __init__(self, device):
        self.sp_avg = 5
        self.b1 = 0.9
        self.m1 = 0
        
        self.device = device
        self.model = CustomCLIP().to(device)
        
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        
    def estimate(self, w, loss_fn, image, label, ck):

        ghats = []

        for idx in range(self.sp_avg):
            
            p_side = (torch.rand(len(w)).reshape(-1,1) + 1)/2
            samples = torch.cat([p_side,-p_side], dim=1)
            perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)).reshape(-1).cuda()
            del samples; del p_side
            
            w_r = w + ck * perturb
            w_l = w - ck * perturb
            
            torch.nn.utils.vector_to_parameters(w_r, self.model.coordinator.dec.parameters())
            output_r = self.model(image)
            
            torch.nn.utils.vector_to_parameters(w_l, self.model.coordinator.dec.parameters())
            output_l = self.model(image)

            loss_r = loss_fn(output_r, label)
            loss_l = loss_fn(output_l, label)
            
            ghat = (loss_r - loss_l)/((2*ck)*perturb)
            ghats.append(ghat.reshape(1, -1))
            
        if self.sp_avg == 1: pass
        else: ghat = torch.cat(ghats, dim=0).mean(dim=0)
                        
        loss = ((loss_r + loss_l)/2)
        acc = ((compute_accuracy(output_l, label)[0]+
                compute_accuracy(output_r, label)[0])/2).item()
        
        # import pdb; pdb.set_trace()
                
        return ghat, loss, acc