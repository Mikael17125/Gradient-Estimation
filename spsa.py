import torch

class SPSA:
    def __init__(self):
        self.sp_avg = 5
        self.o, self.c, self.a, self.alpha, self.gamma = 1.0, 0.01, 0.01, 0.4, 0.1
        
        
    def estimate(self, step, model, loss_fn, image, label):
        w = torch.nn.utils.parameters_to_vector(model.autoencoder.parameters())
        ak = self.a/(((step+1) + self.o)**self.alpha)
        ck = self.c/((step+1)**self.gamma)
        
        ghats = []

        for idx in range(self.sp_avg):
            
            p_side = (torch.rand(len(w)).reshape(-1,1) + 1)/2
            samples = torch.cat([p_side,-p_side], dim=1)
            perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)).reshape(-1).cuda()
            del samples; del p_side
            
            w_r = w + ck * perturb
            w_l = w - ck * perturb
            
            torch.nn.utils.vector_to_parameters(w_r, model.autoencoder.parameters())
            output_r = model(image)
            
            torch.nn.utils.vector_to_parameters(w_l, model.autoencoder.parameters())
            output_l = model(image)

            loss_r = loss_fn(output_r, label)
            loss_l = loss_fn(output_l, label)
            
            ghat = (loss_r - loss_l)/((2*ck)*perturb)
            ghats.append(ghat.reshape(1, -1))
            
        if self.sp_avg == 1: pass
        else: ghat = torch.cat(ghats, dim=0).mean(dim=0)
        
        w_new = w - ak * ghat
        torch.nn.utils.vector_to_parameters(w_new, model.autoencoder.parameters())
                
        loss = ((loss_r + loss_l)/2)
                
        return loss