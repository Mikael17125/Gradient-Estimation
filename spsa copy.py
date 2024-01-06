import torch

class SPSA:
    def __init__(self, model, criterion):
        self.sp_avg = 5
        self.b1 = 0.9
        self.m1 = 0
        self.o, self.c, self.a, self.alpha, self.gamma = 1.0, 0.01, 0.01, 0.4, 0.1
        
        self.model = model
        self.criterion = criterion
        
        self.est_type = 'spsa'
        
        for name, param in self.model.named_parameters():
            param.requires_grad_(False)
        
    def estimate(self, epoch, images, labels, ck):

        ghats = []
        ak = self.a / ((self + self.o) ** self.alpha)
        ck = self.c / (epoch ** self.gamma)
        w = torch.nn.utils.parameters_to_vector(self.model.coordinator.dec.parameters())
        
        for _ in range(self.sp_avg):
            
            p_side = (torch.rand(len(w)).reshape(-1,1) + 1)/2
            samples = torch.cat([p_side,-p_side], dim=1)
            perturb = torch.gather(samples, 1, torch.bernoulli(torch.ones_like(p_side)/2).type(torch.int64)).reshape(-1).cuda()
            del samples; del p_side
            
            w_r = w + ck * perturb
            w_l = w - ck * perturb
            
            torch.nn.utils.vector_to_parameters(w_r, self.model.coordinator.dec.parameters())
            output_r = self.model(images)
            
            torch.nn.utils.vector_to_parameters(w_l, self.model.coordinator.dec.parameters())
            output_l = self.model(images)

            loss_r = self.criterion(output_r, labels)
            loss_l = self.criterion(output_l, labels)
            
            ghat = (loss_r - loss_l)/((2*ck)*perturb)
            ghats.append(ghat.reshape(1, -1))
            
        if self.sp_avg == 1: pass
        else: ghat = torch.cat(ghats, dim=0).mean(dim=0)
                                        
        if self.est_type == 'spsa-gc':
            if epoch > 1: self.m1 = self.b1 * self.m1 + ghat
            else: self.m1 = ghat
            accum_ghat = ghat + self.b1 * self.m1
        elif self.est_type == 'spsa':
            accum_ghat = ghat
        else:
            raise ValueError

        w_new = w - ak * accum_ghat

        torch.nn.utils.vector_to_parameters(w_new, self.model.coordinator.dec.parameters())