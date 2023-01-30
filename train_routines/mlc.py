#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train_routines.basemodel import *

#Adapted from: https://github.com/microsoft/MLC
class MetaNet(nn.Module):
    def __init__(self, hx_dim, cls_dim, h_dim, num_classes, args):
        super().__init__()
        self.num_classes = num_classes        
        self.in_class = self.num_classes 
        self.hdim = h_dim
        self.cls_emb = nn.Embedding(self.in_class, cls_dim)

        if args.base_model == 'logistic_regression': 
            hx_dim = int(17*21) #17 amino acid input. one-hot encoding of 21 amino acid vocabulary

        in_dim = hx_dim + cls_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, self.hdim),
            nn.Tanh(),
            nn.Linear(self.hdim, num_classes, bias=True) 
        )

        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.cls_emb.weight)
        nn.init.xavier_normal_(self.net[0].weight)
        nn.init.xavier_normal_(self.net[2].weight)
        nn.init.xavier_normal_(self.net[4].weight)

        self.net[0].bias.data.zero_()
        self.net[2].bias.data.zero_()

        assert self.in_class == self.num_classes, 'In and out classes conflict!'
        self.net[4].bias.data.zero_()

    def get_alpha(self):
        return torch.zeros(1)

    def forward(self, hx, y):
        y_emb = self.cls_emb(y)
        print(f'y_emb.shape = {y_emb.shape}')
        hin = torch.cat([hx, y_emb], dim=-1)
        print(f'hin.shape = {hin.shape}')
        logit = self.net(hin)
        out = F.softmax(logit, -1)
        return out

def soft_cross_entropy(logit, pseudo_target, reduction='mean'):
    loss = -(pseudo_target * F.log_softmax(logit, -1)).sum(-1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise NotImplementedError('Invalid reduction: %s' % reduction)
        
def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


@torch.no_grad()
def update_params(params, grads, eta, opt, deltaonly=False, return_s=False):
    if isinstance(opt, torch.optim.SGD):
        return update_params_sgd(params, grads, eta, opt, deltaonly, return_s)
    else:
        raise NotImplementedError('Non-supported main model optimizer type!')

# be aware that the opt state dict returns references, hence take care not to
# modify them
def update_params_sgd(params, grads, eta, opt, deltaonly, return_s=False):
    # supports SGD-like optimizers
    ans = []

    if return_s:
        ss = []

    wdecay = opt.defaults['weight_decay']
    momentum = opt.defaults['momentum']
    dampening = opt.defaults['dampening']
    nesterov = opt.defaults['nesterov']

    for i, param in enumerate(params):
        dparam = grads[i] + param * wdecay # s=1
        s = 1

        if momentum > 0:
            try:
                moment = opt.state[param]['momentum_buffer'] * momentum
            except:
                moment = torch.zeros_like(param)

            moment.add_(dparam, alpha=1. -dampening) # s=1.-dampening

            if nesterov:
                dparam = dparam + momentum * moment # s= 1+momentum*(1.-dampening)
                s = 1 + momentum*(1.-dampening)
            else:
                dparam = moment # s=1.-dampening
                s = 1.-dampening

        if deltaonly:
            ans.append(- dparam * eta)
        else:
            ans.append(param - dparam  * eta)

        if return_s:
            ss.append(s*eta)

    if return_s:
        return ans, ss
    else:
        return ans


     
class MLC(BaseModel):

    def __init__(self, args, device):
            super().__init__(args, device)    

            self.vnet_lr = args.vnet_lr
            self.mlc_k_steps = args.mlc_k_steps
            
            cls_dim = 128
            hx_dim = 512
            hidden_dim = 64
            
            self.meta_net = MetaNet(hx_dim, cls_dim, hidden_dim, self.num_classes, args).to(self.device)            
            self.optimizer_vnet = torch.optim.Adam(self.meta_net.parameters(), lr=self.vnet_lr, weight_decay = 0, amsgrad=True, eps = 1e-8)

    def train_step(self, train_loader, meta_loader, epoch, tensorboard_writer,batch_size):
        
        f1 = F1(num_classes = self.num_classes).to(self.device)
        mcc = MatthewsCorrcoef(num_classes = self.num_classes).to(self.device)
        
        self.model.train()
        self.meta_net.train()
        
        length_train_loader = len(train_loader.dataset)
        meta_loader_iter = itertools.cycle(meta_loader)
        
        dw_prev = [0 for param in self.meta_net.parameters()] # 0 for previous iteration
        
        steps = epoch
        gradient_steps = self.mlc_k_steps
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
                      
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs_val, targets_val = next(meta_loader_iter)
            inputs_val, targets_val = inputs_val.to(self.device), targets_val.to(self.device)
            
            if self.scheduler != None: eta = self.scheduler.get_last_lr()[0]
            else: eta = self.learning_rate
            
            # compute gw for updating meta_net
            logit_g = self.get_predictions(self.model, inputs_val)
            loss_g = F.cross_entropy(logit_g, targets_val)        
            gw = torch.autograd.grad(loss_g, self.model.parameters())
            
            # given current meta net, get corrected label
            logit_s, x_s_h = self.mlc_get_predictions(self.model, inputs)
            
            print(f'LOGIT SHAPE = {x_s_h.shape}')
            pseudo_target_s = self.meta_net(x_s_h.detach(), targets)
            loss_s = soft_cross_entropy(logit_s, pseudo_target_s)
            
            f_param_grads = torch.autograd.grad(loss_s, self.model.parameters(), create_graph=True)    
            f_params_new, dparam_s = update_params(self.model.parameters(), f_param_grads, eta, self.optimizer, return_s=True)
            
            # 2. set w as w'
            f_param = []
            for i, param in enumerate(self.model.parameters()):
                f_param.append(param.data.clone())
                param.data = f_params_new[i].data # use data only as f_params_new has graph
            
            #training loss Hessian approximation
            Hw = 1
            
            #3. compute d_w' L_{D}(w')
            logit_g = self.get_predictions(self.model, inputs_val)
            loss_g = F.cross_entropy(logit_g, targets_val)
            
            gw_prime = torch.autograd.grad(loss_g, self.model.parameters())
            
            # 3.5 compute discount factor gw_prime * (I-LH) * gw.t() / |gw|^2
            tmp1 = [(1-Hw*dparam_s[i]) * gw_prime[i] for i in range(len(dparam_s))]
            gw_norm2 = (_concat(gw).norm())**2
            tmp2 = [gw[i]/gw_norm2 for i in range(len(gw))]
            gamma = torch.dot(_concat(tmp1), _concat(tmp2))
        
            # because of dparam_s, need to scale up/down f_params_grads_prime for proxy_g/loss_g
            Lgw_prime = [ dparam_s[i] * gw_prime[i] for i in range(len(dparam_s))]     
        
            proxy_g = -torch.dot(_concat(f_param_grads), _concat(Lgw_prime))
            
            # back prop on alphas
            self.optimizer_vnet.zero_grad()
            proxy_g.backward()
            
            # accumulate discounted iterative gradient
            for i, param in enumerate(self.meta_net.parameters()):
                if param.grad is not None:
                    param.grad.add_(gamma * dw_prev[i])
                    dw_prev[i] = param.grad.clone()
        
            if (steps+1) % (gradient_steps)==0: # T steps proceeded by main_net
                self.optimizer_vnet.step()
                dw_prev = [0 for param in self.meta_net.parameters()] # 0 to reset 
        
            # modify to w, and then do actual update main_net
            for i, param in enumerate(self.model.parameters()):
                param.data = f_param[i]
                param.grad = f_param_grads[i].data
            self.optimizer.step()
                
            batch_f1_mtr, batch_mcc_mtr = super().eval_training_batch(f1, mcc, loss_s, 
                                                                    epoch, batch_idx, logit_s, targets.data, length_train_loader, tensorboard_writer)

        if self.scheduler is not None:
                self.scheduler.step()
                
        with torch.no_grad():
            epoch_f1 = f1.compute()
            epoch_mcc = mcc.compute()
            tensorboard_writer.add_scalar('train/ F1', epoch_f1, epoch)
            tensorboard_writer.add_scalar('train/ MCC', epoch_mcc, epoch)
            tensorboard_writer.flush()
            
def weight_init(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)