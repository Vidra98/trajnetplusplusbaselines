import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SocialNCE():
    '''
        Social NCE: Contrastive Learning of Socially-aware Motion Representations (https://arxiv.org/abs/2012.11717)
    '''
    def __init__(self, obs_length, pred_length, head_projection, encoder_sample, temperature, horizon, sampling):

        # problem setting
        self.obs_length = obs_length
        self.pred_length = pred_length

        # nce models
        self.head_projection = head_projection
        self.encoder_sample = encoder_sample

        # nce loss
        self.criterion = nn.CrossEntropyLoss()

        # nce param
        self.temperature = temperature
        self.horizon = horizon

        # sampling param
        self.sampling=sampling
        self.noise_local = 0.1
        self.min_seperation = 0.2
        self.agent_zone = self.min_seperation * torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [0.707, 0.707], [0.707, -0.707], [-0.707, 0.707], [-0.707, -0.707], [0.0, 0.0]])

    def spatial(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with spatial samples, i.e., samples are locations at a specific time of the future
            Input:
                batch_scene: coordinates of agents in the scene, tensor of shape [obs_length + pred_length, total num of agents in the batch, 2]
                batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
                batch_feat: encoded features of observations, tensor of shape [pred_length, scene, feat_dim]
            Output:                                                                      total num of agents in the batch
                loss: social nce loss
        '''

        # -----------------------------------------------------
        #               Visualize Trajectories 
        #       (Use this block to visualize the raw data)
        # -----------------------------------------------------

        # sample_pos,sample_neg=self._sampling_spatial(batch_scene,batch_split)    
        # for i in range(batch_split.shape[0] - 1):
        #     traj_primary = batch_scene[self.obs_length:-1, batch_split[i]] # [time, 2]
        #     traj_neighbor = batch_scene[self.obs_length:-1, batch_split[i]:batch_split[i+1]] # [time, num, 2]

        #     s_p = sample_pos[:, i] # [time, 2]
        #     nb_neighbours=batch_split[i+1]-batch_split[i]-1
        #     s_n = sample_neg[:, 9*(batch_split[i]-i):9*(batch_split[i]-i+nb_neighbours)] # [time, num, 2]

        #     plot_scene(s_p, s_n, fname='sample_pos_neg_{:d}.png'.format(i))
        #     plot_scene(traj_primary, traj_neighbor, fname='scene_{:d}.png'.format(i))
        # import pdb; pdb.set_trace()
        
        # #####################################################
        #           TODO: fill the following code
        # #####################################################

        # -----------------------------------------------------
        #               Contrastive Sampling 
        # -----------------------------------------------------

        ############ Shape for a random batch #################
        # batch_scene : torch.Size([obs_len+pred_len = 21, total_agent_in_batch = 43, 2]) 
        # batch_split : torch.Size([batch_size = 9])
        # batch_feat: torch.Size([pred_len = 12, total_agent_in_batch = 43, 128])

        # sample_pos : torch.Size([horizon = 4, num_scene = 8, 2])
        # sample_neg : torch.Size([horizon = 4, 9*(total_agent - primary_agent) = 315, 2])

        # emb_obs: torch.Size([pred_len = 12, num_scene = 8, head_dim = 8])
        # emb_pos: torch.Size([horizon = 4, num_scece = 8, head_dim = 8])
        # emb_neg: torch.Size([horizon = 4, 9*(total_agent - primary_agent) = 315, head_dim = 8])

        # query: torch.Size([pred_len = 12, num_scene = 8, head_dim = 8])
        # key_pos: torch.Size([horizon = 4, num_scece = 8, head_dim = 8])
        # key_neg: torch.Size([horizon = 4, 9*(total_agent - primary_agent) = 315, head_dim = 8])

        # sim_pos: torch.Size([4, 12, 8])
        # sim_neg: torch.Size([4, 315, 8])
        # logits: torch.Size([4, 2616])

        #######################################################

        #print('batch_scene :',batch_scene.shape)
        #print('batch_split :',batch_split.shape)
        #print('batch_feat:',batch_feat.shape)

        sample_pos,sample_neg=self._sampling_spatial(batch_scene,batch_split)
        #print('sample_pos :',sample_pos.shape)
        #print('sample_neg :',sample_neg.shape)
 
         # nan pre-process: set nan to 0 in forward to ensure grad
        #sample_neg.masked_fill_(~mask_valid_neg, 0.0)
        #sample_pos.masked_fill_(~mask_valid_pos, 0.0)
        
        # print('sample_pos :',sample_pos.shape)
        # print('sample_neg :',sample_neg.shape)

        # -----------------------------------------------------
        #              Lower-dimensional Embedding 
        # -----------------------------------------------------
        #set the index for the neighbors
        #neighbor_ID=torch.arange(batch_scene.shape[1]+1)
        #neighbor_ID[batch_split]=0
        #neighbor_ID=torch.nonzero(neighbor_ID!=0).view(-1) 

        PrimaryID=batch_split[0:-1] #get the ID from the primary trajectory

        #embedding 
        
        emb_obsv=self.head_projection(batch_feat[:,PrimaryID]) #TODO/TOASK : see difference with whole batch ???? # why it is of dim [:pred_length]
        emb_pos=self.encoder_sample(sample_pos)
        emb_neg=self.encoder_sample(sample_neg)
        #print('emb_obs:',emb_obsv.shape)
        #print('emb_pos:',emb_pos.shape)
        #print('emb_neg:',emb_neg.shape)

        #normalize the value in head_feature dimension   
        query = F.normalize(emb_obsv,dim=2)
        key_pos = F.normalize(emb_pos,dim=2)
        key_neg = F.normalize(emb_neg,dim=2)
        #print('query:',query.shape)
        #print('key_pos:',key_pos.shape)
        #print('key_neg:',key_neg.shape)

        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------
      
        sim_pos = (query * key_pos[:,None,:,:]).sum(dim=-1)
        sim_neg = (query[:self.horizon,None,:,:] * key_neg[:,:,None,:]).sum(dim=-1)
        
        size_pos=sim_pos.size()
        size_neg=sim_neg.size()

        
        sim_pos=sim_pos.reshape((size_pos[0]*size_pos[2],size_pos[1]))
        sim_neg=sim_neg.reshape((size_neg[0]*size_neg[2],size_neg[1]))
        
        #sim_pos = (query[:self.horizon] * key_pos).sum(dim=-1)
        #sim_neg = (query[:self.horizon].unsqueeze(2) * key_neg.unsqueeze(1)).sum(dim=-1)
        #sim_neg = torch.where(torch.isnan(sim_neg), torch.full_like(sim_neg, 0), sim_neg)
        #print('sim_pos:',sim_pos.shape)
        #print('sim_neg:',sim_neg.shape)

        
        #sim_pos_avg = sim_pos.mean(axis=1)
        #size_pos = sim_pos.size()
        #size_neg = sim_neg.size()
        #sim_pos = sim_pos.reshape(size_pos[0]*size_pos[1],-1)
        #sim_neg = sim_neg.reshape(size_neg[0]*size_neg[1],-1) 

        #print('sim_pos:',sim_pos.shape)
        #print('sim_neg:',sim_neg.shape)
        # -----------------------------------------------------
        #                       NCE Loss 
        # -----------------------------------------------------
        # logits
        
        logits = torch.cat([sim_pos,sim_neg],dim=1)/self.temperature
        logits = logits.reshape((logits.shape[0]),-1)
        
        #logits = torch.cat([sim_pos.view(size_neg[0],1), sim_neg_flat.view(size_neg[0],size_neg[1]*size_neg[2])], dim=1)/self.temperature

        #print('logits:',logits.shape)

        # loss

        labels = torch.zeros(logits.size(0), dtype=torch.long)
        # print('labels:',labels.shape)
        loss = self.criterion(logits, labels)
        #print("computed loss = ",loss.item())
        #import pdb; pdb.set_trace()

        return loss

    def event(self, batch_scene, batch_split, batch_feat):
        '''
            Social NCE with event samples, i.e., samples are spatial-temporal events at various time steps of the future
        '''
        raise ValueError("Optional")

    def _valid_check(self, sample_pos, sample_neg):
        '''
        Check validity of sample seeds, mask the frames that are invalid at the end of episodes
        '''
        
        mask_valid_pos = sample_pos.view(sample_pos.shape[0], -1).min(dim=1)[0] > 1e-2
        mask_valid_neg = sample_neg.view(sample_neg.shape[0], -1).min(dim=1)[0] > 1e-2
        print(sample_neg.view(sample_neg.shape[0], -1).shape)
        assert sample_neg[mask_valid_neg].min().item() > self.min_seperation
        return mask_valid_pos,mask_valid_neg
        
    def _sampling_spatial(self, batch_scene, batch_split):
        """
        test shape :
        input:
        batch_scene: coordinates of agents in the scene, tensor of shape [obs_length + pred_length, total num of agents in the batch, 2]
        batch_split: index of scene split in the batch, tensor of shape [batch_size + 1]
        output :
        sample_pos: torch.Size([horizon, nb_scene, 2])
        sample_neg: torch.Size([horizon, (nb_individuals-nb_scene)*9, 2])
        """
        gt_future = batch_scene[self.obs_length: self.obs_length+self.pred_length] 
        # #####################################################
        #           TODO: fill the following code
        # #####################################################
        
        # ----------------------------------------------------
        #                  Positive Samples
        # -----------------------------------------------------

        
        sample_pos=gt_future - batch_scene[self.obs_length-1:self.obs_length+self.pred_length-1]
        sample_pos=sample_pos[:self.horizon,batch_split[:-1]]
        sample_pos = torch.where(torch.isnan(sample_pos), torch.full_like(sample_pos, 0.0), sample_pos)
        sample_pos += torch.rand(sample_pos.size()).sub(0.5) * self.noise_local

        # -----------------------------------------------------
        #                  Negative Samples
        # -----------------------------------------------------    

        #dimension along with we want to the negative sample 
        concat_dim = 0 if self.horizon == 1 else 1
        
        #initial seed for negatif sample
        neg_seed= batch_scene[self.obs_length:self.obs_length+self.horizon, batch_split[0]+1:batch_split[0+1]]-batch_scene[self.obs_length-1:self.obs_length+self.horizon-1, batch_split[0]+1:batch_split[0+1]]
        #compute seed for each scene and concatenate it
        for i in range(1,batch_split.shape[0] - 1):
            neg_seed=torch.cat((neg_seed, batch_scene[self.obs_length:self.obs_length+self.horizon, batch_split[i]+1:batch_split[i+1]]-batch_scene[self.obs_length-1:self.obs_length+self.horizon-1, batch_split[i]+1:batch_split[i+1]]),concat_dim)

        neg_seed = torch.where(torch.isnan(neg_seed), torch.full_like(neg_seed, 0.0), neg_seed)  
        neg_seed+= torch.rand(neg_seed.size()).sub(0.5) * self.noise_local
        #create discomfort zone
        sample_territory = neg_seed[:, :, None, :] + self.agent_zone[None, None, :, :]
        sample_territory = sample_territory.view(sample_territory.size(0), sample_territory.size(1) * sample_territory.size(2), 2)
        sample_neg = sample_territory

        # nan pre-process: set nan to 0 in forward to ensure grad
        #mask_valid_pos,mask_valid_neg=self._valid_check(sample_pos,sample_neg)
        #sample_neg.masked_fill_(~mask_valid_neg, float('nan'))
        #sample_pos.masked_fill_(~mask_valid_pos, float('nan'))
        
        
        # -----------------------------------------------------
        #       Remove negatives that are too hard (optional)
        # -----------------------------------------------------

        # -----------------------------------------------------
        #       Remove negatives that are too easy (optional)
        # -----------------------------------------------------
        return sample_pos,sample_neg #, mask_valid_neg, mask_valid_pos

class EventEncoder(nn.Module):
    '''
        Event encoder that maps an sampled event (location & time) to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):

        super(EventEncoder, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.spatial = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state, time):
        emb_state = self.spatial(state)
        emb_time = self.temporal(time)
        out = self.encoder(torch.cat([emb_time, emb_state], axis=-1))
        return out

class SpatialEncoder(nn.Module):
    '''
        Spatial encoder that maps an sampled location to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):
        super(SpatialEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state):
        return self.encoder(state)

class ProjHead(nn.Module):
    '''
        Nonlinear projection head that maps the extracted motion features to the embedding space
    '''
    def __init__(self, feat_dim, hidden_dim, head_dim):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
            )

    def forward(self, feat):
        return self.head(feat)

def plot_scene(primary, neighbor, fname):
    '''
        Plot raw trajectories
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = plt.figure(frameon=False)
    fig.set_size_inches(16, 9)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(primary[0, 0], primary[0, 1], 'rx')
    ax.plot(primary[:, 0], primary[:, 1], 'k-')
    for i in range(neighbor.size(1)):
        ax.plot(neighbor[:, i, 0], neighbor[:, i, 1], 'b-.')
        ax.plot(neighbor[0, i, 0], neighbor[0, i, 1], 'bx')


    ax.set_aspect('equal')
    plt.grid()
    plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    plt.close(fig)