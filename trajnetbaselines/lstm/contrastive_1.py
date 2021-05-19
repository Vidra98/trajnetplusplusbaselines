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
            Output:
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

        sample_pos, sample_neg = self._sampling_spatial(batch_scene, batch_split)
        
        # -----------------------------------------------------
        #              Lower-dimensional Embedding 
        # -----------------------------------------------------
        
        # Get the ID from the primary trajectory
        PrimaryID = batch_split[:-1] 
        
        # Get the number of scenes
        nb_scenes = batch_split.shape[0] - 1  
        
        # Project the feature vector onto the embedding space for the primary pedestrian
        emb_obsv = self.head_projection(batch_feat[:, PrimaryID]) 
        
        
        # Project the sampled positions at the horizon onto the embedding space 
        emb_pos=self.encoder_sample(sample_pos[-1])
        emb_neg=self.encoder_sample(sample_neg[-1])
        
        #normalize in a unit sphere
        query = F.normalize(emb_obsv,dim=-1)
        key_pos = F.normalize(emb_pos,dim=-1)
        key_neg = F.normalize(emb_neg,dim=-1)
        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------
        
        # -----------------------------------------------------
        #                       NCE Loss 
        # -----------------------------------------------------

        loss = torch.zeros(nb_scenes)
        sim_pos = (query * key_pos[None,:,:]).sum(dim=-1)
        for i in range(nb_scenes):
            # First negative key of the scene
            begin = 9*(batch_split[i]-i)
            
            # Last negative key of the scene
            end = 9*(batch_split[i+1]-(i+1))
            
            sim_neg = (query[:,i,None,:] * key_neg[None,begin:end,:]).sum(dim=-1)
            
            logits = torch.cat((sim_pos[:,i,None], sim_neg), dim=1) / self.temperature
            
            labels = torch.zeros(logits.size(0), dtype = torch.long)

            loss[i] = self.criterion(logits, labels)
        
        loss = loss.mean()
        #print("computed loss = ",loss.item())
        
        return loss

    def event(self, batch_scene, batch_split, batch_feat):
        #sample_pos shape: torch.Size([4, 8, 2])
        #sample_neg shape: torch.Size([4, 2052, 2])
        sample_pos, sample_neg = self._sampling_spatial(batch_scene, batch_split)
        
        # -----------------------------------------------------
        #              Lower-dimensional Embedding 
        # -----------------------------------------------------
        
        # Get the ID from the primary trajectory
        PrimaryID = batch_split[:-1] 
        
        # Get the number of scenes
        nb_scenes = batch_split.shape[0] - 1  
        
        # Project the feature vector onto the embedding space for the primary pedestrian
        # Shape: torch.Size([12, 8, 8])
        emb_obsv = self.head_projection(batch_feat[:, PrimaryID]) 
        
        time = (torch.arange(self.horizon)[:,None]-0.5*(self.horizon-1))/self.horizon
        time_pos = torch.ones(sample_pos.size(0), sample_pos.size(1))*time
        time_neg = torch.ones(sample_neg.size(0), sample_neg.size(1))*time
        # Project the sampled positions at the horizon onto the embedding space 
        emb_pos=self.encoder_sample(sample_pos, time_pos[:,:,None])
        emb_neg=self.encoder_sample(sample_neg, time_neg[:,:,None])
        
        
        #normalize in a unit sphere
        query = F.normalize(emb_obsv,dim=-1)
        key_pos = F.normalize(emb_pos,dim=-1)
        key_neg = F.normalize(emb_neg,dim=-1)
        # -----------------------------------------------------
        #                   Compute Similarity 
        # -----------------------------------------------------
        
        # -----------------------------------------------------
        #                       NCE Loss 
        # -----------------------------------------------------
       
        loss = torch.zeros(nb_scenes)
        sim_pos = (query[:,None,:,:] * key_pos[None,:,:,:]).sum(dim=-1)

        for i in range(nb_scenes):
            # Index of the first negative key of the scene
            begin = 9*(batch_split[i]-i)
            
            # Index of the last negative key of the scene
            end = 9*(batch_split[i+1]-(i+1))

            sim_neg = (query[:,None,i,None,:] * key_neg[None,:,begin:end,:]).sum(dim=-1)
            logits = torch.cat((sim_pos[:,:,i,None], sim_neg), dim=-1) / self.temperature
            logits = logits.view(logits.size(0)*logits.size(1), logits.size(2))
            labels = torch.zeros(logits.size(0), dtype = torch.long)

            loss[i] = self.criterion(logits, labels)
        
        loss = loss.mean()
        #print("computed loss = ",loss.item())
        
        return loss
        
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
        
        # Get the ID of the primary pedestrian in each scene
        primaryID = batch_split[:-1]
        
        primary_current = batch_scene[self.obs_length-1, primaryID]
        horizon = batch_scene[self.obs_length:self.obs_length+self.horizon]
        
        # #####################################################
        #           TODO: fill the following code
        # #####################################################
        
        # ----------------------------------------------------
        #                  Positive Samples
        # -----------------------------------------------------
        
        # Sample the relative future position with respect to the current position for the primary pedestrian in each scene 
        sample_pos = horizon[:,primaryID] - primary_current
        
        # Add noise
        sample_pos += torch.rand(sample_pos.size()).sub(0.5) * self.noise_local

        # -----------------------------------------------------
        #                  Negative Samples
        # -----------------------------------------------------    
        sample_neg = torch.empty(self.horizon, 0 ,2)
        for i in range(batch_split.shape[0]-1):
            sample_neg = torch.cat((sample_neg, horizon[:, batch_split[i]+1:batch_split[i+1]]-primary_current[i]), 1)
        sample_neg = torch.where(torch.isnan(sample_neg), torch.full_like(sample_neg, 0.0), sample_neg) 
        
        sample_neg += torch.rand(sample_neg.size()).sub(0.5) * self.noise_local
        
        #create discomfort zone
        sample_neg = sample_neg[:, :, None, :] + self.agent_zone[None, None, :, :]
        sample_neg = sample_neg.view(sample_neg.size(0), sample_neg.size(1) * sample_neg.size(2), sample_neg.size(3))

        
        # -----------------------------------------------------
        #       Remove negatives that are too hard (optional)
        # -----------------------------------------------------

        # -----------------------------------------------------
        #       Remove negatives that are too easy (optional)
        # -----------------------------------------------------
        return sample_pos,sample_neg 

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
