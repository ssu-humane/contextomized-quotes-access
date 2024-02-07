import torch
import numpy as np
import random
import torch.nn.functional as F
from sentence_transformers import util



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def set_seed(seed):
    n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
  
  
  
# for detection
def most_sim(outs, batch_size, body_len, criterion, do_normalize=True):
    if do_normalize:
        outs = F.normalize(outs, dim=1)
    
    title_embedding = outs[:batch_size]
    sentences_embeddings = outs[batch_size:]
    
    positive = []

    for b in range(len(body_len)):
        i = body_len[b]
        t_embedding = title_embedding[b]
        s_embedding = sentences_embeddings[:i]
        sentences_embeddings = sentences_embeddings[i:]    
        
        cos_scores_t_b = util.pytorch_cos_sim(t_embedding, s_embedding)[0]
        cos_scores_t_b = cos_scores_t_b.cpu().detach().numpy()

        top_results_t_b = cos_scores_t_b.argsort()[::-1][0]
        best_positive = s_embedding[top_results_t_b]
        
        if s_embedding.shape[0] > 1: 
          top_results_t_b_2 = cos_scores_t_b.argsort()[::-1][1]
          
          top_1 = cos_scores_t_b[top_results_t_b]
          top_2 = cos_scores_t_b[top_results_t_b_2]
          
          if top_1 - top_2 <= criterion:
            positive.append([top_results_t_b, top_results_t_b_2])
          else:
            positive.append([top_results_t_b])
        else:
          positive.append([top_results_t_b])
    
    return title_embedding, positive        
    
    
    
  
def search_pos_neg(outs, batch_size, body_len, margin, do_normalize=True):
    if do_normalize:
        outs = F.normalize(outs, dim=1)
  
    
    title_embedding = outs[:batch_size]
    sentences_embeddings = outs[batch_size:]
    
    positive, negative, neg_cnt = [], [], []
    for b in range(len(body_len)):
        i = body_len[b]
        t_embedding = title_embedding[b]
        s_embedding = sentences_embeddings[:i]
        sentences_embeddings = sentences_embeddings[i:]    
        
        cos_scores = util.pytorch_cos_sim(t_embedding, s_embedding)[0]
        cos_scores = cos_scores.cpu().detach().numpy()

        pos_idx = cos_scores.argsort()[::-1][0]
        positive_sim = sorted(cos_scores, reverse=True)[0]
        threshold_sim = positive_sim - margin
        
        neg_sim_candidate = [sim for sim in cos_scores if sim <= threshold_sim]

        modified_neg = []
        modified_neg_idx = []
        for sim in neg_sim_candidate:
          neg_idx = np.where(cos_scores == sim)[0][0]
          modified_neg_idx.append(neg_idx)
        positive.append(pos_idx)
        negative.append(modified_neg_idx)
        neg_cnt.append(len(modified_neg_idx))
    
    results = {'positive':positive, 'negative':negative, 'neg_cnt':neg_cnt}
    return results  
  
  

  
  
def make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder, pos_idx=None, neg_idx=None, neg_cnt=None):
  if args.assignment == 'dynamic':
    encoder.eval()
    with torch.no_grad():
      outs = encoder(
        input_ids = torch.cat([title_id, body_ids] ), 
        attention_mask = torch.cat([title_at, body_atts]),
      )
      
    results = search_pos_neg(outs, args.batch_size, body_len, args.margin)
    pos_idx, neg_idx, neg_cnt = results['positive'], results['negative'], results['neg_cnt']
    
    encoder.train()

  pos_body_ids, neg_body_ids = [], []
  pos_body_atts, neg_body_atts = [], []
  title_ids, title_ats = [], []
  
  for b in range(len(body_len)):
    i = body_len[b]
    b_id, b_at = body['input_ids'][b][:i].to(args.device).long(), body['attention_mask'][b][:i].to(args.device).long()
    
    for j in range(neg_cnt[b]):
      if pos_idx[b] == -1: # negative does not exist
        continue
      else:
        if neg_idx[b][j] != -1:
          pos_body_ids.append(b_id[pos_idx[b]])
          pos_body_atts.append(b_at[pos_idx[b]])
          neg_body_ids.append(b_id[neg_idx[b][j]])
          neg_body_atts.append(b_at[neg_idx[b][j]])
          title_ids.append(title_id[b])
          title_ats.append(title_at[b])
          
  title_ids = torch.stack(title_ids, dim=0)
  title_ats = torch.stack(title_ats, dim=0)
  pos_body_ids = torch.stack(pos_body_ids, dim=0)
  neg_body_ids = torch.stack(neg_body_ids, dim=0)
  pos_body_atts = torch.stack(pos_body_atts, dim=0)
  neg_body_atts = torch.stack(neg_body_atts, dim=0)
  
  return title_ids, title_ats, pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts  
  
  
  
def freeze_params(args, model):
  for i in range(args.freeze_layer):
    for name, param in model.encoder.layer[i].named_parameters():
      param.requires_grad = False
      
      
      
def show_requires_grad(model):
  for name, param in model.named_parameters():
    print(name, ": ", param.requires_grad)