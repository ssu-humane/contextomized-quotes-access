from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.auto import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from util import most_sim

class Contextomized_Detection_Dataset(Dataset):
    def __init__(self, args, title_texts, body_texts, label, max_seq=85):
        self.tokenizer = args.tokenizer
        self.title = []
        self.body = []
        self.label = []
        self.max_seq = max_seq
        self.max_len = args.max_len
        self.body_len = []
        assert len(title_texts) == len(body_texts) 
        
        for idx in tqdm(range(len(title_texts))):
            title = title_texts[idx]
            body = body_texts[idx]
            
            title_input = self.tokenizer(title, padding='max_length', truncation=True,
                                      max_length=self.max_len, return_tensors='pt')
            title_input['input_ids'] = torch.squeeze(title_input['input_ids'])
            title_input['attention_mask'] = torch.squeeze(title_input['attention_mask'])
            
            
            body_input = self.tokenizer(body, padding='max_length', truncation=True,
                                      max_length=self.max_len, return_tensors='pt')
            self.body_len.append(len(body_input['input_ids']))
            
            b_input, b_att, b_token = np.zeros((self.max_seq,self.max_len)), np.zeros((self.max_seq,self.max_len)), np.zeros((self.max_seq,self.max_len))
            
            b_input[:len(body_input['input_ids'])] = body_input['input_ids']
            b_att[:len(body_input['attention_mask'])] = body_input['attention_mask']
            b_token[:len(body_input['token_type_ids'])] = body_input['token_type_ids']
            
            b_input = torch.Tensor(b_input)
            b_att = torch.Tensor(b_att)
            b_token = torch.Tensor(b_token)
            
            body_input['input_ids'] = torch.squeeze(b_input)
            body_input['attention_mask'] = torch.squeeze(b_att)
            body_input['token_type_ids'] = torch.squeeze(b_token)
            
            self.title.append(title_input)
            self.body.append(body_input)
            self.label.append(label[idx])
            
    def __len__(self):
        return len(self.title)
    
    def __getitem__(self, idx):
        return self.title[idx], self.body[idx], self.body_len[idx], torch.tensor(self.label[idx], dtype=torch.long)
    
    

  
    
def create_data_loader(args, df, shuffle, drop_last):
    cd = Contextomized_Detection_Dataset(
        args,
        title_texts=df.headline_quote.to_numpy(),
        body_texts=df.body_quotes.to_numpy(),
        label = df.label.to_numpy(),
        max_seq = max(df.body_quotes.apply(len).values),
    )
  
    return DataLoader(
        cd,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )





def make_tensorloader(args, encoder, data_loader, train=False, drop_last=True):
    output = []
    labels = []
    
    encoder.eval()
    with torch.no_grad():
        for title, body, body_len, label in tqdm(data_loader):
            title_id, title_at = title['input_ids'].to(args.device).long(), title['attention_mask'].to(args.device).long()
            b_ids = []
            b_atts = []
            
            for b in range(len(body_len)):
                i = body_len[b]
                b_id, b_at = body['input_ids'][b][:i].to(args.device).long(), body['attention_mask'][b][:i].to(args.device).long()
                b_ids.append(b_id)
                b_atts.append(b_at)
            body_ids = torch.cat(b_ids, dim=0)
            body_atts = torch.cat(b_atts, dim=0)

            outs = encoder(
                        input_ids = torch.cat([title_id, body_ids]).to(args.device), 
                        attention_mask = torch.cat([title_at, body_atts]).to(args.device),
            )

            s1, s2_idx = most_sim(outs, args.batch_size, body_len, args.criterion)

            s2_input_ids = []
            s2_attention_masks = []
            
            for b in range(len(body_len)):
              i = body_len[b]
              b_id, b_at = body['input_ids'][b][:i].to(args.device).long(), body['attention_mask'][b][:i].to(args.device).long()
              
              if len(s2_idx[b]) > 1:
                s2_ids_1 = b_id[s2_idx[b][0]]
                s2_ids_2 = b_id[s2_idx[b][1]]
                
                s2_atts_1 = b_at[s2_idx[b][0]]
                s2_atts_2 = b_at[s2_idx[b][1]]
                   
                  
                s2_ids_12 = torch.cat([s2_ids_1[s2_ids_1 != 1][:-1], s2_ids_2[s2_ids_2 != 1][1:]])
                s2_ids = torch.cat([s2_ids_12, torch.tensor([1]*(args.max_len-s2_ids_12.shape[0])).cuda()])

                s2_atts_12 = torch.cat([s2_atts_1[s2_atts_1 != 0][:-1], s2_atts_2[s2_atts_2 != 0][1:]])
                s2_atts = torch.cat([s2_atts_12, torch.tensor([0]*(args.max_len-s2_atts_12.shape[0])).cuda()])
                                
              else:
                s2_ids = b_id[s2_idx[b][0]]
                s2_atts = b_at[s2_idx[b][0]]

              if s2_ids.shape[0] > args.max_len:
                s2_ids = s2_ids[:args.max_len].long()
                s2_atts = s2_atts[:args.max_len].long()
                
              s2_input_ids.append(s2_ids)
              s2_attention_masks.append(s2_atts)

            s2_input_ids = torch.stack(s2_input_ids, dim=0)
            s2_attention_masks = torch.stack(s2_attention_masks, dim=0)
                      
            s2 = encoder(s2_input_ids.to(args.device), s2_attention_masks.to(args.device))
            s2 = F.normalize(s2, dim=1)
            s = torch.cat([s1, s2, abs(s1-s2), s1*s2], dim=1)
            
            output.append(s)
            labels.append(label)
            
        output = torch.cat(output, dim=0).contiguous().squeeze()
        labels = torch.cat(labels)

    linear_ds = TensorDataset(output, labels)
    linear_loader = DataLoader(linear_ds, batch_size=args.batch_size, shuffle=train, drop_last=drop_last)
    
    return linear_loader
