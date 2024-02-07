from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import torch.nn as nn
import torch
import argparse
import pandas as pd
import os

from datasets import create_data_loader
from loss_func import QuoteCSELoss
from models import Encoder
from util import make_pair, AverageMeter, set_seed, freeze_params



def train(args, tbar1, tbar2, encoder, optimizer, loss_func, losses, epoch, loss_data, best_valid_loss, best_state_dict):
  for title, body, body_len, pos_idx, neg_idx, neg_cnt in tbar1: 
    if args.d_iter >= args.static_iterations:
      args.assignment = 'dynamic'

    if args.d_iter >= args.static_iterations + args.dynamic_iterations:
      args.stop = True
      
      # valid 
      encoder, loss_data, best_valid_loss, best_state_dict = valid(args, tbar2, encoder, loss_func, epoch, loss_data, best_valid_loss, best_state_dict)
      break

    args.d_iter += args.batch_size

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

    if args.assignment == 'static':
      title_ids, title_ats, pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts = make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder, pos_idx, neg_idx, neg_cnt)

    elif args.assignment == 'dynamic':
      title_ids, title_ats, pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts = make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder)

    del body_ids, body_atts, body_len, title_id, title_at

    outputs = encoder(
      input_ids = torch.cat([title_ids, pos_body_ids, neg_body_ids]),
      attention_mask = torch.cat([title_ats, pos_body_atts, neg_body_atts]),
    )

    loss = loss_func(outputs)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), args.batch_size)
    tbar1.set_description("loss: {0:.6f}".format(losses.avg), refresh=True)

    del pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts, outputs, loss, title_ids, title_ats

    if args.d_iter%50048 == 0: 
      loss_data.append([epoch, losses.avg, 'Train', args.d_iter, 'by_iteration'])
      
      # valid
      encoder, loss_data, best_valid_loss, best_state_dict = valid(args, tbar2, encoder, loss_func, epoch, loss_data, best_valid_loss, best_state_dict)

  loss_data.append([epoch, losses.avg, 'Train', args.d_iter, 'by_epoch'])
  
  return encoder, optimizer, loss_data, best_valid_loss, best_state_dict






def valid(args, tbar2, encoder, loss_func, epoch, loss_data, best_valid_loss, best_state_dict):
  valid_loss = []

  with torch.no_grad():
    for title, body, body_len, pos_idx, neg_idx, neg_cnt in tbar2:
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

      if args.assignment == 'static':
        title_ids, title_ats, pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts = make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder, pos_idx, neg_idx, neg_cnt)

      elif args.assignment == 'dynamic':
        title_ids, title_ats, pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts = make_pair(args, body, title_id, title_at, body_ids, body_atts, body_len, encoder)

      del body_ids, body_atts, body_len, title_id, title_at

      outputs = encoder(
        input_ids = torch.cat([title_ids, pos_body_ids, neg_body_ids]),
        attention_mask = torch.cat([title_ats, pos_body_atts, neg_body_atts]),
      )

      loss = loss_func(outputs)
      valid_loss.append(loss.item())

      del pos_body_ids, neg_body_ids, pos_body_atts, neg_body_atts, outputs, loss, title_ids, title_ats

    avg_valid_loss = sum(valid_loss) / len(valid_loss)
    loss_data.append([epoch, avg_valid_loss, 'Valid', args.d_iter, 'by_iteration'])
    
    if best_valid_loss == None or avg_valid_loss < best_valid_loss:
      best_state_dict = encoder.state_dict()
      best_valid_loss = avg_valid_loss

    print(str(epoch), 'th epoch, Avg Valid Loss: ', str(avg_valid_loss), 'd_iter:', args.d_iter, 'assignment:', args.assignment)

    return encoder, loss_data, best_valid_loss, best_state_dict











def main():
    parser = argparse.ArgumentParser()
    
    # arguments
    parser.add_argument("--seed", default=123, type=int, help="set seed") 
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--max_len", default=100, type=int, help="max length")     
    parser.add_argument("--num_workers", default=16, type=int, help="number of workers")    
    parser.add_argument("--dimension_size", default=768, type=int, help="dimension size") 
    parser.add_argument("--hidden_size", default=100, type=int, help="hidden size")     
    parser.add_argument("--learning_rate", default=1e-6, type=float, help="learning rate") 
    parser.add_argument("--weight_decay", default=1e-7, type=float, help="weight decay")   
    parser.add_argument("--epochs", default=500, type=int, help="epoch")    
    parser.add_argument("--margin", default=0.04, type=float, help="margin")   
    
    parser.add_argument("--freeze", default=True, type=bool, help="whether to freeze")   
    parser.add_argument("--freeze_layer", default=9, type=int, help="a number of layer to freeze")
    parser.add_argument("--mode", default="train", type=str, help="mode")
    
    parser.add_argument("--iteration", default=1043200, type=int, help="total iteration to train")   
    parser.add_argument("--static_iterations", default=500000, type=int, help="epoch for static")   
    parser.add_argument("--dynamic_iterations", default=543200, type=int, help="epoch for dynamic")
    parser.add_argument("--d_iter", default=0, type=int, help="training iteration")   
    parser.add_argument("--stop", default=False, type=bool, help="whether stop or not") 
    parser.add_argument("--temperature", default=0.05, type=float, help="temperature")   
    parser.add_argument("--assignment", default='static', type=str, help="assignment type")   
    
    parser.add_argument("--MODEL_DIR", default='./model/', type=str, help="where to save the trained model") 
    parser.add_argument("--MODIFIED_DATA_PATH", default='./data/modified.pkl', type=str, help="data for pretraining")    
    parser.add_argument("--VERBATIM_DATA_PATH", default='./data/verbatim.pkl', type=str, help="data for pretraining")        
    
    args = parser.parse_args()
    
    if not os.path.exists(args.MODEL_DIR):
        os.makedirs(args.MODEL_DIR)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    set_seed(args.seed)

    MODEL_PATH = 'klue/roberta-base'
    args.backbone_model = AutoModel.from_pretrained(MODEL_PATH)
    args.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = args.batch_size * torch.cuda.device_count()
    
    if args.freeze:
      freeze_params(args, args.backbone_model) # layer freeze   
    
    encoder = Encoder(args)    
    encoder = nn.DataParallel(encoder)
    encoder = encoder.to(args.device)
      
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()
    
    loss_func = QuoteCSELoss(temperature=args.temperature, batch_size=args.batch_size)
    
    
    print('Making Dataloader')
    modified_df = pd.read_pickle(args.MODIFIED_DATA_PATH)
    verbatim_df = pd.read_pickle(args.VERBATIM_DATA_PATH)
    
    train_modified_df, test_modified_df = train_test_split(modified_df, test_size=0.2, random_state=args.seed)
    valid_modified_df, test_modified_df = train_test_split(test_modified_df, test_size=0.5, random_state=args.seed)

    train_verbatim_df, test_verbatim_df = train_test_split(verbatim_df, test_size=0.2, random_state=args.seed)
    valid_verbatim_df, test_verbatim_df = train_test_split(test_verbatim_df, test_size=0.5, random_state=args.seed)
    
    train_df = pd.concat([train_modified_df, train_verbatim_df])
    valid_df = pd.concat([valid_modified_df, valid_verbatim_df])
    test_df = pd.concat([test_modified_df, test_verbatim_df])
    
    train_data_loader = create_data_loader(args,
                                           df = train_df, 
                                           shuffle = True,
                                           drop_last = True)
    valid_data_loader = create_data_loader(args,
                                           df = valid_df, 
                                           shuffle = False,
                                           drop_last = True)

    
    # train    
    print('Start Training')

    loss_data = []
    best_valid_loss = None
    best_state_dict = None
    
    for epoch in range(args.epochs):
        if args.stop == True:
          break

        losses = AverageMeter()
        
        tbar1 = tqdm(train_data_loader)
        tbar2 = tqdm(valid_data_loader)

        encoder.train()
        encoder, optimizer, loss_data, best_valid_loss, best_state_dict = train(args, tbar1, tbar2, encoder, optimizer, loss_func, losses, epoch, loss_data, best_valid_loss, best_state_dict)
        
          
    torch.save(best_state_dict, args.MODEL_DIR + 'best_checkpoint.pt')

    # save loss
    df_loss = pd.DataFrame(loss_data, columns=('Epoch', 'Loss', 'Type', 'Iteration', 'Save_Criterion'))
    df_loss.to_csv(args.MODEL_DIR + 'loss.csv', sep=',', index=False)

    
if __name__ == "__main__":
    main()
