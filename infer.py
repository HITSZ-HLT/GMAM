import sys
sys.path.append('./')
import os
import argparse
from model.utils import seed_everything

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='essay', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--opinion_first', action='store_true', default=False)
parser.add_argument('--n_epochs', default=75, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--save_model', type=int, default=1)
parser.add_argument('--encoder_scale', type=float, default=1)
parser.add_argument('--cuda_rank', type=int, default=0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--token_cls', action='store_true', default=False)
parser.add_argument('--token_cls_norm', action='store_true', default=False)
parser.add_argument('--biloss', action='store_true', default=True)
parser.add_argument('--replace_pos', action='store_true', default=True)
parser.add_argument('--decode_mask', action='store_true', default=True)
parser.add_argument('--position_type',type=int,default=0)


args= parser.parse_args()

lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
dataset_name = args.dataset_name
opinion_first = args.opinion_first
length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
use_encoder_mlp = args.use_encoder_mlp
save_model = args.save_model
token_cls = args.token_cls
biloss = args.biloss
seed = args.seed
token_cls_norm = args.token_cls_norm
encoder_scale = args.encoder_scale
replace_pos = args.replace_pos
decode_mask  = args.decode_mask
position_type = args.position_type
_first = False

####### model in the save_models dir
model_to_load = "save_models/best_SequenceGeneratorModel_triple_f_2023-04-16-23-07-25-556294"
####### model in the save_models dir

seed_everything(seed)

if 'p' in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['p']
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.cuda_rank)
print("cuda rank",os.environ['CUDA_VISIBLE_DEVICES'])
#######hyper
#######hyper

import warnings
warnings.filterwarnings('ignore')
from model.bart_am import BartSeq2SeqModel
from fastNLP import cache_results
from fastNLP.core.sampler import SortedSampler
from model.generator import SequenceGeneratorModel
import fitlog

demo = False
if demo:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}_demo.pt"
else:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{opinion_first}.pt"

max_len = {
    'essay':10, # BartAMPipe sup/atta 有component的对应词表
    'cdcp':20
}[dataset_name]

max_len_a = {
    'essay':0.5, # BartAMPipe sup/atta 有component的对应词表
    'cdcp':0.5,
}[dataset_name]


fitlog.set_log_dir('logs/{}'.format(dataset_name.split('/')[-1]))
fitlog.add_hyper(args)


if 'essay' in dataset_name:
    from data.pipe import BartAMPipe_essay as BartPipe
    from model.metrics import Seq2SeqSpanMetric_essay as Seq2SeqSpanMetric
elif 'cdcp' in dataset_name:
    from data.pipe import BartAMPipe_cdcp as BartPipe
    from model.metrics import Seq2SeqSpanMetric_cdcp as Seq2SeqSpanMetric


    
@cache_results(cache_fn, _refresh=True)
def get_data():
    pipe = BartPipe(tokenizer=bart_name, _first=_first)
    data_bundle = pipe.process_from_file(f'./data/{dataset_name}', demo=demo)
    return data_bundle, pipe.tokenizer, pipe.mapping2id, pipe.mapping2targetid , pipe.relation_ids, pipe.component_ids, pipe.none_ids

data_bundle, tokenizer, mapping2id, mapping2targetid,relation_ids,component_ids,none_ids = get_data()

print("The number of tokens in tokenizer ", len(tokenizer.decoder))

bos_token_id = 0  #
eos_token_id = 1  #
label_ids = list(mapping2id.values())
model = BartSeq2SeqModel.build_model(bart_name, tokenizer, label_ids=label_ids, decoder_type=decoder_type,
                                     copy_gate=False, use_encoder_mlp=use_encoder_mlp, use_recur_pos=False,
                                     replace_pos = replace_pos,position_type=position_type)
vocab_size = len(tokenizer)
print(vocab_size, model.decoder.decoder.embed_tokens.weight.data.size(0))
model = SequenceGeneratorModel(model, bos_token_id=bos_token_id,
                               eos_token_id=eos_token_id,
                               max_length=max_len, max_len_a=max_len_a,num_beams=num_beams, do_sample=False,
                               repetition_penalty=1, length_penalty=length_penalty, pad_token_id=eos_token_id,
                               restricter=None,decode_mask=decode_mask,relation_ids=relation_ids,component_ids=component_ids,none_ids=none_ids)

tag_labels = model.seq2seq_model.decoder.mapping.tolist()
tag_tokens = tokenizer.convert_ids_to_tokens(tag_labels)

mapping = dict(zip(tag_tokens,range(len(tag_tokens))))
print(mapping)

import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

from fastNLP import DataSetIter
metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids),label_mapping = mapping ,_first=_first,)
model=(torch.load(model_to_load))
test_sampler=SortedSampler('src_seq_len')

data_iterator = DataSetIter(dataset=data_bundle.get_dataset('test'), batch_size=64, sampler=test_sampler)

model.to(device)
model.eval()

input_batch = []
output_batch = []

from tqdm import tqdm
with torch.no_grad():
    for batch_x, batch_y in tqdm(data_iterator):
        input_batch.append({'batch_x':batch_x,'batch_y':batch_y})
        
        res = model.predict(batch_x['src_tokens'].cuda(),batch_x['src_seq_len'].cuda())
        true_tag = batch_y['tgt_tokens']
        for k in res:
            if res[k] is not None:
                res[k]=res[k].cpu()
        output_batch.append({'batch_res':res})
        pred_tag = res['pred'].cpu()
        span = metric.evaluate(target_span = batch_y['target_span'],pred = res['pred'].cpu(),tgt_tokens = batch_y['tgt_tokens'])
print(metric.get_metric())

import pickle

# dump
to_dump = './test_batch.pkl'
pickle.dump({'input_batch':input_batch,'output_batch':output_batch},open(to_dump,'wb'))

# load the pkl and show inference result

test_res = pickle.load(open("./test_batch.pkl",'rb'))

# show result in the first batch
input_batch = test_res['input_batch'][0]
output_batch = test_res['output_batch'][0]

for sample_id,(src_token,pred,target) in enumerate(zip(input_batch['batch_x']['src_tokens'],output_batch['batch_res']['pred'],input_batch['batch_y']['tgt_tokens'])):
    print("ID{}".format(sample_id))
    tokens = tokenizer.convert_ids_to_tokens(src_token)
    ps,_ = metric.build_pair(pred.tolist())
    ts,_ = metric.build_pair(target.tolist())
    # preds 
    print("----------prediction results----------")
    for tup in ps:
        # sent1 target
        # sent2 src
        sent1 = tokenizer.convert_tokens_to_string(tokens[tup[0]-len(metric.id2label):tup[1]-len(metric.id2label)+1])
        lab1 = metric.id2label[tup[2]]
        sent2 = tokenizer.convert_tokens_to_string(tokens[tup[3]-len(metric.id2label):tup[4]-len(metric.id2label)+1])
        lab2 = metric.id2label[tup[5]]
        rel = metric.id2label[tup[6]]
        print('target:',sent1)
        print('source:',sent2)
        print(lab1,lab2,rel)
    
    # targets
    print("----------target results----------")
    for tup in ts:
        sent1 = tokenizer.convert_tokens_to_string(tokens[tup[0]-len(metric.id2label):tup[1]-len(metric.id2label)+1])
        lab1 = metric.id2label[tup[2]]
        sent2 = tokenizer.convert_tokens_to_string(tokens[tup[3]-len(metric.id2label):tup[4]-len(metric.id2label)+1])
        lab2 = metric.id2label[tup[5]]
        rel = metric.id2label[tup[6]]
        print('target:',sent1)
        print('source:',sent2)
        print(lab1,lab2,rel)
        