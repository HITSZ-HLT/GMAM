import sys
sys.path.append('./')
import os
import argparse
from model.utils import seed_everything
# this work is inspired by BARTABSA https://github.com/yhcc/BARTABSA
parser = argparse.ArgumentParser()
# parser.add_argument('--dataset_name', default='cdcp', type=str)
parser.add_argument('--dataset_name', default='essay', type=str)
parser.add_argument('--lr', default=5e-5, type=float)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--n_epochs', default=75, type=int)
parser.add_argument('--decoder_type', default='avg_score', type=str, choices=['None', 'avg_score'])
parser.add_argument('--length_penalty', default=1.0, type=float)
parser.add_argument('--bart_name', default='facebook/bart-base', type=str)
parser.add_argument('--use_encoder_mlp', type=int, default=1)
parser.add_argument('--save_model', type=int, default=0)
parser.add_argument('--cuda_rank', type=int, default=2)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--layernorm_decay', type=float, default=0.001)

parser.add_argument('--biloss', action='store_true', default=False)
parser.add_argument('--replace_pos', action='store_true', default=True)
parser.add_argument('--decode_mask', action='store_true', default=False)
parser.add_argument('--position_type',type=int,default=0)


args= parser.parse_args()

lr = args.lr
n_epochs = args.n_epochs
batch_size = args.batch_size
num_beams = args.num_beams
dataset_name = args.dataset_name
_first = False
length_penalty = args.length_penalty
if isinstance(args.decoder_type, str) and args.decoder_type.lower() == 'none':
    args.decoder_type = None
decoder_type = args.decoder_type
bart_name = args.bart_name
use_encoder_mlp = args.use_encoder_mlp
save_model = args.save_model
biloss = args.biloss
seed = args.seed
replace_pos = args.replace_pos
decode_mask  = args.decode_mask
position_type = args.position_type
layernorm_decay = args.layernorm_decay
seed_everything(seed)


os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(args.cuda_rank)
print("cuda rank",os.environ['CUDA_VISIBLE_DEVICES'])
#######hyper
#######hyper

import warnings
warnings.filterwarnings('ignore')
from model.bart_absa import BartSeq2SeqModel
from fastNLP import Trainer
from model.losses import Seq2SeqLoss
from fastNLP import BucketSampler, GradientClipCallback, cache_results, WarmupCallback
from fastNLP import FitlogCallback
from fastNLP.core.sampler import SortedSampler
from model.generator import SequenceGeneratorModel
import fitlog

demo = False
if demo:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{_first}_demo.pt"
else:
    cache_fn = f"caches/data_{bart_name}_{dataset_name}_{_first}.pt"

max_len = {
    'essay':10, # BartAMPipe sup/atta 有component的对应词表
    'cdcp':20
}[dataset_name]

max_len_a = {
    'essay':0.5, # BartAMPipe sup/atta 有component的对应词表
    'cdcp':0.5,
}[dataset_name]

if not os.path.exists('logs'):
    os.mkdir("logs")
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
from torch import optim
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# norm for not bart layer
parameters = []
params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = [param for name, param in model.named_parameters() if not ('bart_encoder' in name or 'bart_decoder' in name)]
parameters.append(params)

params = {'lr':lr, 'weight_decay':1e-2}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

params = {'lr':lr, 'weight_decay':layernorm_decay}
params['params'] = []
for name, param in model.named_parameters():
    if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
        params['params'].append(param)
parameters.append(params)

optimizer = optim.AdamW(parameters)
# optimizer = optim.RMSprop(parameters)


callbacks = []
callbacks.append(GradientClipCallback(clip_value=5, clip_type='value'))
callbacks.append(WarmupCallback(warmup=0.01, schedule='linear'))
callbacks.append(FitlogCallback(data_bundle.get_dataset('test')))

sampler = None
sampler = BucketSampler(seq_len_field_name='src_seq_len')
metric = Seq2SeqSpanMetric(eos_token_id, num_labels=len(label_ids),label_mapping = mapping ,_first=_first,)


model_path = None
if save_model:
    model_path = 'save_models/'
seed_everything(seed)
trainer = Trainer(train_data=data_bundle.get_dataset('train'), model=model, optimizer=optimizer,
                  loss=Seq2SeqLoss(biloss=biloss),
                  batch_size=batch_size, sampler=sampler, drop_last=False, update_every=1,
                  num_workers=2, n_epochs=n_epochs, print_every=1,
                  dev_data=data_bundle.get_dataset('dev'), metrics=metric, metric_key='triple_f',
                  validate_every=-1, save_path=model_path, use_tqdm=True, device=device,
                  callbacks=callbacks, check_code_level=0, test_use_tqdm=False,
                  test_sampler=SortedSampler('src_seq_len'), dev_batch_size=64)

trainer.train(load_best_model=False)