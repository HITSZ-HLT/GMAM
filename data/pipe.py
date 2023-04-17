from fastNLP.io import Pipe, DataBundle, Loader
import os
import json
from fastNLP import DataSet, Instance
from transformers import AutoTokenizer
import numpy as np
from itertools import chain
from functools import cmp_to_key
from itertools import chain

# aspect with relation  component_r
# opinion without relation component

def cmp_src(v1, v2):
    if v1[0]['from']==v2[0]['from']:
        return v1[1]['from'] - v2[1]['from']
    return v1[0]['from'] - v2[0]['from']

def cmp_tg(v1, v2):
    if v1[1]['from']==v2[1]['from']:
        return v1[0]['from'] - v2[0]['from']
    return v1[1]['from'] - v2[1]['from']

def CPM_prepare(src_len:int,target:list,component_ids,relation_ids,none_ids,shift=0):
        def pointer_tag(last,t,idx,arr,component_ids,relation_ids):
            if t == 0: # start # c1 [0, 1]
                arr[:shift]=0
            elif idx % 7 == 0: # c1 [0,1, 23]
                arr[:t]=0
            elif idx % 7 == 1: # tc1 [0,1,23, tc] component标签设为1
                arr = np.zeros_like(arr,dtype=int)
                for i in component_ids:
                    arr[i] = 1
            elif idx % 7 == 2: # c2 [0,1,23,tc, 45]
                arr[:shift]=0
                arr[last[-3]:last[-2]]=0
            elif idx % 7 == 3: # c2 [0,1,23,tc,45, 67]
                arr[:t] = 0
                if t < last[-4]:
                    arr[last[-4]:]=0
                else:
                    arr[last[-4]:last[-3]]=0
            elif idx % 7 == 4: # tc2 [0,1,23,tc,45,67, tc]
                arr = np.zeros_like(arr,dtype=int)
                for i in component_ids:
                    arr[i] = 1
            elif idx % 7 == 5: # r [0,1,23,tc,45,67,tc, r]
                arr = np.zeros_like(arr,dtype=int)
                for i in relation_ids:
                    arr[i] = 1
            elif idx % 7 == 6: # next
                arr[:shift]=0
            return arr
        # pad for 0
        likely_hood = np.ones(src_len+shift,dtype=int)
        likely_hood[:shift]=0
        CMP_tag = [likely_hood]
        for idx,t in enumerate(target[:-1]):
            last7 = target[idx-7 if idx-7>0 else 0:idx+1]
            likely_hood = np.ones(src_len+shift,dtype=int)
            tag = pointer_tag(last7,t,idx,likely_hood,component_ids,relation_ids)
            tag[none_ids] = 1
            CMP_tag.append(tag)
        last_end = np.zeros(src_len+shift,dtype=int)
        last_end[none_ids] = 1
        last_end[target[-1]] = 1
        CMP_tag[-1]=last_end
        CMP_tag = [i.tolist() for i in CMP_tag]
        return CMP_tag

class AMLoader(Loader):
    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo

    def _load(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ds = DataSet()
        for ins in data:
            tokens = ins['words']
            cp_src_r = ins['component_src_r']
            cp_tg = ins['component_tg']
            source=None
            if 'source' in ins.keys():
                source = ins['source']
            assert len(cp_src_r)==len(cp_tg)
            ins = Instance(raw_words=tokens, cp_source=cp_src_r, cp_target=cp_tg,source=source)
            ds.append(ins)
            if self.demo and len(ds)>30:
                break
        return ds

# _2idx_2tag_2idx_3tag
class BartAMPipe_essay(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', _first=False):
        super(BartAMPipe_essay, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mapping_relation = {  # so that the label word can be initialized in a better embedding.
            'supports': '<<positive>>',
            'attacks': '<<negative>>',
            'none':'<<none>>',
            'MajorClaim':'<<MC>>',
            'Claim':"<<C>>",
            "Premise":"<<P>>",
        }
        self.component_keys = ['MajorClaim','Claim','Premise']
        self.relation_keys = ['supports','attacks']
        self.none_key = 'none'
        self.component = ['MajorClaim','Claim','Premise']
        
        self.mapping = self.mapping_relation
        self._first = _first  # 是否先生成opinion

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)
        print()
        self.component_ids = [self.mapping2targetid[i] + 2 for i in  self.component_keys]
        self.relation_ids = [self.mapping2targetid[i] + 2 for i in  self.relation_keys]
        self.none_ids = self.mapping2targetid[self.none_key] + 2

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        aspects: [{
            'index': int
            'from': int
            'to': int
            'polarity': str
            'term': List[str]
        }],
        opinions: [{
            'index': int
            'from': int
            'to': int
            'term': List[str]
        }]

        输出为[o_s, o_e, a_s, a_e, c]或者[a_s, a_e, o_s, o_e, c]
        :param data_bundle:
        :return:
        """
        target_shift = len(self.mapping) + 2  # 是由于第一位是sos，紧接着是eos, 然后是
        component_ids = self.component_ids
        relation_ids = self.relation_ids
        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            word_tokens = ['<pad>']
            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                word_tokens.extend(bpes)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.tokenizer.eos_token_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            target = [0]  # 特殊的开始
            target_spans = []
            _word_bpes = list(chain(*word_bpes))
            
            src_len = len(_word_bpes)
            
            
            sources_targets = [(s, t) for s, t in zip(ins['cp_source'], ins['cp_target'])]
            if self._first:
                sources_targets = sorted(sources_targets, key=cmp_to_key(cmp_tg))
            else:
                sources_targets = sorted(sources_targets, key=cmp_to_key(cmp_src))

            for sources, targets in sources_targets:  # 预测bpe的start
                assert sources['index'] == targets['index']
                s_start_bpe = cum_lens[sources['from']]  # 因为有一个sos shift
                s_end_bpe = cum_lens[sources['to']-1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
                t_start_bpe = cum_lens[targets['from']]  # 因为有一个sos shift
                t_end_bpe = cum_lens[targets['to']-1]  # 因为有一个sos shift
                # 这里需要evaluate是否是对齐的
                for idx, word in zip((t_start_bpe, t_end_bpe, s_start_bpe, s_end_bpe),
                                     (targets['term'][0], targets['term'][-1], sources['term'][0], sources['term'][-1])):
                    assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                           _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]

                if self._first:
                    target_spans.append([t_start_bpe+target_shift, t_end_bpe+target_shift,self.mapping2targetid[targets['component']]+2,
                                         s_start_bpe+target_shift, s_end_bpe+target_shift,self.mapping2targetid[sources['component']]+2])
                else:
                    target_spans.append([t_start_bpe+target_shift, t_end_bpe+target_shift,self.mapping2targetid[targets['component']]+2,
                                         s_start_bpe+target_shift, s_end_bpe+target_shift,self.mapping2targetid[sources['component']]+2])
                if sources['polarity'] == 'loop':
                    target_spans[-1] = target_spans[-1][:3]
                    target_spans[-1].extend([self.mapping2targetid['none']+2]*4)
                else:
                    target_spans[-1].append(self.mapping2targetid[sources['polarity']]+2)   # 前面有sos和eos
                
                target_spans[-1] = tuple(target_spans[-1])
                

            target.extend(list(chain(*target_spans)))
            target.append(1)  # append 1是由于特殊的eos

            
            
            src_len = len(_word_bpes)
            CPM_tag = CPM_prepare(src_len=src_len, target=target[1:], shift=target_shift, component_ids=component_ids, relation_ids=relation_ids,none_ids = self.mapping2targetid[self.none_key]+2)
            assert CPM_tag is not None
            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': list(chain(*word_bpes)),'CPM_tag':CPM_tag}

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')
        
        # data_bundle = self.train_permu(data_bundle)

        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)
        data_bundle.set_pad_val('CPM_tag', -1)
        

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len','word_tokens','CPM_tag')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'src_seq_len','target_span')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = AMLoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle

# _2idx_2tag_2idx_3tag
class BartAMPipe_cdcp(Pipe):
    def __init__(self, tokenizer='facebook/bart-base', _first=False):
        super(BartAMPipe_cdcp, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.mapping_relation = {  # so that the label word can be initialized in a better embedding.
            'evidence': '<<positive>>',
            'reason': '<<negative>>',
            'none':'<<none>>',
            'value':'<<v>>',
            'policy':"<<p>>",
            "testimony":"<<t>>",
            "fact":"<<f>>",
            "reference":"<<r>>",
        }
        self.component_keys = ['value','policy','testimony','fact','reference']
        self.relation_keys = ['evidence','reason']
        self.none_key = 'none'
        self.component = ['entity']
        self.component = ['value','policy','testimony','fact','reference']
        
        self.mapping = self.mapping_relation
        self._first = _first  # 是否先生成opinion

        cur_num_tokens = self.tokenizer.vocab_size
        self.cur_num_token = cur_num_tokens

        tokens_to_add = sorted(list(self.mapping.values()), key=lambda x:len(x), reverse=True)
        unique_no_split_tokens = self.tokenizer.unique_no_split_tokens
        sorted_add_tokens = sorted(list(tokens_to_add), key=lambda x:len(x), reverse=True)
        for tok in sorted_add_tokens:
            assert self.tokenizer.convert_tokens_to_ids([tok])[0]==self.tokenizer.unk_token_id
        self.tokenizer.unique_no_split_tokens = unique_no_split_tokens + sorted_add_tokens
        self.tokenizer.add_tokens(sorted_add_tokens)
        self.mapping2id = {}
        self.mapping2targetid = {}

        for key, value in self.mapping.items():
            key_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(value))
            assert len(key_id) == 1, value
            assert key_id[0] >= cur_num_tokens
            self.mapping2id[key] = key_id[0]
            self.mapping2targetid[key] = len(self.mapping2targetid)
        print()
        self.component_ids = [self.mapping2targetid[i] + 2 for i in  self.component_keys]
        self.relation_ids = [self.mapping2targetid[i] + 2 for i in  self.relation_keys]
        self.none_ids = self.mapping2targetid[self.none_key]+2

    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        words: List[str]
        aspects: [{
            'index': int
            'from': int
            'to': int
            'polarity': str
            'term': List[str]
        }],
        opinions: [{
            'index': int
            'from': int
            'to': int
            'term': List[str]
        }]

        输出为[o_s, o_e, a_s, a_e, c]或者[a_s, a_e, o_s, o_e, c]
        :param data_bundle:
        :return:
        """
        target_shift = len(self.mapping) + 2  # 是由于第一位是sos，紧接着是eos, 然后是
        component_ids = self.component_ids
        relation_ids = self.relation_ids
        def prepare_target(ins):
            raw_words = ins['raw_words']
            word_bpes = [[self.tokenizer.bos_token_id]]
            word_tokens = ['<pad>']
            for word in raw_words:
                bpes = self.tokenizer.tokenize(word, add_prefix_space=True)
                word_tokens.extend(bpes)
                bpes = self.tokenizer.convert_tokens_to_ids(bpes)
                word_bpes.append(bpes)
            word_bpes.append([self.tokenizer.eos_token_id])

            lens = list(map(len, word_bpes))
            cum_lens = np.cumsum(list(lens)).tolist()
            target = [0]  # 特殊的开始
            target_spans = []
            _word_bpes = list(chain(*word_bpes))
            
            src_len = len(_word_bpes)
            
            
            sources_targets = [(s, t) for s, t in zip(ins['cp_source'], ins['cp_target'])]
            if self._first:
                sources_targets = sorted(sources_targets, key=cmp_to_key(cmp_tg))
            else:
                sources_targets = sorted(sources_targets, key=cmp_to_key(cmp_src))

            for sources, targets in sources_targets:  # 预测bpe的start
                assert sources['index'] == targets['index']
                s_start_bpe = cum_lens[sources['from']]  # 因为有一个sos shift
                s_end_bpe = cum_lens[sources['to']-1]  # 这里由于之前是开区间，刚好取到最后一个word的开头
                t_start_bpe = cum_lens[targets['from']]  # 因为有一个sos shift
                t_end_bpe = cum_lens[targets['to']-1]  # 因为有一个sos shift
                # 这里需要evaluate是否是对齐的
                for idx, word in zip((t_start_bpe, t_end_bpe, s_start_bpe, s_end_bpe),
                                     (targets['term'][0], targets['term'][-1], sources['term'][0], sources['term'][-1])):
                    assert _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[:1])[0] or \
                           _word_bpes[idx] == self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word, add_prefix_space=True)[-1:])[0]

                if self._first:
                    target_spans.append([t_start_bpe+target_shift, t_end_bpe+target_shift,self.mapping2targetid[targets['component']]+2,
                                         s_start_bpe+target_shift, s_end_bpe+target_shift,self.mapping2targetid[sources['component']]+2])
                else:
                    target_spans.append([t_start_bpe+target_shift, t_end_bpe+target_shift,self.mapping2targetid[targets['component']]+2,
                                         s_start_bpe+target_shift, s_end_bpe+target_shift,self.mapping2targetid[sources['component']]+2])
                if sources['polarity'] == 'loop':
                    target_spans[-1] = target_spans[-1][:3]
                    target_spans[-1].extend([self.mapping2targetid['none']+2]*4)
                else:
                    target_spans[-1].append(self.mapping2targetid[sources['polarity']]+2)   # 前面有sos和eos
                
                target_spans[-1] = tuple(target_spans[-1])
                
            target.extend(list(chain(*target_spans)))
            target.append(1)  # append 1是由于特殊的eos

            
            
            src_len = len(_word_bpes)
            CPM_tag = CPM_prepare(src_len=src_len, target=target[1:], shift=target_shift, component_ids=component_ids, relation_ids=relation_ids,none_ids = self.mapping2targetid[self.none_key]+2)
            assert CPM_tag is not None
            return {'tgt_tokens': target, 'target_span': target_spans, 'src_tokens': list(chain(*word_bpes)),
                    'CPM_tag':CPM_tag
                    }

        data_bundle.apply_more(prepare_target, use_tqdm=True, tqdm_desc='Pre. tgt.')
        
        data_bundle.set_ignore_type('target_span')
        data_bundle.set_pad_val('tgt_tokens', 1)  # 设置为eos所在的id
        data_bundle.set_pad_val('src_tokens', self.tokenizer.pad_token_id)
        data_bundle.set_pad_val('CPM_tag', -1)
        

        data_bundle.apply_field(lambda x: len(x), field_name='src_tokens', new_field_name='src_seq_len')
        data_bundle.apply_field(lambda x: len(x), field_name='tgt_tokens', new_field_name='tgt_seq_len')
        data_bundle.set_input('tgt_tokens', 'src_tokens', 'src_seq_len', 'tgt_seq_len','word_tokens','CPM_tag')
        data_bundle.set_target('tgt_tokens', 'tgt_seq_len', 'src_seq_len','target_span')

        return data_bundle

    def process_from_file(self, paths, demo=False) -> DataBundle:
        """

        :param paths: 支持路径类型参见 :class:`fastNLP.io.loader.ConllLoader` 的load函数。
        :return: DataBundle
        """
        # 读取数据
        data_bundle = AMLoader(demo=demo).load(paths)
        data_bundle = self.process(data_bundle)

        return data_bundle



if __name__ == '__main__':
    data_bundle = BartAMPipe_essay().process_from_file('/data/heyuhang/workspace/AMBart_bak/data/he/cdcp')
    print(data_bundle)

