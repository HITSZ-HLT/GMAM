from fastNLP import MetricBase
from fastNLP.core.metrics import _compute_f_pre_rec
from collections import Counter

import numpy as np


class Seq2SeqSpanMetric_cdcp(MetricBase):
    def __init__(self, eos_token_id, num_labels,label_mapping=None, _first=True):
        super(Seq2SeqSpanMetric_cdcp, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.label_mapping = label_mapping
            
        self.id2label = dict(zip(label_mapping.values(),label_mapping.keys()))
        self.none = label_mapping["<<none>>"]
        self.component = ["<<v>>","<<t>>","<<p>>","<<f>>","<<r>>"]
        self.relation = list(set(label_mapping.keys())-set(self.component)-set(["<<none>>","<s>","</s>"]))
        self.component_idx = list(map(lambda x: self.label_mapping[x],self.component))
        self.relation_idx = list(map(lambda x: self.label_mapping[x],self.relation))
        
        self.word_start_index = num_labels + 2  # +2, shift for sos and eos
        self.add_token = list(self.id2label.keys())
        
        self.component_metric = ComponentScore(self.id2label)
        self.ae_oe_fp = 0
        self.ae_oe_tp = 0
        self.ae_oe_fn = 0
        self.triple_fp = 0
        self.triple_tp = 0
        self.triple_fn = 0
        self.em = 0
        self.invalid = 0
        self.total =  1e-13
        self.ae_sc_fp = 0
        self.ae_sc_tp = 0
        self.ae_sc_fn = 0
        
        self.am_component_fp = 0
        self.am_component_tp = 0
        self.am_component_fn = 0
        
        self.component_fp = 0
        self.component_tp = 0
        self.component_fn = 0
        
        self.invalid_len = 0
        self.invalid_order = 0
        self.invalid_cross = 0
        self.invalid_cover  = 0
        
        assert _first is False, "Current metric only supports aspect first"

        self._first = _first


    def build_pair(self,tag_seq,if_skip_cross=True):
        invalid_len = 0
        invalid_order = 0
        invalid_cross = 0
        invalid_cover = 0
        skip = False
        pairs = []
        cur_pair = []
        add_token = self.add_token
        if len(tag_seq):
            for i in tag_seq:
                if i  in self.relation_idx or(i == self.none and len(cur_pair)==6):
                    cur_pair.append(i)
                    if len(cur_pair)!=7:
                        skip = True
                        invalid_len=1
                    elif self.none in cur_pair:
                        if not (cur_pair[2] in add_token and cur_pair[5] in add_token and cur_pair[6] in add_token):
                            skip=True
                        else:
                            skip = False
                    else: # 解码长度正确
                        # 检查位置是否正确 <s1,e1,t1,s2,e2,t2,t3>
                        if cur_pair[0]>cur_pair[1] or cur_pair[3]>cur_pair[4]:
                            skip = True
                            invalid_order=1                        
                        elif  not (cur_pair[1]<cur_pair[3] or cur_pair[0]>cur_pair[4]) :
                            skip = True
                            invalid_cover=1
                        if cur_pair[2] in self.relation_idx or cur_pair[5] in self.relation_idx or cur_pair[6] in self.component_idx:
                            if if_skip_cross: # 可以考虑做多一层限制，防止relation 和 component标签错位
                                skip = True
                            invalid_cross=1
                        tag = set([cur_pair[2],cur_pair[5],cur_pair[6]])
                        RC_idx = self.relation_idx+self.component_idx
                        if not (cur_pair[2] in RC_idx and cur_pair[5] in RC_idx and cur_pair[6] in RC_idx):
                            skip=True
                            invalid_cross=1
                    if skip:
                        skip=False
                    else:
                        pairs.append(tuple(cur_pair))
                    cur_pair = []
                else:
                    cur_pair.append(i)
        # pairs = list(set(pairs))
        return pairs,(invalid_len,invalid_order,invalid_cross,invalid_cover)

    def evaluate(self, target_span, pred, tgt_tokens,):
        self.total += pred.size(0)

        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # delete </s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            em = 0
            ps = ps[:pred_seq_len[i]]
            # self.total += len(ps) // 7
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em

            
            # 计算多种情况的invalid
            # invalid, pairs  = self.compute_invalid_v2(ps)
            
            # 分别计算三种情况的invalid
            pairs,invalid = self.build_pair(ps)
            self.invalid += sum(invalid)
            self.invalid_len += invalid[0]
            self.invalid_order += invalid[1]
            self.invalid_cross += invalid[2]
            self.invalid_cover += invalid[3]
            
            
            pred_spans.append(pairs.copy())
            


            
            # am component+label
            am_component_target = [(t[3], t[4], self.id2label[t[5]]) for t in ts]+[(t[0], t[1], self.id2label[t[2]]) for t in ts]
            am_component_pred = [(p[3], p[4], self.id2label[p[5]]) for p in pairs]+[(p[0], p[1], self.id2label[p[2]]) for p in pairs]
            
            
            component_ts = set([tuple(t) for t in am_component_target])
            component_ps = set(am_component_pred)
            
            for p in list(component_ps):
                if self.label_mapping[p[-1]]==self.none:
                    component_ps.remove(p)
            for t in list(component_ts):
                if self.label_mapping[t[-1]]==self.none:
                    component_ts.remove(t)
            self.component_metric.update(component_ts,component_ps)
            
            for p in list(component_ps):  # pairs is a 5-tuple
                if p in component_ts:
                    component_ts.remove(p)
                    self.am_component_tp += 1
                else:
                    self.am_component_fp += 1
            self.am_component_fn += len(component_ts)
            
            

            
            # triple -> component&relation
            ts = set([tuple(t) for t in ts])
            ps = set(pairs)
            for t in list(ts):
                if self.none in t:
                    ts.remove(t)
            for p in list(ps):
                if self.none in p:
                    ps.remove(p)
            
            for p in list(ps):
                if p in ts:
                    ts.remove(p)
                    self.triple_tp += 1
                else:
                    self.triple_fp += 1

            self.triple_fn += len(ts)
            
        
        
        return pred_spans

    def get_metric(self, reset=True):
        res = {}

        f, pre, rec = compute_f_pre_rec(1, self.triple_tp, self.triple_fn, self.triple_fp)

        res['triple_f'] = round(f, 4)*100
        res['triple_rec'] = round(rec, 4)*100
        res['triple_pre'] = round(pre, 4)*100


        
        f, pre, rec = compute_f_pre_rec(1, self.am_component_tp, self.am_component_fn, self.am_component_fp)
        res["am_component_f"] = round(f, 4)*100
        res["am_component_rec"] = round(rec, 4)*100
        res["am_component_pre"] = round(pre, 4)*100
        


        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)
        
        overall_info, component_info = self.component_metric.result()
        res['entity_info']=component_info
        res['entity_overall']=overall_info
        
        
        
        res['invalid_len'] = round(self.invalid_len / self.total, 4)
        res['invalid_order'] = round(self.invalid_order / self.total, 4)
        res['invalid_cross'] = round(self.invalid_cross / self.total, 4)
        res['invalid_cover'] = round(self.invalid_cover / self.total, 4)
        if reset:
            self.component_metric.reset()
            self.triple_fp = 0
            self.triple_tp = 0
            self.triple_fn = 0
            self.em = 0
            self.invalid = 0
            self.total = 1e-13

            self.am_component_fp = 0
            self.am_component_tp = 0
            self.am_component_fn = 0
            
            self.invalid_len,self.invalid_order,self.invalid_cross,self.invalid_cover=0,0,0,0
        return res

class Seq2SeqSpanMetric_essay(MetricBase):
    def __init__(self, eos_token_id, num_labels,label_mapping=None, _first=True, eval_token_cls=True):
        super(Seq2SeqSpanMetric_essay, self).__init__()
        self.eos_token_id = eos_token_id
        self.num_labels = num_labels
        self.label_mapping = label_mapping

        self.eval_token_cls = eval_token_cls
            
        self.id2label = dict(zip(label_mapping.values(),label_mapping.keys()))
        self.none = label_mapping["<<none>>"]
        self.component = ["<<MC>>","<<C>>","<<P>>"]
        self.relation = list(set(label_mapping.keys())-set(self.component)-set(["<<none>>","<s>","</s>"]))
        self.component_idx = list(map(lambda x: self.label_mapping[x],self.component))
        self.relation_idx = list(map(lambda x: self.label_mapping[x],self.relation))
        
        self.word_start_index = num_labels + 2  # +2, shift for sos and eos
        self.add_token = list(self.id2label.keys())
        
        self.component_metric = ComponentScore(self.id2label)
        self.ae_oe_fp = 0
        self.ae_oe_tp = 0
        self.ae_oe_fn = 0
        self.triple_fp = 0
        self.triple_tp = 0
        self.triple_fn = 0
        self.em = 0
        self.invalid = 0
        self.total =  1e-13
        self.ae_sc_fp = 0
        self.ae_sc_tp = 0
        self.ae_sc_fn = 0
        
        self.am_component_fp = 0
        self.am_component_tp = 0
        self.am_component_fn = 0
        
        self.component_fp = 0
        self.component_tp = 0
        self.component_fn = 0
        
        self.invalid_len = 0
        self.invalid_order = 0
        self.invalid_cross = 0
        self.invalid_cover  = 0
        
        assert _first is False, "Current metric only supports aspect first"

        self._first = _first


    def build_pair(self,tag_seq,if_skip_cross=True):
        invalid_len = 0
        invalid_order = 0
        invalid_cross = 0
        invalid_cover = 0
        skip = False
        pairs = []
        cur_pair = []
        add_token = self.add_token
        if len(tag_seq):
            for i in tag_seq:
                if i  in self.relation_idx or(i == self.none and len(cur_pair)==6):
                    cur_pair.append(i)
                    if len(cur_pair)!=7:
                        skip = True
                        invalid_len=1
                    elif self.none in cur_pair:
                        tag = set([cur_pair[2],cur_pair[5],cur_pair[6]])
                        if not (cur_pair[2] in add_token and cur_pair[5] in add_token and cur_pair[6] in add_token):
                        # if not tag.issubset(add_token):
                            skip=True
                        else:
                            skip = False
                    else: # 解码长度正确
                        # 检查位置是否正确 <s1,e1,t1,s2,e2,t2,t3>
                        if cur_pair[0]>cur_pair[1] or cur_pair[3]>cur_pair[4]:
                            skip = True
                            invalid_order=1                        
                        elif  not (cur_pair[1]<cur_pair[3] or cur_pair[0]>cur_pair[4]) :
                            skip = True
                            invalid_cover=1
                        if cur_pair[2] in self.relation_idx or cur_pair[5] in self.relation_idx or cur_pair[6] in self.component_idx:
                            if if_skip_cross: # 可以考虑做多一层限制，防止relation 和 component标签错位
                                skip = True
                            invalid_cross=1
                        tag = set([cur_pair[2],cur_pair[5],cur_pair[6]])
                        RC_idx = self.relation_idx+self.component_idx
                        if not (cur_pair[2] in RC_idx and cur_pair[5] in RC_idx and cur_pair[6] in RC_idx):
                        # if not tag.issubset(self.relation_idx+self.component_idx):
                            skip=True
                            invalid_cross=1

                    if skip:
                        skip=False
                    else:
                        pairs.append(tuple(cur_pair))
                    cur_pair = []
                else:
                    cur_pair.append(i)
        # pairs = list(set(pairs))
        return pairs,(invalid_len,invalid_order,invalid_cross,invalid_cover)

    def evaluate(self, target_span, pred, tgt_tokens):
        self.total += pred.size(0)

        pred_eos_index = pred.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()
        target_eos_index = tgt_tokens.flip(dims=[1]).eq(self.eos_token_id).cumsum(dim=1).long()

        pred = pred[:, 1:]  # delete </s>
        tgt_tokens = tgt_tokens[:, 1:]
        pred_seq_len = pred_eos_index.flip(dims=[1]).eq(pred_eos_index[:, -1:]).sum(dim=1)  # bsz
        pred_seq_len = (pred_seq_len - 2).tolist()
        target_seq_len = target_eos_index.flip(dims=[1]).eq(target_eos_index[:, -1:]).sum(dim=1)  # bsz
        target_seq_len = (target_seq_len - 2).tolist()
        pred_spans = []
        for i, (ts, ps) in enumerate(zip(target_span, pred.tolist())):
            em = 0
            ps = ps[:pred_seq_len[i]]
            # self.total += len(ps) // 7
            if pred_seq_len[i] == target_seq_len[i]:
                em = int(
                    tgt_tokens[i, :target_seq_len[i]].eq(pred[i, :target_seq_len[i]]).sum().item() == target_seq_len[i])
            self.em += em

            
            # 计算多种情况的invalid
            # invalid, pairs  = self.compute_invalid_v2(ps)
            
            # 分别计算三种情况的invalid
            pairs,invalid = self.build_pair(ps)
            self.invalid += sum(invalid)
            self.invalid_len += invalid[0]
            self.invalid_order += invalid[1]
            self.invalid_cross += invalid[2]
            self.invalid_cover += invalid[3]
            
            
            pred_spans.append(pairs.copy())
            


            
            # am component+label
            am_component_target = [(t[3], t[4], self.id2label[t[5]]) for t in ts]+[(t[0], t[1], self.id2label[t[2]]) for t in ts]
            am_component_pred = [(p[3], p[4], self.id2label[p[5]]) for p in pairs]+[(p[0], p[1], self.id2label[p[2]]) for p in pairs]
            
            
            component_ts = set([tuple(t) for t in am_component_target])
            component_ps = set(am_component_pred)
            
            for p in list(component_ps):
                if self.label_mapping[p[-1]]==self.none:
                    component_ps.remove(p)
            for t in list(component_ts):
                if self.label_mapping[t[-1]]==self.none:
                    component_ts.remove(t)
            self.component_metric.update(component_ts,component_ps)
            
            for p in list(component_ps):  # pairs is a 5-tuple
                if p in component_ts:
                    component_ts.remove(p)
                    self.am_component_tp += 1
                else:
                    self.am_component_fp += 1
            self.am_component_fn += len(component_ts)
            
            

            
            # triple -> component&relation
            ts = set([tuple(t) for t in ts])
            ps = set(pairs)
            for t in list(ts):
                if self.none in t:
                    ts.remove(t)
            for p in list(ps):
                if self.none in p:
                    ps.remove(p)
            
            for p in list(ps):
                if p in ts:
                    ts.remove(p)
                    self.triple_tp += 1
                else:
                    self.triple_fp += 1

            self.triple_fn += len(ts)
            
        
        
        return pred_spans

    def get_metric(self, reset=True):
        res = {}

        f, pre, rec = compute_f_pre_rec(1, self.triple_tp, self.triple_fn, self.triple_fp)

        res['triple_f'] = round(f, 4)*100
        res['triple_rec'] = round(rec, 4)*100
        res['triple_pre'] = round(pre, 4)*100


        
        f, pre, rec = compute_f_pre_rec(1, self.am_component_tp, self.am_component_fn, self.am_component_fp)
        res["am_component_f"] = round(f, 4)*100
        res["am_component_rec"] = round(rec, 4)*100
        res["am_component_pre"] = round(pre, 4)*100
        


        res['em'] = round(self.em / self.total, 4)
        res['invalid'] = round(self.invalid / self.total, 4)
        
        overall_info, component_info = self.component_metric.result()
        res['entity_info']=component_info
        res['entity_overall']=overall_info
        
        
        res['invalid_len'] = round(self.invalid_len / self.total, 4)
        res['invalid_order'] = round(self.invalid_order / self.total, 4)
        res['invalid_cross'] = round(self.invalid_cross / self.total, 4)
        res['invalid_cover'] = round(self.invalid_cover / self.total, 4)
        if reset:
            self.component_metric.reset()
            self.triple_fp = 0
            self.triple_tp = 0
            self.triple_fn = 0
            self.em = 0
            self.invalid = 0
            self.total = 1e-13

            self.am_component_fp = 0
            self.am_component_tp = 0
            self.am_component_fn = 0
            
            self.invalid_len,self.invalid_order,self.invalid_cross,self.invalid_cover=0,0,0,0
        return res



def compute_f_pre_rec(beta_square, tp, fn, fp):
    r"""

    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    pre = tp / (fp + tp + 1e-13)
    rec = tp / (fn + tp + 1e-13)
    f = (1 + beta_square) * tp / (fn+fp+(1 + beta_square) *tp+ 1e-13)

    return f, pre, rec



class ComponentScore(object):
    def __init__(self, id2label):
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall*100, precision*100, f1*100

    def result(self):
        class_info = {}
        origin_counter = Counter([x[-1] for x in self.origins])
        found_counter = Counter([x[-1] for x in self.founds])
        right_counter = Counter([x[-1] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, label_entities, pre_entities):
        '''

        '''
        # for label_entities, pre_entities in zip(label_paths, pred_paths):
        label_entities = list(set(label_entities))
        pre_entities = list(set(pre_entities))
        self.origins.extend(label_entities)
        self.founds.extend(pre_entities)
        self.rights.extend([pre_entity for pre_entity in pre_entities if pre_entity in label_entities])
