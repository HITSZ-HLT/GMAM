import torch
from .modeling_bart import BartEncoder, BartDecoder, BartModel
from transformers import BartTokenizer
from fastNLP import seq_len_to_mask
from fastNLP.modules import Seq2SeqEncoder, Seq2SeqDecoder, State
import torch.nn.functional as F
from fastNLP.models import Seq2SeqModel
from torch import device, nn
import numpy as np

class FBartEncoder(Seq2SeqEncoder):
    def __init__(self, encoder):
        super().__init__()
        assert isinstance(encoder, BartEncoder)
        self.bart_encoder = encoder

    def forward(self, src_tokens, src_seq_len):
        mask = seq_len_to_mask(src_seq_len, max_len=src_tokens.size(1))
        dict = self.bart_encoder(input_ids=src_tokens, attention_mask=mask, return_dict=True,
                                 output_hidden_states=True)
        encoder_outputs = dict.last_hidden_state
        hidden_states = dict.hidden_states
        return encoder_outputs, mask, hidden_states


class FBartDecoder(Seq2SeqDecoder):
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=True):
        super().__init__()
        assert isinstance(decoder, BartDecoder)
        self.decoder = decoder
        causal_mask = torch.zeros(512, 512).fill_(float('-inf'))
        causal_mask = causal_mask.triu(diagonal=1)
        self.register_buffer('causal_masks', causal_mask.float())
        self.pad_token_id = pad_token_id
        self.label_start_id = label_ids[0]
        self.label_end_id = label_ids[-1]+1
        # 0th position is <s>, 1st position is </s>
        mapping = torch.LongTensor([0, 2]+sorted(label_ids, reverse=False))
        self.register_buffer('mapping', mapping)
        self.src_start_index = len(mapping)  # 加上一个
        hidden_size = decoder.embed_tokens.weight.size(1)
        self.bi_encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))
        if use_encoder_mlp:
            self.encoder_mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.Dropout(0.3),
                                             nn.ReLU(),
                                             nn.Linear(hidden_size, hidden_size))

    def forward(self, tokens, state):
        raise NotImplementedError('not implement')
        
    def decode(self, tokens, state):
        raise NotImplementedError('not implement')


class CaGFBartDecoder(FBartDecoder):
    # Copy and generate,
    def __init__(self, decoder, pad_token_id, label_ids, use_encoder_mlp=False,position_type=0,replace_pos=True):
        super().__init__(decoder, pad_token_id, label_ids, use_encoder_mlp=use_encoder_mlp)
        self.position_type = position_type
        self.replace_pos = replace_pos
        if position_type==0:            
            self.decoder.embed_positions_replace.weight = self.decoder.embed_positions.weight
            repeat_pos = torch.tensor([2,2,3,2,2,3,3])# pad 0 start 1
        elif position_type == 1:
            repeat_pos = torch.tensor([2,3,4,2,3,4,4])# pad 0 start 1
        elif position_type == 2:
            repeat_pos = torch.tensor([2,3,4,5,6,7,8])# pad 0 start 1
        elif position_type == 3:
            repeat_pos = torch.tensor([2,3,4,2,3,4,5])# pad 0 start 1
        elif position_type == 4:
            repeat_pos = torch.tensor([2,2,2,2,2,2,2])
        elif position_type == 5:
            repeat_pos = torch.tensor([2,2,2,2,2,2,2])
        elif position_type == 6:
            repeat_pos = torch.tensor([2,2,3,2,2,3,4])
        elif position_type == 7:
            self.replace_pos=False
            self.decoder.embed_positions.reset_parameters()
            repeat_pos = torch.tensor([2,2,2,2,2,2,2]) # not used
        elif position_type == 8:
            self.decoder.embed_positions_replace.weight = self.decoder.embed_positions.weight
            repeat_pos = torch.tensor([2,2,3,2,2,3,3])# pad 0 start 1
        # 2 2 3 2 2 3 3
        # 2 3 4 5 6 7 8
        # 2 3 4 2 3 4 4
        # 2 3 4 2 3 4 5
        # 1 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3
        pad_pos = torch.tensor(0)
        start_pos = torch.tensor(1)
        self.register_buffer('repeat_pos',repeat_pos)
        self.register_buffer('pad_pos',pad_pos)
        self.register_buffer('start_pos',start_pos)
        

        
    def prepare_RPE(self,tokens,tag_mapped_tokens=None):
        if tag_mapped_tokens == None:
            mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
            tag_mapped_tokens = self.mapping[mapped_tokens]
        bsz,tokens_len = tokens.size()
        repeat_num = tokens_len//7+1 #  1 if int((tokens_len-1)/7)== 0 else
        pos_tokens = self.repeat_pos.repeat(bsz,repeat_num)
        if self.position_type == 4:
            reshape_pos = pos_tokens.view(bsz,-1,7)
            shift_pos = reshape_pos.size(1)
            add_shift_pos = torch.range(0,shift_pos-1).repeat(bsz).view(bsz,-1).unsqueeze(-1)
            reshape_pos = add_shift_pos.to(reshape_pos.device)+reshape_pos
            pos_tokens = reshape_pos.view(bsz,-1).long()
        pos_tokens = torch.cat([self.start_pos.repeat(bsz,1),pos_tokens],dim=-1)
        pos_tokens = pos_tokens[:bsz,:tokens_len]
        pos_tokens = pos_tokens.masked_fill(tag_mapped_tokens.eq(2), self.pad_pos.data)
        return pos_tokens

    def forward(self, tokens, state,CPM_tag=None,):
        
        encoder_outputs = state.encoder_output
        encoder_pad_mask = state.encoder_mask

        first = state.first

        cumsum = tokens.eq(1).flip(dims=[1]).cumsum(dim=-1)
        tgt_pad_mask = cumsum.flip(dims=[1]).ne(cumsum[:, -1:])

        mapping_token_mask = tokens.lt(self.src_start_index)
        mapped_tokens = tokens.masked_fill(tokens.ge(self.src_start_index), 0)
        tag_mapped_tokens = self.mapping[mapped_tokens]

        src_tokens_index = tokens - self.src_start_index # bsz x num_src_token
        src_tokens_index = src_tokens_index.masked_fill(src_tokens_index.lt(0), 0)
        src_tokens = state.src_tokens
        if first is not None:
            src_tokens = src_tokens.gather(index=first, dim=1)
        assert src_tokens_index.max() <1024
        word_mapped_tokens = src_tokens.gather(index=src_tokens_index, dim=1)

        tokens = torch.where(mapping_token_mask, tag_mapped_tokens, word_mapped_tokens)  # bsz x max_len
        tokens = tokens.masked_fill(tgt_pad_mask, self.pad_token_id)
        if self.replace_pos:
            pos_tokens = self.prepare_RPE(tokens=tokens,tag_mapped_tokens=tag_mapped_tokens)
        else:
            pos_tokens = None
        if self.training:
            assert CPM_tag is not None
            # bsz,input_d,_ = tokens.shape()
            if pos_tokens is not None:
                positions = pos_tokens[:, :-1]
            else:
                positions = None
            tokens = tokens[:, :-1]
            decoder_pad_mask = tokens.eq(self.pad_token_id)  # decoder需要让pad位置为1
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=decoder_pad_mask,
                                decoder_causal_mask=self.causal_masks[:tokens.size(1), :tokens.size(1)],
                                return_dict=True,
                                pos_emb = positions,
                                )
        else:
            assert CPM_tag is None
            positions = pos_tokens
            past_key_values = state.past_key_values
            dict = self.decoder(input_ids=tokens,
                                encoder_hidden_states=encoder_outputs,
                                encoder_padding_mask=encoder_pad_mask,
                                decoder_padding_mask=None,
                                decoder_causal_mask=None,
                                past_key_values=past_key_values,
                                use_cache=True,
                                return_dict=True,
                                pos_emb = positions,
                                )
        hidden_state = dict.last_hidden_state  # bsz x max_len x hidden_size
        if not self.training:
            state.past_key_values = dict.past_key_values

        logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),
                                       fill_value=-1e24)

        eos_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[2:3])  # bsz x max_len x 1
        tag_scores = F.linear(hidden_state, self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id])  # bsz x max_len x num_class


        # bsz x max_bpe_len x hidden_size
        src_outputs = state.encoder_output
        if hasattr(self, 'encoder_mlp'):
            src_outputs = self.encoder_mlp(src_outputs)
        
        if hasattr(self, 'bi_encoder_mlp'):
            bi_outputs = self.bi_encoder_mlp(src_outputs)
            bi_tag_scores = F.linear(hidden_state, self.bi_encoder_mlp(self.decoder.embed_tokens.weight[self.label_start_id:self.label_end_id]))

        if first is not None:
            mask = first.eq(0)  # bsz x 1 x max_word_len, 为1的地方是padding
            # bsz x max_word_len x hidden_size
            src_outputs = src_outputs.gather(index=first.unsqueeze(2).repeat(1, 1, src_outputs.size(-1)), dim=1)
        else:
            mask = state.encoder_mask.eq(0)
        mask = mask.unsqueeze(1)
        input_embed = self.decoder.embed_tokens(src_tokens)  # bsz x max_word_len x hidden_size

        word_scores = torch.einsum('blh,bnh->bln', hidden_state, src_outputs)  # bsz x max_len x max_word_len
        gen_scores = torch.einsum('blh,bnh->bln', hidden_state, input_embed)  # bsz x max_len x max_word_len
        avg_word_scores = (gen_scores + word_scores)/2
        mask = mask.__or__(src_tokens.eq(2).cumsum(dim=1).ge(1).unsqueeze(1))
        avg_word_scores = avg_word_scores.masked_fill(mask, -1e32)
        word_scores = word_scores.masked_fill(mask, -1e32)

        logits[:, :, 1:2] = eos_scores
        logits[:, :, 2:self.src_start_index] = tag_scores
        logits[:, :, self.src_start_index:] = avg_word_scores


        bi_logits = torch.einsum('blh,bnh->bln', hidden_state, bi_outputs)  # bsz x max_len x max_word_len
        constrain_logits = hidden_state.new_full((hidden_state.size(0), hidden_state.size(1), self.src_start_index+src_tokens.size(-1)),fill_value=-1e24)
        constrain_logits[:, :, 2:self.src_start_index] = bi_tag_scores
        constrain_logits[:, :, self.src_start_index:] = bi_logits
        constrain_tag = None
        if CPM_tag is not None:
            constrain_tag = CPM_tag.float()[...,2:]
            constrain_logits = constrain_logits[...,2:] 

        return (logits,(constrain_logits,constrain_tag),None)
 
    def decode(self, tokens, state):
        voc_logits, _, token_cls_scores  =  self(tokens, state)
        voc_logits = voc_logits[:,-1]
        return voc_logits, None, token_cls_scores


class BartSeq2SeqModel(Seq2SeqModel):
    @classmethod
    def build_model(cls, bart_model, tokenizer, label_ids, decoder_type=None, copy_gate=False,
                    use_encoder_mlp=False, use_recur_pos=False, tag_first=False,
                    token_cls=False, replace_pos = True,position_type=0):
        model = BartModel.from_pretrained(bart_model,use_cdn=False)
        num_tokens, _ = model.encoder.embed_tokens.weight.shape
        model.resize_token_embeddings(len(tokenizer.unique_no_split_tokens)+num_tokens)
        encoder = model.encoder
        decoder = model.decoder

        if use_recur_pos:
            decoder.set_position_embedding(label_ids[0], tag_first)

        _tokenizer = BartTokenizer.from_pretrained(bart_model)
        for token in tokenizer.unique_no_split_tokens:
            if token[:2] == '<<':
                index = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
                if len(index)>1:
                    raise RuntimeError(f"{token} wrong split")
                else:
                    index = index[0]
                assert index>=num_tokens, (index, num_tokens, token)
                indexes = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(token[2:-2]))
                embed = model.encoder.embed_tokens.weight.data[indexes[0]]
                for i in indexes[1:]:
                    embed += model.decoder.embed_tokens.weight.data[i]
                embed /= len(indexes)
                model.decoder.embed_tokens.weight.data[index] = embed

        encoder = FBartEncoder(encoder)
        
        cls.token_cls = token_cls
            
        label_ids = sorted(label_ids)
        if decoder_type is None:
            assert copy_gate is False
            decoder = FBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids)
        elif decoder_type =='avg_score':
            decoder = CaGFBartDecoder(decoder, pad_token_id=tokenizer.pad_token_id, label_ids=label_ids,
                                              use_encoder_mlp=use_encoder_mlp,position_type=position_type,replace_pos=replace_pos)
        else:
            raise RuntimeError("Unsupported feature.")
        # decoder.replace_pos =  replace_pos
        return cls(encoder=encoder, decoder=decoder)

    def prepare_state(self, src_tokens, src_seq_len=None, first=None):
        encoder_outputs, encoder_mask, hidden_states = self.encoder(src_tokens, src_seq_len)
        src_embed_outputs = hidden_states[0]
        state = BartState(encoder_outputs, encoder_mask, src_tokens, first, src_embed_outputs)
        # setattr(state, 'tgt_seq_len', tgt_seq_len)
        return state

    def forward(self, src_tokens, tgt_tokens, src_seq_len,  first,CPM_tag):
        """

        :param torch.LongTensor src_tokens: source的token
        :param torch.LongTensor tgt_tokens: target的token
        :param torch.LongTensor first: 显示每个, bsz x max_word_len
        :param torch.LongTensor src_seq_len: src的长度
        :param torch.LongTensor tgt_seq_len: target的长度，默认用不上
        :return: {'pred': torch.Tensor}, 其中pred的shape为bsz x max_len x vocab_size
        """
        state = self.prepare_state(src_tokens, src_seq_len, first)
        decoder_output = self.decoder(tokens = tgt_tokens, state = state, CPM_tag = CPM_tag)
        if isinstance(decoder_output, torch.Tensor):
            return {'pred': decoder_output}
        elif isinstance(decoder_output, (tuple, list)):
            return {'pred': decoder_output[0],
                    'constrain_pred':decoder_output[1],
                    }
        else:
            raise TypeError(f"Unsupported return type from Decoder:{type(self.decoder)}")



class BartState(State):
    def __init__(self, encoder_output, encoder_mask, src_tokens, first, src_embed_outputs):
        super().__init__(encoder_output, encoder_mask)
        self.past_key_values = None
        self.src_tokens = src_tokens
        self.first = first
        self.src_embed_outputs = src_embed_outputs

    def reorder_state(self, indices: torch.LongTensor):
        super().reorder_state(indices)
        self.src_tokens = self._reorder_state(self.src_tokens, indices)
        if self.first is not None:
            self.first = self._reorder_state(self.first, indices)
        self.src_embed_outputs = self._reorder_state(self.src_embed_outputs, indices)
        if self.past_key_values is not None:
            new = []
            for layer in self.past_key_values:
                new_layer = {}
                for key1 in list(layer.keys()):
                    new_layer_ = {}
                    for key2 in list(layer[key1].keys()):
                        if layer[key1][key2] is not None:
                            layer[key1][key2] = self._reorder_state(layer[key1][key2], indices)
                            # print(key1, key2, layer[key1][key2].shape)
                        new_layer_[key2] = layer[key1][key2]
                    new_layer[key1] = new_layer_
                new.append(new_layer)
            self.past_key_values = new