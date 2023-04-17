
from fastNLP import LossBase
import torch.nn.functional as F
import torch
from fastNLP import seq_len_to_mask


class Seq2SeqLoss(LossBase):
    def __init__(self,biloss):
        super().__init__()
        self.biloss = biloss

    def get_loss(self,tgt_tokens, tgt_seq_len, pred,constrain_pred):
        """

        :param tgt_tokens: bsz x max_len, [sos, tokens, eos]
        :param pred: bsz x max_len-1 x vocab_size
        :return:
        """
        tgt_seq_len = tgt_seq_len - 1
        mask = seq_len_to_mask(
            tgt_seq_len, max_len=tgt_tokens.size(1) - 1).eq(0)
        tgt_tokens = tgt_tokens[:, 1:].masked_fill(mask, -100)
        loss = F.cross_entropy(target=tgt_tokens, input=pred.transpose(1, 2))
        if self.biloss:
            unlikely_label = constrain_pred[1]
            unlikely_label = unlikely_label[mask.eq(0),:]
            pred = constrain_pred[0][mask.eq(0),:]
            active_unlikely = unlikely_label.ge(0).view(-1) # 取label 0和1 （invlid/valid）
            active_pred = pred.view(-1)[active_unlikely]
            active_unlikely_label = unlikely_label.view(-1)[active_unlikely]
            input = F.sigmoid(active_pred)
            loss_c = F.binary_cross_entropy(target=active_unlikely_label, input=input)
            return loss + loss_c
        else:
            return loss
