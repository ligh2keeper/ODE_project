import copy
from src.model import *


def make_model(device, vocab_size, pad_idx, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout, device)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    emb = nn.Sequential(Embeddings(d_model, vocab_size, device), position)
    model = Seq2Seq(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(d_model, vocab_size, DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        emb,
        emb,
        pad_idx,
        device
    )
    return model



class Evaluator(object):
    def __init__(self, model, device, vocab, max_len=256, vocab_size=63):
        self.max_len = max_len
        self.device = device
        self.vocab_size = vocab_size
        self.model = model
        self.token2id = vocab['token2id']
        self.id2token = vocab['id2token']

    def greedy_decode(self, input):
        src_tensor = torch.LongTensor(input).unsqueeze(0).to(self.device)
        src_mask = self.model.make_src_mask(src_tensor)
        with torch.no_grad():
            enc_output = self.model.encode(src_tensor, src_mask)
        trg_indexes = [self.token2id['<s>']]
        for i in range(self.max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(self.device)
            trg_mask = self.model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output = self.model.decode(enc_output, src_mask, trg_tensor, trg_mask)#[1, trg_len, vocab_size]
            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)
            if pred_token == self.token2id['</s>']:
                break
        return trg_indexes[1:-1]

    def beam_decode(self, input, beam_size=5, alpha=0.75):
        src_tensor = torch.LongTensor(input).unsqueeze(0).to(self.device)
        src_mask = self.model.make_src_mask(src_tensor)
        with torch.no_grad():
            enc_output = self.model.encode(src_tensor, src_mask) #[1, src_len, d_model]

        trg_start = torch.LongTensor([self.token2id['<s>']]).unsqueeze(0).to(self.device)
        trg_mask = self.model.make_trg_mask(trg_start)
        with torch.no_grad():
            output = self.model.decode(enc_output, src_mask, trg_start, trg_mask)#[1, 1, vocab_size]
        log_probs = F.log_softmax(output.squeeze(0), dim=-1)
        init_scores, init_words = torch.topk(log_probs, beam_size)

        generated = src_tensor.new(beam_size, self.max_len).to(self.device)  # upcoming output
        generated.fill_(self.token2id['<pad>'])
        generated[:, 0].fill_(self.token2id['<s>'])
        generated[:, 1] = init_words

        beam_scores = init_scores.transpose(0, 1)
        enc_output = enc_output.expand(beam_size, -1, -1).contiguous() #[beam_size, src_len, d_model]

        not_done = [True for _ in range(beam_size)]
        hyps = []
        hyps_probs = []

        cur_len = 2
        cur_beam_size = beam_size

        for cur_len in range(2, self.max_len-1):
            trg_mask = self.model.make_trg_mask(generated[:, :cur_len])
            with torch.no_grad():
                output = self.model.decode(enc_output[:cur_beam_size], src_mask, generated[:, :cur_len], trg_mask) #[beam_size, cur_len, vocab_size]
            log_probs = F.log_softmax(output, dim=-1)[:, -1] #[beam_size, trg_len, vocab_size]

            scores = log_probs + beam_scores
            scores = scores.view(cur_beam_size*self.vocab_size)
            next_scores, next_words = torch.topk(scores, cur_beam_size)
            beam_scores = next_scores.view(cur_beam_size, 1)

            old_generated = torch.empty_like(generated).copy_(generated)
            idxs_to_save = []
            for i, word in enumerate(next_words):
                word_id = word.item()
                beam_num = word_id // self.vocab_size
                id_in_beam = word_id % self.vocab_size
                generated[i] = old_generated[beam_num].clone()
                generated[i, cur_len] = id_in_beam
                if id_in_beam != self.token2id['</s>']:
                    idxs_to_save.append(i)
                else:
                    hyps.append(generated[i, 1:cur_len].tolist())
                    hyps_probs.append(beam_scores[i].item()/(len(hyps[-1]))**alpha)

            generated = generated[idxs_to_save]
            beam_scores = beam_scores[idxs_to_save]
            cur_beam_size = len(idxs_to_save)
            if cur_beam_size == 0:
                break
            del old_generated

        hyps_probs, hyps = zip(*sorted(zip(hyps_probs, hyps), reverse=True))

        return hyps[0], hyps, hyps_probs
