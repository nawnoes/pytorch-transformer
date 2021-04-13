from collections import namedtuple

import torch
from torch import nn
from model.util import clones, temperature_sampling
import torch.nn.functional as F
from model.util import log, gumbel_sample, mask_with_tokens, prob_mask_like, get_mask_subset_with_prob
from model.transformer import PositionalEmbedding,Encoder
from transformers.activations import get_activation
from torch.nn import CrossEntropyLoss


Results = namedtuple('Results', [
  'loss',
  'mlm_loss',
  'disc_loss',
  'gen_acc',
  'disc_acc',
  'disc_labels',
  'disc_predictions'
])

class TransformerEncoderModel(nn.Module):
  def __init__(self, vocab_size, dim, emb_dim, max_seq_len, depth, head_num, dropout =0.1):
    super(TransformerEncoderModel,self).__init__()
    self.dim=dim
    self.emb_dim=emb_dim

    self.token_emb= nn.Embedding(vocab_size, emb_dim)
    self.position_emb = PositionalEmbedding(emb_dim, max_seq_len)
    self.encoders = clones(Encoder(d_model=dim, head_num=head_num, dropout=dropout), depth)
    self.norm = nn.LayerNorm(dim)

    if dim != emb_dim:
      self.embeddings_project = nn.Linear(emb_dim, dim)

  def get_input_embeddings(self):
      return self.token_emb

  def set_input_embeddings(self, value):
      self.token_emb = value

  def _tie_or_clone_weights(self, first_module, second_module):
    """ Tie or clone module weights depending of weither we are using TorchScript or not
    """
    if self.config.torchscript:
      first_module.weight = nn.Parameter(second_module.weight.clone())
    else:
      first_module.weight = second_module.weight

    if hasattr(first_module, 'bias') and first_module.bias is not None:
      first_module.bias.data = torch.nn.functional.pad(first_module.bias.data, (0, first_module.weight.shape[0] - first_module.bias.shape[0]),'constant',0)

  def forward(self, input_ids, input_mask):
    x = self.token_emb(input_ids)
    x = x + self.position_emb(input_ids).type_as(x)

    if self.emb_dim != self.dim:
      x = self.embeddings_project(x)

    for encoder in self.encoders:
      x = encoder(x, input_mask)
    x = self.norm(x)

    return x
class GeneratorHead(nn.Module):
  def __init__(self, vocab_size, dim, emb_dim, layer_norm_eps=1e-12):
    super().__init__()
    self.vocab_size = vocab_size

    self.dense = nn.Linear(dim, emb_dim)
    self.activation = F.gelu
    self.norm = nn.LayerNorm(emb_dim, eps=layer_norm_eps)
    self.decoder = nn.Linear(emb_dim, vocab_size, bias=False)
    self.bias = nn.Parameter(torch.zeros(vocab_size))

    # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
    self.decoder.bias = self.bias

  def forward(self, hidden_states, masked_lm_labels=None):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.activation(hidden_states)
    hidden_states = self.norm(hidden_states)

    logits = self.decoder(hidden_states)
    outputs = (logits,)

    if masked_lm_labels is not None:
      loss_fct = nn.CrossEntropyLoss()
      genenater_loss = loss_fct(logits.view(-1, self.vocab_size), masked_lm_labels.view(-1))
      outputs += (genenater_loss,)
    return outputs

class DiscriminatorHead(nn.Module):
  def __init__(self, dim, layer_norm_eps=1e-12):
    super().__init__()
    self.dense = nn.Linear(dim, dim)
    # self.activation = F.gelu
    # self.norm = nn.LayerNorm(dim, eps=layer_norm_eps)
    self.classifier = nn.Linear(dim, 1)

  def forward(self, hidden_states,is_replaced_label = None, input_mask=None):
    hidden_states = self.dense(hidden_states)
    # hidden_states = self.activation(hidden_states)
    # hidden_states = self.norm(hidden_states)
    logits = self.classifier(hidden_states)

    outputs = (logits,)

    if is_replaced_label is not None:
      loss_fct = nn.BCEWithLogitsLoss()

      if input_mask is not None:
        active_loss = input_mask.view(-1, hidden_states.shape[1])==1
        active_logits = logits.view(-1, hidden_states.shape[1])[active_loss]
        active_labels = is_replaced_label[active_loss]
        disc_loss = loss_fct(active_logits, active_labels.float())
      else:
        disc_loss = loss_fct(logits.view(-1, hidden_states.shape[1]), is_replaced_label.float())

      outputs += (disc_loss, )

    return outputs

class Electra(nn.Module):
  def __init__(self,
               config,
               gen_config,
               disc_config,
               num_tokens,
               disc_weight=50.,
               gen_weight=1.,
               temperature=1.):
    super().__init__()
    # Electra Generator
    self.generator = TransformerEncoderModel(vocab_size=num_tokens,
                                             max_seq_len=config.max_seq_len,
                                             dim=gen_config.dim,
                                             emb_dim=gen_config.emb_dim,
                                             depth=gen_config.depth,
                                             head_num=gen_config.head_num)
    self.generator_head = GeneratorHead(vocab_size=num_tokens,
                                        dim=gen_config.dim,
                                        emb_dim=gen_config.emb_dim)
    # Electra Discriminator
    self.discriminator = TransformerEncoderModel(vocab_size=num_tokens,
                                             max_seq_len=config.max_seq_len,
                                             dim=disc_config.dim,
                                             emb_dim=disc_config.emb_dim,
                                             depth=disc_config.depth,
                                             head_num=disc_config.head_num)

    self.discriminator_head = DiscriminatorHead(dim=disc_config.dim)

    self.disc_weight = disc_weight
    self.gen_weight = gen_weight
    self.temperature = temperature

  def tie_embedding_weight(self):
    # 4.2 weight tie the token and positional embeddings of generator and discriminator
    # 제너레이터와 디스크리미네이터의 토큰, 포지션 임베딩을 공유한다(tie).
    self.generator.token_emb = self.discriminator.token_emb
    self.generator.position_emb = self.discriminator.position_emb

  def forward(self, input_ids, input_mask, mlm_label=None):

    gen_output = self.generator(input_ids=input_ids, input_mask=input_mask)
    gen_logits, gen_loss = self.generator_head(gen_output, masked_lm_labels=mlm_label)

    masked_indice = (mlm_label.long() != -100) # mlm 라벨에서 마스킹된 인덱스 찾기

    sample_logits = gen_logits[masked_indice] # use mask from before to select logits that need sampling
    sampled = gumbel_sample(sample_logits, temperature=self.temperature) # sample from sample logits

    disc_input = input_ids.clone() # copy input_ids
    disc_input[masked_indice] = sampled.detach() # inject sample ids
    is_replace_label = (input_ids != disc_input).float().detach() # make is_replace_label

    disc_ouput = self.discriminator(input_ids=disc_input, input_mask=input_mask)
    disc_logits, disc_loss = self.discriminator_head(disc_ouput, is_replaced_label=is_replace_label, input_mask=input_mask)

    # gather metrics
    with torch.no_grad():
      gen_predictions = torch.argmax(gen_logits, dim=-1)
      disc_predictions = torch.round((torch.sign(disc_logits) + 1.0) * 0.5)
      gen_acc = (mlm_label[masked_indice] == gen_predictions[masked_indice]).float().mean()
      disc_acc = 0.5 * (is_replace_label[masked_indice] == disc_predictions[masked_indice]).float().mean() + 0.5 * (is_replace_label[~masked_indice] == disc_predictions[~masked_indice]).float().mean()

    # return weighted sum of losses
    total_loss = self.gen_weight * gen_loss + self.disc_weight * disc_loss
    return total_loss, gen_loss, disc_loss, gen_acc, disc_acc, is_replace_label, disc_predictions

class ElectraMRCHead(nn.Module):
  def __init__(self, dim, num_labels,dropout_prob):
    super().__init__()
    self.dense = nn.Linear(dim, 1*dim)
    self.dropout = nn.Dropout(dropout_prob)
    self.out_proj = nn.Linear(1*dim,num_labels)

  def forward(self, x):
    # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
    x = self.dropout(x)
    x = self.dense(x)
    x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
    x = self.dropout(x)
    x = self.out_proj(x)

    return x

class ElectraMRCModel(nn.Module):
  def __init__(self, electra, dim, num_labels=2, dropout_prob=0.3):
    super().__init__()
    self.electra = electra
    self.mrc_head = ElectraMRCHead(dim, num_labels, dropout_prob)

  def forward(self,
              input_ids=None,
              input_mask=None,
              start_positions=None,
              end_positions=None):

    # 1. electra
    outputs = self.electra(input_ids, input_mask)

    # 2. mrc head
    logits = self.mrc_head(outputs)

    start_logits, end_logits = logits.split(1, dim=-1)
    start_logits = start_logits.squeeze(-1)
    end_logits = end_logits.squeeze(-1)

    if start_positions is not None and end_positions is not None:

      # If we are on multi-GPU, split add a dimension
      if len(start_positions.size()) > 1:
        start_positions = start_positions.squeeze(-1)
      if len(end_positions.size()) > 1:
        end_positions = end_positions.squeeze(-1)

      # sometimes the start/end positions are outside our model inputs, we ignore these terms
      ignored_index = start_logits.size(1)
      start_positions.clamp_(0, ignored_index)
      end_positions.clamp_(0, ignored_index)

      loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
      start_loss = loss_fct(start_logits, start_positions)
      end_loss = loss_fct(end_logits, end_positions)
      total_loss = (start_loss + end_loss) / 2
      return total_loss
    else:
      return start_logits, end_logits