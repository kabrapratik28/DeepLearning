
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import codecs
import unidecode
import string
import copy
import random
from random import shuffle
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch import functional as F

# Eval metric
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
chencherry = SmoothingFunction()

# reproduce !
random.seed(11)

#https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
from logger import Logger
logger = Logger('/gpu_data/pkabara/data/translation/logs')

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# In[2]:


translator = str.maketrans('', '', string.punctuation)
filename = "/Users/pkabara/translation/deu.txt"


# In[3]:


def pair_of_sentences(filename):
    pairs = []
    with codecs.open(filename,'r',encoding='utf8') as f:
        for line in f:
            line = line.strip()
            
            # lower case
            line = line.lower()
            # remove accents "ą/ę/ś/ć" with "a/e/s/c"
            line = unidecode.unidecode(line)
            # remove punctuations
            line = line.translate(translator)
            #remove digit
            line = ''.join(i for i in line if not i.isdigit())
            
            sentence, conversion = line.split("\t") 
            pairs.append([sentence,conversion])
            
    return pairs    


# In[4]:


class Language:
    def __init__(self):
        self.SOS = "<SOS>" # Start of Sentence
        self.EOS = "<EOS>" # End of Sentence
        self.counter = 3
        self.word2index = {"<PAD>":0,"<SOS>":1, "<EOS>":2}
        self.index2word = {0:"<PAD>",1:"<SOS>", 2:"<EOS>"}
        self.word2count = {}
    
    def add_word(self,word):
        if word in self.word2index:
            self.word2count[word] += 1
        else:
            self.word2index[word] = self.counter
            self.index2word[self.counter] = word
            self.word2count[word] = 1
            self.counter += 1
    
    def add_sentense(self,words):
        for each in words:
            self.add_word(each)
    
    def filter_words(self,threshold):
        counter = 3
        word2index = {"<PAD>":0,"<SOS>":1, "<EOS>":2}
        index2word = {0:"<PAD>",1:"<SOS>", 2:"<EOS>"}
        word2count = {}
        tlen = len(self.word2index)
        for each in self.word2count:
            count = self.word2count[each]
            if count >= threshold:
                word2index[each] = counter
                index2word[counter] = each
                word2count[each] = self.word2count[each]
                counter += 1
        self.word2index = word2index
        self.index2word = index2word
        self.word2count = word2count
        self.counter = counter
        print ("words keep ratio {}".format(len(self.word2index)/tlen))


# In[5]:


def is_all_word_present(list_words,language):
    is_include = True
    for word in list_words:
        if word not in language.word2index:
            is_include = False
            break
    return is_include

def filter_sentences(splitted_sentence_pair,language1,language2,max_len=20,min_len=3):
    new_list = []
    for each in splitted_sentence_pair:
        is_include = (min_len <= len(each[0]) <= max_len) and (min_len <= len(each[1]) <= max_len)
        is_include = is_include and is_all_word_present(each[0],language1) and is_all_word_present(each[1],language2)
        if is_include:
            new_list.append([each[0],each[1]])
    return new_list


# In[6]:


sentence_pair = pair_of_sentences(filename)
splitted_sentence_pair = []
for each in sentence_pair:
    splitted_sentence_pair.append((each[0].split(" "), each[1].split(" ")))

del sentence_pair

english = Language()
german = Language()

for each in splitted_sentence_pair:
    english.add_sentense(each[0])
    german.add_sentense(each[1])
    
english.filter_words(3)
german.filter_words(3)

tlen = len(splitted_sentence_pair)
splitted_sentence_pair = filter_sentences(splitted_sentence_pair,english,german)
tlen2 = len(splitted_sentence_pair)
print ("before {} after {} ratio {}".format(tlen,tlen2,tlen2/tlen))

# shuffle !
shuffle(splitted_sentence_pair)

number_val = 50
val_splitted_sentence_pair = splitted_sentence_pair[-number_val:]
train_splitted_sentence_pair  = splitted_sentence_pair[:-number_val]
del splitted_sentence_pair


# In[7]:


class Dataset:
    def __init__(self,sentence_pairs,src_lang,target_lang,batch_size=32):
        self.sentence_pairs = sentence_pairs
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.len = len(self.sentence_pairs)
        self.c = 0 
        self.batch_size = batch_size
    
    def add_eos(self,pairs):
        for i in range(len(pairs)):
            pairs[i][0].append("<EOS>")
            pairs[i][1].append("<EOS>")
    
    def get_max_len(self,pairs):
        m1, m2 = 0, 0
        src, tar = [], []
        for each in pairs:
            l1 = len(each[0])
            l2 = len(each[1])
            m1 = max(m1,l1)
            m2 = max(m2,l2)
            src.append(l1)
            tar.append(l2)
        return m1,m2,src,tar
    
    def pad_src_target(self,pairs,src_max,target_max):
        for i in range(len(pairs)):
            pairs[i][0] += ["<PAD>",] * (src_max - len(pairs[i][0]))
            pairs[i][1] += ["<PAD>",] * (target_max - len(pairs[i][1]))
    
    def covert_sentence_index(self,sentence,language):
        words = []
        for each in sentence:
            words.append(language.word2index[each])
        return words
    
    def convert_to_index(self,pairs):
        new_pairs = []
        for each in pairs:
            indexed_sen_src = self.covert_sentence_index(each[0],self.src_lang)
            indexed_sen_tar = self.covert_sentence_index(each[1],self.target_lang)
            new_pairs.append([indexed_sen_src,indexed_sen_tar])
        return new_pairs
            
    def get_batch(self):
        pairs = []
        for i in range(self.batch_size):
            if self.c >= self.len:
                shuffle(self.sentence_pairs)
                self.c = 0
            pairs.append(self.sentence_pairs[self.c])
            self.c += 1
        
        # don't want to affect original pairs ... 
        # make a deepcopy 
        pairs = copy.deepcopy(pairs)
        
        # add end of sentence tag !
        self.add_eos(pairs)
        # sort by reverse as torch nn pack_padded_sequence required !
        pairs.sort(key=lambda p: len(p[0]), reverse=True)
        
        m1, m2, src_lens, tar_lens = self.get_max_len(pairs)
        
        # pad source language with m1 and target language with m2
        self.pad_src_target(pairs,m1,m2)
        
        # convert to index 
        pairs = self.convert_to_index(pairs)
        
        src, tar = zip(*pairs)
        
        # batch * timesteps * word
        return src, tar, src_lens, tar_lens


# In[8]:


class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=2,dropout=0.5):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,dropout=dropout,
                          bidirectional=True)
    
    def forward(self,input_data,input_len,hidden_data=None):
        """
        # time x batch x words (T x B x W)
        """
        input_embedded = self.embedding(input_data) # T x B x W x Embed_Size
        input_embedded = torch.nn.utils.rnn.pack_padded_sequence(input_embedded,input_len) # T x B x W x Embed_Size
        output, hidden = self.gru(input_embedded,hidden_data)
        output, output_len = torch.nn.utils.rnn.pad_packed_sequence(output)
        # output_len will alwatys be same as input_len !
        # add forward pass output and backward pass output (Bidirectional RNN) 
        output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:] # T x B x Hidden_Size
        return output, hidden


# In[9]:


class Attn(nn.Module):
    def __init__(self,score_type,hidden_size):
        super(Attn,self).__init__()
        if score_type not in ('dot','concat','general'):
            raise ValueError('score should be one of the in dot, concat, general !')
        
        self.score_type = score_type
        self.hidden_size = hidden_size 
        
        if score_type=='general':
            self.w_a = torch.nn.Linear(hidden_size,hidden_size)
        if score_type=='concat':
            self.w_a = torch.nn.Linear(2*hidden_size,hidden_size)
            self.v_a = torch.nn.Parameter(torch.FloatTensor(hidden_size))
        
    def forward(self,encoder_outputs,hidden_state):
        """
        encoder_outputs # T x B x Hidden_Size
        hidden_state # B x Hidden_Size
        """
        hidden_state = hidden_state.unsqueeze(0) # 1 x B x Hidden_Size
        activation = None
        if self.score_type=='dot':
            dot_prod = hidden_state * encoder_outputs # T x B x Hidden_Size
            dot_prod = torch.sum(dot_prod,dim=2) # T x B 
            activation = torch.softmax(dot_prod,dim=0) # T x B (across time softmax)
        if self.score_type=='general':
            # only 1st step is different than dot !
            dot_prod = self.w_a(encoder_outputs) # T x B x Hidden_Size 
            dot_prod = hidden_state * encoder_outputs # T x B x Hidden_Size
            dot_prod = torch.sum(dot_prod,dim=2) # T x B 
            activation = torch.softmax(dot_prod,dim=0) # T x B (across time softmax)
        if self.score_type=='concat':
            # concat !
            T = encoder_outputs.shape[0]
            hidden_state = hidden_state.repeat((T,1,1)) # T x B x Hidden_Size
            # one decoder step concatenated to all encoder states
            concat_tenc_1dec = torch.cat((encoder_outputs,hidden_state),dim=2)  # T x B x (2*Hidden_Size)
            concat_tenc_1dec = torch.tanh(self.w_a(concat_tenc_1dec)) # T x B x Hidden_Size
            dot_prod = self.v_a * concat_tenc_1dec # T x B x Hidden_Size
            dot_prod = torch.sum(dot_prod,dim=2) # T x B 
            activation = torch.softmax(dot_prod,dim=0) # T x B (across time softmax)
        return activation


# In[10]:


class Decoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers=2,dropout=0.5,score_type='concat'):
        """
        input_size => number of words in target language
        """
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size,hidden_size)
        self.gru = nn.GRU(input_size=hidden_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,dropout=dropout,
                          bidirectional=False)
        self.attn = Attn(score_type,hidden_size)
        self.w_c = torch.nn.Linear(2*hidden_size,hidden_size)
        self.w_s = torch.nn.Linear(hidden_size,input_size)
        
    def forward(self,previous_hidden,input_data,encoder_outputs):
        """
        input_data => B x Word
        previous_hidden => num_layers x Batch x hidden_size
        encoder_outputs => T x B x Hidden_Size
        """
        embedded_input = self.embedding(input_data) # B x Word x embedding_size
        dummy_time_embedded_input = embedded_input.unsqueeze(0) # 1 x B x Word x embedding_size
        output, hidden = self.gru(dummy_time_embedded_input,previous_hidden)
        # output # 1 x B x hidden_size
        output = output.squeeze(0) # B x hidden_size
        attention = self.attn.forward(encoder_outputs,output) # T x B
        attention = attention.unsqueeze(2) # T x B X 1
        weighted_context = encoder_outputs * attention # T x B x Hidden_Size
        weighted_context = torch.sum(weighted_context,dim=0) # B x Hidden_Size
        contexted_output = torch.cat((weighted_context,output),dim=1) # B x (2*Hidden_Size)
        contexted_output = torch.tanh ( self.w_c(contexted_output) ) # equation 5 # B x Hidden_Size
        contexted_output = torch.softmax ( self.w_s(contexted_output) ,dim=1) # equation 6 # B x input_size
        return contexted_output, hidden


# In[11]:


def loss(target,prediction,mask):
    """
    # target => Batch
    # prediction => Batch * Words
    # mask => Batch
    """
    non_zero_ele = mask.sum()
    pred_at_y = torch.gather(prediction,dim=1,index=target.view((-1,1)))
    loss = -torch.log(pred_at_y)
    masked_selected_loss = loss.masked_select(mask.view((-1,1)))
    total_avgloss = masked_selected_loss.mean()
    return total_avgloss, non_zero_ele


# In[12]:


def train_a_batch(encoder,decoder,src,src_lens,tar,tar_lens,
                  max_tar_len,
                  encoder_optimizer,decoder_optimizer,
                  num_layers, num_directions, batch, hidden_size,
                  teacher_forcing_ratio=0.75,clip=50.0):
        """
        src = B x T x word_indexes
        tar = B x T x word_indexes
        """
        # train mode
        encoder.train()
        decoder.train()
        
        # clean out gradients
        encoder_optimizer.zero_grad()   # zero the gradient buffers
        decoder_optimizer.zero_grad()   # zero the gradient buffers
        
        # convert input to tensors !
        src = torch.LongTensor(src)
        src = src.transpose(1,0) #  T x B x words
        src_lens = torch.LongTensor(src_lens) # B
        tar = torch.LongTensor(tar)
        tar = tar.transpose(1,0) #  T x B x words
        tar_lens = torch.LongTensor(tar_lens) # B
        
        src = src.to(device)
        src_lens = src_lens.to(device)
        tar = tar.to(device)
        tar_lens = tar_lens.to(device)
        
        # pass inputs from encoder
        enc_output, enc_hidden = encoder.forward(src,src_lens)
        enc_hidden = enc_hidden.view(num_layers, num_directions, batch, hidden_size)
        enc_forward_hidden_layer = enc_hidden[:,0] # out of bidirectional GRU
        # try to comment below line check what happens ! (Nothing happen on CPU but GPU get below error)
        # it throws and exception ... after googling the exception (rnn: hx is not contiguous)
        # https://stackoverflow.com/questions/48915810/pytorch-contiguous
        enc_forward_hidden_layer = enc_forward_hidden_layer.contiguous()
        
        # decoder intial input # <SOS> start of sentence (its mapped to one in our case)
        dec_input = torch.ones((batch)).long() # B 
        dec_input = dec_input.to(device)
        is_teacher_force = random.uniform(0, 1) < teacher_forcing_ratio
        
        # loop over from decoder
        total_avgloss, total_loss, total_count = 0.0, 0.0, 0.0
        if is_teacher_force:
            # feed actual output
            dec_hidden = enc_forward_hidden_layer
            for i in range(max_tar_len):
                curr_target = tar[i]
                curr_mask = i < tar_lens
                if i!=0:
                    # fisrt input will be <SOS>
                    # i-1 because first word will be feeded to second decoder (check the seq2seq fig. online)
                    dec_input = tar[i-1]
                # dec_hidden assign to self to for next step
                dec_pred, dec_hidden = decoder.forward(dec_hidden, dec_input, enc_output)
                # dec_pred # B x vocab_size_of_translated_lang
                curr_avgloss, curr_non_zero = loss(curr_target,dec_pred,curr_mask)
                total_avgloss += curr_avgloss
                total_count += curr_non_zero.item()
                total_loss += curr_avgloss.item() * curr_non_zero.item() 
        else:
            # feed predicted output by decoder
            dec_hidden = enc_forward_hidden_layer
            for i in range(max_tar_len):
                curr_target = tar[i]
                curr_mask = i < tar_lens
                # dec_hidden assign to self to for next step
                dec_pred, dec_hidden = decoder.forward(dec_hidden, dec_input, enc_output)
                # dec_pred # B x vocab_size_of_translated_lang
                # assign previous output as input
                dec_input = dec_pred.argmax(dim=1)
                # loss calculation
                curr_avgloss, curr_non_zero = loss(curr_target,dec_pred,curr_mask)
                total_avgloss += curr_avgloss
                total_count += curr_non_zero.item()
                total_loss += curr_avgloss.item() * curr_non_zero.item() 
        
        # backpropagate loss
        total_avgloss.backward()
        
        # clip the gradients # save from exploring gradient
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
        
        # optimizer perform the step # Does the update
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        return total_loss / total_count


# In[13]:


class beam_candidate:
    def __init__(self):
        # candidate's backtacking history
        self.output_word_index_till_now = []
        # log probability to avoid underflow => a*b = log(a)+log(b)
        self.log_prob_till_now = 0 
        # required for next state decoder RNN
        self.latest_hidden_state_generated = None
        # https://www.coursera.org/lecture/nlp-sequence-models/refinements-to-beam-search-AkjG2
        self.length_normalized_res = 0.0
        self.last_word_seen = 1 # <SOS> (this can also get from above array last element)
        
    def add_element(self,word_index,prob,latest_hidden_state):
        self.output_word_index_till_now.append(word_index)
        self.log_prob_till_now += np.log(prob)
        self.latest_hidden_state_generated = latest_hidden_state
        self.length_normalized_res = 1.0 / (len(self.output_word_index_till_now)**0.7) * self.log_prob_till_now
        self.last_word_seen = word_index
        return self
    
    def deepcopy(self):
        b = beam_candidate()
        b.output_word_index_till_now = self.output_word_index_till_now[:]
        b.log_prob_till_now = self.log_prob_till_now
        b.latest_hidden_state_generated = self.latest_hidden_state_generated
        b.length_normalized_res = self.length_normalized_res
        b.last_word_seen = self.last_word_seen
        return b
    
    # debug purpose printing the results !
    def __str__(self):
        return "output_word_index_till_now : " + str(self.output_word_index_till_now) + " length_normalized_res " + str(self.length_normalized_res)
    
    def __repr__(self):
        return "output_word_index_till_now : " + str(self.output_word_index_till_now) + " length_normalized_res " + str(self.length_normalized_res)
        
def beam_search(enc,dec,beam_width,input_seq, num_layers, num_directions,hidden_size, max_lenght=30):
    """
    # this beam search is not based on batch !
    input = word_indexes tensor
    """
    # beam search for validation and testing !
    enc.eval()
    dec.eval()
    
    # convert input to tensors !
    src = torch.LongTensor(input_seq)
    src = src.view((-1,1)) # T x B x words 
    src_len = torch.LongTensor([len(input_seq),]) # 1 x lenght # dummy batched version
    
    src = src.to(device)
    src_lens = src_len.to(device)
    
    enc_output, enc_hidden = enc.forward(src,src_lens)
    enc_hidden = enc_hidden.view(num_layers, num_directions, 1, hidden_size) # batch = 1 , dummy batch
    enc_forward_hidden_layer = enc_hidden[:,0]
    enc_forward_hidden_layer = enc_forward_hidden_layer.contiguous()
    
    dec_input = torch.LongTensor([1,]) # dummy batch # <SOS> as first character !
    dec_input = dec_input.to(device)
    
    start_candidate = beam_candidate().add_element(1,1,enc_forward_hidden_layer)
    
    # beam search start !
    candidates = [start_candidate]
    next_round_candidates = []
    results = []
    for c in range(max_lenght):
        if len(results) >= beam_width:
            break
        
        if len(candidates)==0:
            break
        for each in candidates:
            # pass through decoder
            last_word = torch.LongTensor([each.last_word_seen,]).to(device) # dummy batched tensor
            dec_pred, dec_hidden = dec.forward(each.latest_hidden_state_generated,last_word,
                        enc_output)
            dec_pred = dec_pred.squeeze(0) # dummy batch removed
            topk_prob, topk_indexes = torch.topk(dec_pred,beam_width)
            for each_prob, each_index in zip(topk_prob,topk_indexes):
                each_next = each.deepcopy()
                each_next.add_element(each_index.item(), each_prob.item(), dec_hidden)
                next_round_candidates.append(each_next)
        
        # sort the candidates and pick top B candidates (B=beam_width)
        next_round_candidates.sort(key=lambda x: x.length_normalized_res, reverse=True)
        # before picking top beam width candidates
        # select candidates with <EOS> and put in the results 
        # else add them for further search
        next_round_candidates_top_b = []
        for each_cand in next_round_candidates:
            # take top B candidates only (B=beam width)
            if len(next_round_candidates_top_b) >= beam_width:
                break
                
            if each_cand.last_word_seen == 2 or c==max_lenght-1: # check for <EOS> or last round 
                results.append(each_cand)
            else: 
                next_round_candidates_top_b.append(each_cand)
                
        candidates = next_round_candidates_top_b[:]
        next_round_candidates = []
        
    # final candidates ... top picks !
    results.sort(key=lambda x: x.length_normalized_res, reverse=True)
    best_candidate_words = results[0].output_word_index_till_now
    # remove <SOS> and <EOS> from best output !
    if 1 in best_candidate_words:
        best_candidate_words.remove(1)
    if 2 in best_candidate_words:
        best_candidate_words.remove(2)
    return best_candidate_words


# In[14]:


def validation(enc,dec,beam_width,num_layers,num_directions,hidden_size):
    """
    returns a blue score for test dataset
    """
    batch = 1
    number_iter = len(val_splitted_sentence_pair)
    val_dataset = Dataset(val_splitted_sentence_pair,english,german,batch_size=1)
    blue_scores = []
    # dataset get_batch having infinite loop ... get limited loop and iterate once over data
    for i in range(number_iter):
        src, tar, src_lens, tar_lens = val_dataset.get_batch()
        # target batch size is 1 # remove <EOS> tag !
        if 2 in tar[0]:
            tar[0].remove(2)
        res = beam_search(enc,dec,beam_width=beam_width,input_seq=src[0],num_layers=num_layers, 
                          num_directions=num_directions,hidden_size=hidden_size)
        score = sentence_bleu(tar, res,smoothing_function=chencherry.method2)
        blue_scores.append(score)
    return np.mean(blue_scores)


# In[15]:


def save_checkpoint(state,file_name):
    torch.save(state, file_name)
    
def load_checkpoint(file_name,enc_model,dec_model,enc_optimizer,dec_optimizer):
    epoch, best_acc = 0 , 0.0
    if os.path.isfile(file_name):
        print("=> loading checkpoint '{}'".format(file_name))
        checkpoint = torch.load(file_name)
        
        epoch = int(checkpoint['epoch'])
        best_acc = checkpoint['best_acc']
        
        enc_model.load_state_dict(checkpoint['enc_state_dict'])
        dec_model.load_state_dict(checkpoint['dec_state_dict'])
        enc_optimizer.load_state_dict(checkpoint['enc_optimizer'])
        dec_optimizer.load_state_dict(checkpoint['dec_optimizer'])
        
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(file_name, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(file_name))
    
    return epoch, best_acc, enc_model, dec_model, enc_optimizer, dec_optimizer


# In[16]:


def train():
    learning_rate = 0.0001
    clip = 50.0
    teacher_forcing_ratio = 0.75
    num_layers, num_directions, batch, hidden_size = 2, 2, 32, 500
    beam_width = 30
    num_epochs = 500000
    print_loss_every_iter = 100
    val_every_iter = 100
    save_model_every_iter = 1000
    tensorboard_histogram_display = 5000
    file_name = "/gpu_data/pkabara/data/translation/seq2seq"
    dataset = Dataset(train_splitted_sentence_pair,english,german,batch_size=batch)
    
    enc = Encoder(len(english.word2index),hidden_size,num_layers=num_layers)
    dec = Decoder(len(german.word2index),hidden_size)
    enc_optimizer = optim.Adam(enc.parameters(), lr=learning_rate)
    dec_optimizer = optim.Adam(dec.parameters(), lr=learning_rate)
    
    enc = enc.to(device)
    dec = dec.to(device)
    
    epoch, best_acc, enc, dec, enc_optimizer, dec_optimizer = load_checkpoint(file_name,enc,dec,enc_optimizer,dec_optimizer)
    
    running_loss = 0.0
    running_count = 0
    
    for i in range(epoch,epoch+num_epochs):
        src, tar, src_lens, tar_lens = dataset.get_batch()
        max_tar_len = max(tar_lens)
        
        batch_loss = train_a_batch(enc,dec,src,src_lens,tar,tar_lens,
                  max_tar_len,
                  enc_optimizer,dec_optimizer,
                  num_layers, num_directions, batch, hidden_size,
                  teacher_forcing_ratio=teacher_forcing_ratio,clip=clip)
        
        running_loss += batch_loss
        running_count += 1
        
        # for logging
        info = {'loss':batch_loss}
        
        if running_count % print_loss_every_iter == 0:
            print ("Iteration {} Running loss {}".format(running_count, running_loss/running_count))
        
        if running_count % val_every_iter == 0:
            score = validation(enc,dec,beam_width,num_layers,num_directions,hidden_size)
            print ("Iteration {} Validation BLUE {}".format(running_count, score))
            info['accuracy'] = score
            if score > best_acc:
                best_acc = score
        
                # save every some iterations ... if running_count % save_model_every_iter == 0:
                state = {
                    'epoch': epoch+1,
                    'best_acc': best_acc,
                    'enc_state_dict': enc.state_dict(),
                    'dec_state_dict': dec.state_dict(),
                    'enc_optimizer' : enc_optimizer.state_dict(),
                    'dec_optimizer' : dec_optimizer.state_dict()
                }
                save_checkpoint(state,file_name)
                print ("saved checkpoint !")
                
        # update in tensorboard !
        for tag, value in info.items():
            logger.scalar_summary(tag, value, i)
        
        # model weights and gradients # heavy write operation
        # write after some iterations
        if running_count % tensorboard_histogram_display == 0:
            for name,model in {"enc":enc,"dec":dec}.items():
                # 2. Log values and gradients of the parameters (histogram summary)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    # model names appended !
                    tag = name + "/" + tag
                    logger.histo_summary(tag, value.data.cpu().numpy(), i)
                    logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i)
        


# In[17]:


train()

