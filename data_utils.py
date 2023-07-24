import os, logging, json, torch
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset
from collections import defaultdict
# from omegaconf import OmegaConf, open_dict
from transformers import T5Tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, text, ent_infos, rel_infos):
        self.text = text
        self.ent_infos = ent_infos
        self.rel_infos = rel_infos


def get_decoder_tokenizer(args):  # mode: ent / rel
    decoder_tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    decoder_tokenizer.add_special_tokens(
        {"bos_token": '<extra_id_0>', "sep_token": '<extra_id_1>'})
    
    for context, token in decoder_tokenizer.special_tokens_map.items():
        print(f"\t\t{context} -- {token}  -- {decoder_tokenizer.encode(token, add_special_tokens=False)}")
    
    decoder_tokenizer.mask_token = None
    return decoder_tokenizer


def read_and_load_data(og_path, args, mode, encoder_tokenizer, ent_tokenizer, rel_tokenizer):
    # if os.path.exists(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth"):
    #     return torch.load(f"cache/{mode}_{args.way_num}_{args.shot_num}.pth")
    
    data = json.load(open(os.path.join(og_path, args.dataset_path, f"{mode}_data.json"), "r"))
    # data = json.load(open(os.path.join(og_path, args.dataset_path, f"{mode}_data.json"), "r"))[:20]  # for test
    
    logger.info("Reading tasks from {}...".format(os.path.join(args.dataset_path, f"{mode}_data.json")))
    logger.info(
        f"For decoder output, the ent label and rel label separator are {ent_tokenizer.sep_token}({ent_tokenizer.sep_token_id}) "
        f"and {rel_tokenizer.sep_token}({rel_tokenizer.sep_token_id})")
    
    bad_token_ins_num = 0
    out_boundary_ins_num = 0
    input_features = []
    for instance_dict in tqdm(data, desc=f"load {mode} data"):
        fea = convert_example_to_features(instance_dict, encoder_tokenizer, args, ent_tokenizer, rel_tokenizer)
        if type(fea) == InputFeatures:
            input_features.append(fea)  # 越界情况跳过
        elif fea == "bad case token":
            bad_token_ins_num += 1
        elif fea == "no ent / no rel":
            out_boundary_ins_num += 1
    
    logger.info(
        f"Get {len(input_features)} instances from file : {mode}_data.json, omit cases {bad_token_ins_num}, oo-boudnary cases {out_boundary_ins_num}")
    return input_features


def convert_example_to_features(
        instance_dict, tokenizer, args, ent_tokenizer, rel_tokenizer
):
    ent_return, rel_return = defaultdict(set), defaultdict(set)
    
    _res = tokenizer(instance_dict['text'],
                     return_offsets_mapping=True,
                     max_length=args.max_seq_len,
                     truncation=True)
    token2char_span_mapping = _res["offset_mapping"]
    # input_ids = _res['input_ids']
    
    if not token2char_span_mapping:  # ! 有特殊字符报错("\u2063") 导致 word_tokens = []
        return "bad case token"
    
    # { 每个token的开始字符的索引: 第几个token } and { end_index : token_index }
    start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
    end_mapping = {j[-1]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
    
    # 将raw_text的下标 与 token的start和end下标对应
    for ent_info in instance_dict["entity_list"]:  # 全都是index索引，label不用额外转换
        span, [start, end], type = ent_info['text'], ent_info['char_span'], ent_info['type']
        
        # GPT2 / flan-t5 特殊编码，前置空格，所以 start -1
        if 'flan-t5' in tokenizer.name_or_path or 'gpt' in tokenizer.name_or_path or 'v1' in tokenizer.name_or_path:
            if start != 0: start -= 1
        
        if start in start_mapping and end in end_mapping:
            start = start_mapping[start]
            end = end_mapping[end]  # 这里 end 就是最后一个 token 的索引
            ent_return[((start, end), span)].add(type)
            # assert tokenizer.decode(input_ids[start: end + 1]).lower().strip() == span.lower()
        
        else:
            print(f"\tEntity {ent_info['char_span']} out of max seq_len {args.max_seq_len}, "
                  f"text {instance_dict['text'][:50]} ...")
    
    for rel_info in instance_dict['relation_list']:
        sub_start, sub_end = rel_info['subj_char_span']
        obj_start, obj_end = rel_info['obj_char_span']
        
        #  GPT2 / flan-t5 特殊编码，前置空格，所以 start -1
        if 'flan-t5' in tokenizer.name_or_path or 'gpt' in tokenizer.name_or_path or 'v1' in tokenizer.name_or_path:
            if sub_start != 0: sub_start -= 1
            if obj_start != 0: obj_start -= 1
        
        if sub_start in start_mapping and sub_end in end_mapping and obj_start in start_mapping and obj_end in end_mapping:
            sub_start, sub_end = start_mapping[sub_start], end_mapping[sub_end]
            obj_start, obj_end = start_mapping[obj_start], end_mapping[obj_end]
            type = rel_info['predicate']
            rel_return[(sub_start, sub_end, obj_start, obj_end)].add(type)
        else:
            print(
                f"\tRelation ({rel_info['subject']}, {rel_info['predicate']}, {rel_info['object']}) out of max seq_len {args.max_seq_len}, "
                f"text {instance_dict['text'][:50]} ...")
    
    ent_bos, ent_sep, ent_eos = ent_tokenizer.bos_token, ent_tokenizer.sep_token, ent_tokenizer.eos_token
    rel_bos, rel_sep, rel_eos = rel_tokenizer.bos_token, rel_tokenizer.sep_token, rel_tokenizer.eos_token
    
    if ent_return and rel_return:  # 直接整理成为 decoder 输出格式 先不加 cls， eos
        for i in ent_return: ent_return[i] = f" {ent_sep} ".join(ent_return[i]) + f" {ent_eos}"
        
        for i in rel_return:
            if len(rel_return[i]) > 2:
                logging.info(f"multiple relation: {i} -- {rel_return[i]}")
            rel_return[i] = f" {rel_sep} ".join(rel_return[i]) + f" {ent_eos}"
        
        # if len(set(list(rel_return.values()))) != len(rel_return.values()):
        #     print(rel_return)
        
        return InputFeatures(instance_dict['text'], ent_return, rel_return)
    else:
        return "no ent / no rel"


class MyDataset(Dataset):
    def __init__(self, data, args, encoder_tokenizer, ent_tokenizer, rel_tokenizer, mode):
        self.data = data
        self.args = args
        self.mode = mode
        self.encoder_tokenizer = encoder_tokenizer
        self.ent_tokenizer = ent_tokenizer
        self.rel_tokenizer = rel_tokenizer
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        seq_dims : 控制维度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs],
                            axis=0)  # label_num, max_seq_len, max_seq_len，注意这里 max_seq_len 是同batch内最长句子的长度
        elif not hasattr(length, '__getitem__'):
            length = [length]
        
        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]
        
        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            
            # pad_width是在各维度的各个方向上想要填补的长度,如（（1，2），（2，2））
            # 表示在第一个维度上水平方向上padding=1,垂直方向上padding=2
            # 在第二个维度上水平方向上padding=2,垂直方向上padding=2。
            # 如果直接输入一个整数，则说明各个维度和各个方向所填补的长度都一样。
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)
        
        return np.array(outputs)
    
    def generate_inference_batch_of_inputs(self, features):
        batch_input_ids, batch_input_mask = [], []
        ent_labels, rel_labels, texts = [], [], []
        for idx, _feature in enumerate(features):
            encoder_txt = self.encoder_tokenizer.encode_plus(_feature.text, max_length=self.args.max_seq_len,
                                                             truncation=True)
            batch_input_ids.append(encoder_txt["input_ids"])
            batch_input_mask.append(encoder_txt["attention_mask"])
            
            ent_labels.append(_feature.ent_infos)
            rel_labels.append(_feature.rel_infos)
            texts.append(_feature.text)
        
        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask)).float()
        
        return batch_input_ids, batch_input_mask, ent_labels, rel_labels, texts
    
    def negative_sample_ent(self, input_ids, ent_pos_tuples, ent_negative_sample):
        sample_ratio = torch.ones(len(input_ids) * len(input_ids)).reshape((len(input_ids), len(input_ids)))
        
        if ent_negative_sample.lower() == 'bernoulli':
            # 过滤不合法span，根据长度进行偏移符合分布
            target_num = len(ent_pos_tuples) * 2
            max_span_len = max(j - i for i, j in ent_pos_tuples)
            for i in range(len(input_ids)):
                for j in range(len(input_ids)):
                    if i > j: sample_ratio[i][j] = 0
                    # if j - i > max_span_len: sample_ratio[i][j] = 0
            
            sample_ratio = target_num * sample_ratio / sample_ratio.sum()
            sample_ratio[sample_ratio > 1] = 1
            sample_ratio = torch.bernoulli(sample_ratio)
        
        # elif ent_negative_sample == "all":
        
        # sample_ratio[:, -1] = 0  # 删除最后的 [sep] token
        # sample_ratio[-1, :] = 0  # 删除最后的 [sep] token
        
        for i, j in ent_pos_tuples:  # 删除 gold 部分
            sample_ratio[i, j] = 0
        
        rels_pos = list(zip(*np.where(sample_ratio > 0)))
        
        rels_texts = [self.encoder_tokenizer.decode(input_ids[i:j + 1]) for i, j in rels_pos]
        return rels_pos, rels_texts
    
    def generate_batch(self, features):
        """
            inputs = {"input_ids": torch.tensor([input_ids])}
            outputs = model(**inputs, labels=torch.tensor([input_ids]))
        """
        batch_input_ids, batch_input_mask = [], []  # for encoder
        instance_txt, rel_pos_p, rel_pos_f, ent_labels, rel_labels = [], [], [], [], []  # for decoder
        ent_pos, ent_txt, ent_pos_neg, ent_txt_neg = [], [], [], []
        
        # Decoder - Prompt 部分：构造模版，保存相对应的位置信息
        ent_prompt = self.args.ent_prompt.replace("[agg_ent_vector]", self.ent_tokenizer.bos_token)
        rel_prompt = self.args.rel_prompt.replace("[agg_rel_vector]", self.rel_tokenizer.bos_token)
        rel_prefixes_p, rel_prefixes_f, ent_prefixes_p, ent_prefixes_f = [], [], [], []
        
        for idx, _feature in enumerate(features):
            encoder_txt = self.encoder_tokenizer.encode_plus(_feature.text,
                                                             max_length=self.args.max_seq_len,
                                                             truncation=True)
            batch_input_ids.append(encoder_txt["input_ids"])
            batch_input_mask.append(encoder_txt["attention_mask"])
            
            ent_tuples = list(_feature.ent_infos.keys())
            ent_pos_tuples = [i[0] for i in ent_tuples]
            ent_txt_tuples = [i[1] for i in ent_tuples]
            ent_pos_tuples2id = {k: i for i, k in enumerate(ent_pos_tuples)}
            
            instance_txt.append(_feature.text)
            ent_pos.append(ent_pos_tuples)  # instance level
            ent_txt.append(ent_txt_tuples)
            
            ent_pos_tuples_neg, ent_txt_tuples_neg = [], []
            if self.args.ent_negative_sample.lower() != "none":
                ent_pos_tuples_neg, ent_txt_tuples_neg = self.negative_sample_ent(encoder_txt["input_ids"],
                                                                                  ent_pos_tuples,
                                                                                  self.args.ent_negative_sample)
            
            ent_pos_neg.append(ent_pos_tuples_neg)
            ent_txt_neg.append(ent_txt_tuples_neg)
            
            this_instance_rel_pos_p = [(ent_pos_tuples2id[(s_s, s_e)], ent_pos_tuples2id[(o_s, o_e)])
                                       for s_s, s_e, o_s, o_e
                                       in list(_feature.rel_infos.keys())]
            rel_pos_p.append(this_instance_rel_pos_p)
            
            this_instance_rel_pos_f = [(i, j)
                                       for i in range(len(_feature.ent_infos))
                                       for j in range(len(_feature.ent_infos))
                                       if (i, j) not in this_instance_rel_pos_p]
            
            assert len(this_instance_rel_pos_p) > 0
            if self.mode != "test" \
                    and self.args.rel_negative_sample == 'bernoulli' \
                    and len(this_instance_rel_pos_f) > 0:
                # and self.mode == "train"
                
                if len(this_instance_rel_pos_p) > len(this_instance_rel_pos_f):
                    sample_id = torch.ones(len(this_instance_rel_pos_f))
                else:
                    sample_id = torch.bernoulli(
                        torch.ones(len(this_instance_rel_pos_f))
                        * len(this_instance_rel_pos_p) / len(this_instance_rel_pos_f)
                    )  # bernoulli sample
                # if sum(sample_id) == 0:  # 如果采样全部为 0，则强制采样一个
                #     sample_id = torch.zeros(len(this_instance_rel_pos_f))
                #     sample_id[random.randint(0, len(this_instance_rel_pos_f) - 1)] = 1
                this_instance_rel_pos_f = [i for i, j in zip(this_instance_rel_pos_f, sample_id) if j == 1]
            # else: 'complete' sample by default
            rel_pos_f.append(this_instance_rel_pos_f)
            
            # TODO: 存入该数据对应的所有的单个的ent/rel 的输出的 text 标签，和前缀信息
            ent_labels.extend(list(_feature.ent_infos.values()))  # 不是句子level的了，而是 aggregation vector 纬度的
            ent_prefixes_p += [ent_prompt.replace("{ent}", i) for i in ent_txt_tuples]
            ent_prefixes_f += [ent_prompt.replace("{ent}", i) for i in ent_txt_tuples_neg]
            
            rel_labels.extend(list(_feature.rel_infos.values()))
            rel_prefixes_p += [
                rel_prompt.replace("{sub}", ent_txt_tuples[sub_]).replace("{obj}", ent_txt_tuples[obj_])
                for (sub_, obj_) in this_instance_rel_pos_p]
            rel_prefixes_f += [
                rel_prompt.replace("{sub}", ent_txt_tuples[sub_]).replace("{obj}", ent_txt_tuples[obj_])
                for (sub_, obj_) in this_instance_rel_pos_f]
        
        non_rel_labels = self.rel_tokenizer.encode(
            f"{self.args.none_label_prompt} {self.rel_tokenizer.eos_token}",
            add_special_tokens=False)
        non_ent_labels = self.ent_tokenizer.encode(
            f"{self.args.none_ent_prompt} {self.ent_tokenizer.eos_token}",
            add_special_tokens=False)
        
        # 将 prefix 和 label 进行拼接得到 input  ，并且设置 label
        assert len(rel_labels) == len(rel_prefixes_p)
        ent_labels = self.ent_tokenizer.batch_encode_plus(ent_labels, add_special_tokens=False).input_ids
        ent_prefixes_p = self.ent_tokenizer.batch_encode_plus(ent_prefixes_p, add_special_tokens=False).input_ids
        ent_prefixes_f = self.ent_tokenizer.batch_encode_plus(ent_prefixes_f,
                                                              add_special_tokens=False).input_ids if ent_prefixes_f else []
        rel_labels = self.rel_tokenizer.batch_encode_plus(rel_labels, add_special_tokens=False).input_ids
        rel_prefixes_p = self.rel_tokenizer.batch_encode_plus(rel_prefixes_p, add_special_tokens=False).input_ids
        rel_prefixes_f = self.rel_tokenizer.batch_encode_plus(rel_prefixes_f, add_special_tokens=False).input_ids
        
        ent_prefix_lens_p = [len(i) for i in ent_prefixes_p]
        ent_prefix_lens_f = [len(i) for i in ent_prefixes_f]
        rel_prefix_lens_p = [len(i) for i in rel_prefixes_p]
        rel_prefix_lens_f = [len(i) for i in rel_prefixes_f]
        
        if "[agg_ent_vector]" in self.args.ent_prompt:
            agg_idx_prefix_ent_p = [i.index(self.ent_tokenizer.bos_token_id) for i in ent_prefixes_p]
            agg_idx_prefix_ent_f = [i.index(self.ent_tokenizer.bos_token_id) for i in ent_prefixes_f]
        else:
            agg_idx_prefix_ent_p, agg_idx_prefix_ent_f = None, None
        
        if "[agg_rel_vector]" in self.args.rel_prompt:
            agg_idx_prefix_rel_p = [i.index(self.rel_tokenizer.bos_token_id) for i in rel_prefixes_p]
            agg_idx_prefix_rel_f = [i.index(self.rel_tokenizer.bos_token_id) for i in rel_prefixes_f]
        else:
            agg_idx_prefix_rel_p, agg_idx_prefix_rel_f = None, None
        
        # padding
        batch_input_ids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_input_mask = torch.tensor(self.sequence_padding(batch_input_mask)).float()
        
        if self.mode == "train":
            ent_labels_p = [prefixes_p + label for prefixes_p, label in zip(ent_prefixes_p, ent_labels)]
            ent_labels_f = [prefixes_f + non_ent_labels for prefixes_f in ent_prefixes_f] if ent_prefixes_f else []
            rel_labels_p = [prefixes_p + label for prefixes_p, label in zip(rel_prefixes_p, rel_labels)]
            rel_labels_f = [prefixes_f + non_rel_labels for prefixes_f in rel_prefixes_f]
            
            # gpt2 不计算 -100 的损失
            ent_inputs_p = torch.tensor(self.sequence_padding(ent_labels_p, value=self.ent_tokenizer.pad_token_id))
            ent_inputs_f = torch.tensor(
                self.sequence_padding(ent_labels_f, value=self.ent_tokenizer.pad_token_id)) if ent_labels_f else []
            rel_inputs_p = torch.tensor(self.sequence_padding(rel_labels_p, value=self.rel_tokenizer.pad_token_id))
            rel_inputs_f = torch.tensor(self.sequence_padding(rel_labels_f, value=self.rel_tokenizer.pad_token_id))
            
            ent_labels_p = torch.tensor(self.sequence_padding(ent_labels_p, value=-100))
            ent_labels_f = torch.tensor(self.sequence_padding(ent_labels_f, value=-100)) if ent_labels_f else []
            rel_labels_p = torch.tensor(self.sequence_padding(rel_labels_p, value=-100))
            rel_labels_f = torch.tensor(self.sequence_padding(rel_labels_f, value=-100))
            
            for i in range(len(ent_labels_p)):  # prefix 部分不计算 loss，所以设置为 -100
                ent_labels_p[i][:ent_prefix_lens_p[i]] = -100
            for i in range(len(ent_labels_f)):  # prefix 部分不计算 loss，所以设置为 -100
                ent_labels_f[i][:ent_prefix_lens_f[i]] = -100
            for i in range(len(rel_labels_p)):
                rel_labels_p[i][:rel_prefix_lens_p[i]] = -100
            for i in range(len(rel_labels_f)):
                rel_labels_f[i][:rel_prefix_lens_f[i]] = -100
            
            # For T5 only, prepare label
            ent_labels_p = torch.cat(
                (ent_labels_p[:, 1:], torch.ones((ent_labels_p.shape[0], 1), dtype=ent_labels_p.dtype) * -100), dim=-1)
            ent_labels_f = torch.cat(
                (ent_labels_f[:, 1:], torch.ones((ent_labels_f.shape[0], 1), dtype=ent_labels_f.dtype) * -100), dim=-1) \
                if ent_labels_f != [] else []
            rel_labels_p = torch.cat(
                (rel_labels_p[:, 1:], torch.ones((rel_labels_p.shape[0], 1), dtype=ent_labels_p.dtype) * -100), dim=-1)
            rel_labels_f = torch.cat(
                (rel_labels_f[:, 1:], torch.ones((rel_labels_f.shape[0], 1), dtype=ent_labels_p.dtype) * -100), dim=-1)
            
            
            return batch_input_ids, batch_input_mask, \
                   ent_labels_p, ent_labels_f, rel_labels_p, rel_labels_f, ent_inputs_p, ent_inputs_f, rel_inputs_p, rel_inputs_f, \
                   ent_pos, ent_pos_neg, rel_pos_p, rel_pos_f, \
                   agg_idx_prefix_rel_p, agg_idx_prefix_rel_f, agg_idx_prefix_ent_p, agg_idx_prefix_ent_f
        
        # else:  # 预测时，只需要前缀
        #     # 构造输入
        #     ent_inputs_p = torch.tensor(self.sequence_padding(ent_prefixes_p, value=self.ent_tokenizer.pad_token_id))
        #     rel_inputs_p = torch.tensor(self.sequence_padding(rel_prefixes_p, value=self.rel_tokenizer.pad_token_id))
        #     rel_inputs_f = torch.tensor(self.sequence_padding(rel_prefixes_f, value=self.rel_tokenizer.pad_token_id))
        #     # 根据输入，构造 mask
        #     ent_mask_p = torch.ones_like(ent_inputs_p).float()
        #     ent_mask_p[ent_inputs_p == self.ent_tokenizer.pad_token_id] = 0
        #     rel_mask_p = torch.ones_like(rel_inputs_p).float()
        #     rel_mask_p[rel_inputs_p == self.rel_tokenizer.pad_token_id] = 0
        #     rel_mask_f = torch.ones_like(rel_inputs_f).float()
        #     rel_mask_f[rel_inputs_f == self.rel_tokenizer.pad_token_id] = 0
        #
        #     ent_prefix_lens_p, rel_prefix_lens_p, rel_prefix_lens_f = \
        #         max(ent_prefix_lens_p), max(rel_prefix_lens_p), max(rel_prefix_lens_f)
        #
        #     return batch_input_ids, batch_input_mask, \
        #            ent_inputs_p, rel_inputs_p, rel_inputs_f, \
        #            ent_mask_p, rel_mask_p, rel_mask_f, \
        #            ent_pos, rel_pos_p, rel_pos_f, \
        #            agg_idx_prefix_rel_p, agg_idx_prefix_rel_f, agg_idx_prefix_ent, \
        #            ent_prefix_lens_p, rel_prefix_lens_p, rel_prefix_lens_f, \
        #            ent_txt, instance_txt, ent_labels, rel_labels
    
    """
     ent_pos, rel_pos_p, rel_pos_f  # 位置信息，用于模型聚合，提供位置信息
     ent_prefix_lens_p, rel_prefix_lens_p, rel_prefix_lens_f  # 前缀的长度，在 inference 时，根据这个数值进行截断输入，并且 matrix 时用于截断输出序列
    """
