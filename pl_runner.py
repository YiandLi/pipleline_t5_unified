import json
import sys
import warnings

from omegaconf import open_dict

warnings.filterwarnings("ignore")

sys.path.append("./")
import logging
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5EncoderModel
from Metric import Metric
from data_utils import read_and_load_data, MyDataset, get_decoder_tokenizer
from pl_model import LitModel
import os

os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(config):
    if config.agg_method.lower() == "biaffine":
        from biaffine_model import CustomT5ForConditionalGeneration, ScFreeModel
    else:
        from concat_model import CustomT5ForConditionalGeneration, ScFreeModel
    
    og_path = hydra.utils.get_original_cwd()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"========== hydra config =========")
    for k, v in config.items():
        logger.info(f"{k}:{v}")
    
    if config.model_name.startswith("../"):
        with open_dict(config):
            config.model_name = os.path.join(og_path, config.model_name)
            logging.info(f"The model will be loaded from : {config.model_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"========== device: {device} =========")
    if torch.cuda.is_available():
        logger.info("========== " + f"Using Gpu: {torch.cuda.get_device_name(0)} " + " =========")
    
    seed_everything(42, workers=True)
    # TODO: get tokenizer
    encoder_tokenizer = T5TokenizerFast.from_pretrained(config.model_name)
    ent_tokenizer = get_decoder_tokenizer(config)
    rel_tokenizer = ent_tokenizer
    
    # rel_tokenizer = get_decoder_tokenizer(config)
    
    # TODO: get model
    ent_decoder = CustomT5ForConditionalGeneration.from_pretrained(config.model_name,
                                                                   bos_token_id=ent_tokenizer.bos_token_id,
                                                                   pad_token_id=ent_tokenizer.pad_token_id,
                                                                   eos_token_id=ent_tokenizer.eos_token_id,
                                                                   # add_cross_attention=True,
                                                                   # ignore_mismatched_sizes=True
                                                                   )
    
    rel_decoder = CustomT5ForConditionalGeneration.from_pretrained(config.model_name,
                                                                   bos_token_id=rel_tokenizer.bos_token_id,
                                                                   pad_token_id=rel_tokenizer.pad_token_id,
                                                                   eos_token_id=rel_tokenizer.eos_token_id,
                                                                   # add_cross_attention=True,
                                                                   # ignore_mismatched_sizes=True
                                                                   )
    
    # ent_decoder.encoder = None
    # rel_decoder.encoder = None
    
    # label_text_encoder = T5EncoderModel.from_pretrained(config.model_name).get_input_embeddings()  # 只用浅层表示即可
    
    encoder = T5EncoderModel.from_pretrained(config.model_name)
    
    if config.share_emb:
        ent_decoder.shared = rel_decoder.shared
        encoder.shared = rel_decoder.shared
    
    # span_encoder = T5EncoderModel.from_pretrained(config.model_name)
    
    # TODO: get data
    assert config.ent_negative_sample.lower() in ['none', 'all', 'bernoulli'], \
        f"{config.ent_negative_sample} [ .lower() ] not in ['none', 'all', 'bernoulli']"
    
    rel_set = json.load(open(os.path.join(og_path, config.dataset_path, "rel2id.json"), "r")).keys()
    ent_set = json.load(open(os.path.join(og_path, config.dataset_path, "ent2id.json"), "r")).keys()
    
    # ent2id = {ent.replace(" ", "").lower(): i for i, ent in enumerate(ent_set)}  # 和 matrix 一致
    # ent2ptid = ent_tokenizer.batch_encode_plus(list(ent_set), add_special_tokens=False, return_tensors='pt')
    
    # train_data = read_and_load_data(og_path, config, "train", encoder_tokenizer,
    #                                 ent_tokenizer, rel_tokenizer)[:50]
    # dev_data = read_and_load_data(og_path, config, "train", encoder_tokenizer,
    #                               ent_tokenizer, rel_tokenizer)[:10]
    
    train_data = read_and_load_data(og_path, config, "train", encoder_tokenizer,
                                    ent_tokenizer, rel_tokenizer)
    dev_data = read_and_load_data(og_path, config, "valid", encoder_tokenizer,
                                  ent_tokenizer, rel_tokenizer)
    test_data = read_and_load_data(og_path, config, "test", encoder_tokenizer,
                                   ent_tokenizer, rel_tokenizer)
    
    train_dataset = MyDataset(train_data, config, encoder_tokenizer, ent_tokenizer, rel_tokenizer, mode="train")
    dev_dataset = MyDataset(dev_data, config, encoder_tokenizer, ent_tokenizer, rel_tokenizer, mode="dev")
    test_dataset = MyDataset(test_data, config, encoder_tokenizer, ent_tokenizer, rel_tokenizer, mode="test")
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=6 if device == 'cuda' else 0,
                                  drop_last=False,
                                  collate_fn=train_dataset.generate_batch,
                                  )
    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=6 if device == 'cuda' else 0,
                                drop_last=False,
                                # collate_fn=dev_dataset.generate_batch,
                                collate_fn=dev_dataset.generate_inference_batch_of_inputs
                                )
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=False,
                                 num_workers=6 if device == 'cuda' else 0,
                                 drop_last=False,
                                 collate_fn=test_dataset.generate_inference_batch_of_inputs,
                                 )
    
    with open_dict(config):
        config.train_batch_num = len(train_dataloader)
        config.train_ins_num = len(train_data)
        config.dev_ins_num = len(dev_data)
        config.test_ins_num = len(test_data)
    
    # TODO: lightning module
    logger.info(
        f"batch_size is {config['batch_size']} , "
        f"train batch num: {len(train_dataloader)}, "
        f"valid batch num: {len(dev_dataloader)}, "
        f"test batch num: {len(test_dataloader)}")
    
    metric = Metric(ent_set, rel_set, bad_case_output_path=config.bad_case_output_path)
    base_model = ScFreeModel(config, encoder, ent_decoder, rel_decoder)
    model = LitModel(base_model, metric, ent_tokenizer, rel_tokenizer, config)
    
    checkpoint_callback = ModelCheckpoint(monitor="eval_epoch_relation_f1", mode='max', save_top_k=1,
                                          save_weights_only=True,
                                          # dirpath  = 'checkpoints'
                                          )
    # earlystop_callback = EarlyStopping(monitor="eval_epoch_relation_f1", patience=100, verbose=False, mode="max")
    
    trainer = pl.Trainer(callbacks=[checkpoint_callback,
                                    # earlystop_callback
                                    ],
                         max_epochs=config["epochs"],
                         check_val_every_n_epoch=1,
                         accelerator="auto",
                         num_sanity_val_steps=0,  # for the bug: https://github.com/mindslab-ai/faceshifter/issues/5
                         logger=False,
                         enable_progress_bar=False,
                         # 禁用 tqdm https://stackoverflow.com/questions/59455268/how-to-disable-progress-bar-in-pytorch-lightning
                         deterministic=True,  # 确保复现
    
                         )
    
    logging.info("Get pytorch_lightning trainer")
    logging.info("Span identification task will be trained from epoch 0")
    logging.info(
        f"Entity and Relation label generation task will be trained from epoch {config.complete_train_begin_epoch} "
        f"and evaluated from epoch {config.complete_eval_begin_epoch}")
    
    trainer.fit(model,
                train_dataloaders=train_dataloader,
                val_dataloaders=dev_dataloader,
                )
    
    # automatically auto-loads the best weights from the previous run
    trainer.test(dataloaders=test_dataloader)


if __name__ == '__main__':
    main()
