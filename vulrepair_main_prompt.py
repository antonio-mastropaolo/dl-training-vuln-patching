# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          T5ForConditionalGeneration, RobertaTokenizer, T5Config)
from tqdm import tqdm
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import datasets
from sklearn.model_selection import train_test_split

from openprompt import PromptDataLoader, PromptForGeneration
from openprompt.plms import T5TokenizerWrapper
from openprompt.prompts import PrefixTuningTemplate, SoftTemplate
from openprompt.data_utils import InputExample


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


cpu_cont = 16
logger = logging.getLogger(__name__)

# Setup CUDA, GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

soft_prompts_template = {
    1: '<CWE_NAME> {"placeholder":"text_b"} </CWE_NAME> Generate a patch for this vulnerable code {"placeholder":"text_a"} as follows: {"mask"} ',
    2: 'Generate a patch for this vulnerable code {"placeholder":"text_a"} as follows: {"mask"} ',
    3: 'This vulnerable code {"placeholder":"text_a"} is fixed by: {"mask"} ',
    4: 'Patch the following vulnerable code {"placeholder":"text_a"} with: {"mask"} ',
    5: 'This text <CWE_NAME> {"placeholder":"text_b"} </CWE_NAME> describes the vulnerable code {"placeholder":"text_a"} fixed by: {"mask"} '
}


hard_prompts_template = {
    1: '<CWE_NAME> {} </CWE_NAME> Generate a patch for this vulnerable code {} as follows: ',
    2: 'Generate a patch for this vulnerable code {} as follows: ',
    3: 'This vulnerable code {} is fixed by: ',
    4: 'Patch the following vulnerable code {} with: ',
    5: 'This text <CWE_NAME> {} </CWE_NAME> describes the vulnerable code {} fixed by: '

}


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_ids,
                 label,
                 decoder_input_ids):
        self.input_ids = input_ids
        self.label = label
        self.decoder_input_ids = decoder_input_ids


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, data, isTrain=False):

        self.examples = []
        for i in tqdm(range(len(data))):
            source = data[i].source
            label = data[i].target
            self.examples.append(convert_examples_to_features(
                source, label, tokenizer, args))

        if isTrain:
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_ids: {}".format(
                    ' '.join(map(str, example.input_ids))))
                logger.info("decoder_input_ids: {}".format(
                    ' '.join(map(str, example.decoder_input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i].input_ids, self.examples[i].input_ids.ne(0), self.examples[i].label, self.examples[i].decoder_input_ids

################################ HELPER FUNCTIONS ################################


def convert_examples_to_features(source, label, tokenizer, args):
    # encode - subword tokenize
    source_ids = tokenizer.encode(
        source, truncation=True, max_length=args.encoder_block_size, padding='max_length', return_tensors='pt')
    decoder_input_ids = tokenizer.encode(
        label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    label = tokenizer.encode(
        label, truncation=True, max_length=args.decoder_block_size, padding='max_length', return_tensors='pt')
    return InputFeatures(source_ids, label, decoder_input_ids)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


# Hard prompt
def read_examples(args, filename):
    """Read examples from filename."""

    examples = []
    df = pd.read_csv(filename)
    sources = list(df['source'])
    targets = list(df['target'])
    cwe_descriptions = list(df['cwe_name'])

    for (idxInstance, (inp, target, cwe)) in enumerate(zip(sources, targets, cwe_descriptions)):
        if args.prompt_number == 1 or args.prompt_number == 5:
            examples.append(
                Example(
                    idx=idxInstance,
                    source=hard_prompts_template[args.prompt_number].format(
                        cwe, inp),
                    target=target
                )
            )
        else:
            examples.append(
                Example(
                    idx=idxInstance,
                    source=hard_prompts_template[args.prompt_number].format(
                        inp),
                    target=target
                )
            )

    return examples


# Soft prompt
def read_prompt_examples(args, filename):
    """Read examples from filename."""
    examples = []
    df = pd.read_csv(filename)
    sources = list(df['source'])
    targets = list(df['target'])
    cwe_descriptions = list(df['cwe_name'])

    for (idxInstance, (inp, target, cwe)) in enumerate(zip(sources, targets, cwe_descriptions)):

        if args.prompt_number == 1 or args.prompt_number == 5:
            examples.append(
                InputExample(
                    guid=idxInstance,
                    text_a=inp,
                    text_b=cwe,
                    tgt_text=target
                )
            )

        else:
            examples.append(
                InputExample(
                    guid=idxInstance,
                    text_a=inp,
                    tgt_text=target
                )
            )

    return examples


######################################################################


def train(args, model, train_examples, eval_data, tokenizer, promptTemplate=None, tokenizer_wrapper_class=None):
    """ Train the model """

    if args.soft_prompt:
        train_dataloader = PromptDataLoader(
            dataset=train_examples,
            tokenizer=tokenizer,
            template=promptTemplate,
            tokenizer_wrapper_class=tokenizer_wrapper_class,
            max_seq_length=args.encoder_block_size,
            decoder_max_length=args.decoder_block_size,
            shuffle=True,
            teacher_forcing=True,
            predict_eos_token=True,
            batch_size=args.train_batch_size
        )

        eval_dataset = read_prompt_examples(args, eval_data)

    else:
        train_dataset = TextDataset(
            tokenizer, args, train_examples, isTrain=True)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

        eval_examples = read_examples(args, eval_data)
        eval_dataset = TextDataset(tokenizer, args, eval_examples)

    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate model per epoch
    args.save_steps = len(train_dataloader) * 1

    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",
                args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_loss = 100

    writer_path = "tb_cwe_prompt/codet5_training_loss"
    writer = SummaryWriter(writer_path)

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):

            if args.soft_prompt:
                batch = batch.to(device)

                loss = model(batch)

                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            else:
                (input_ids, attention_mask, labels, decoder_input_ids) = [
                    x.squeeze(1).to(args.device) for x in batch]
                model.train()
                # the forward function automatically creates the correct decoder_input_ids
                loss = model(input_ids=input_ids,
                             attention_mask=attention_mask, labels=labels).loss
                if args.n_gpu > 1:
                    loss = loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm)
            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 5)
                if global_step % args.save_steps == 0:
                    # placeholder of evaluation
                    eval_loss = evaluate(args, model, tokenizer, eval_dataset,
                                         promptTemplate, tokenizer_wrapper_class, eval_when_training=True)
                    # Save model checkpoint
                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        logger.info("  "+"*"*20)
                        logger.info("  Best Loss:%s", round(best_loss, 5))
                        logger.info("  "+"*"*20)
                        checkpoint_prefix = 'checkpoint-best-loss'
                        output_dir = os.path.join(
                            args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info(
                            "Saving model checkpoint to %s", output_dir)


def clean_tokens(tokens):
    tokens = tokens.replace("<pad>", "")
    tokens = tokens.replace("<s>", "")
    tokens = tokens.replace("</s>", "")
    tokens = tokens.strip("\n")
    tokens = tokens.strip()
    return tokens


def evaluate(args, model, tokenizer, eval_examples, promptTemplate=None, WrapperClass=None, eval_when_training=False):
    # build dataloader

    if args.soft_prompt:
        print('\nIn SOFT-PROMPT EVAL...')
        eval_dataloader = PromptDataLoader(
            dataset=eval_examples,
            tokenizer=tokenizer,
            template=promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=args.encoder_block_size,
            decoder_max_length=args.decoder_block_size,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=True,
            batch_size=args.eval_batch_size
        )

    else:
        eval_sampler = SequentialSampler(eval_examples)
        eval_dataloader = DataLoader(
            eval_examples, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model.eval()

    eval_loss, num = 0, 0
    bar = tqdm(eval_dataloader, total=len(eval_dataloader))

    if args.soft_prompt:

        for batch in bar:
            batch = batch.to(device)

            with torch.no_grad():
                loss = model(batch)

            if args.n_gpu > 1:
                loss = loss.mean()

            eval_loss += loss.sum().item()

            num += 1

        eval_loss = round(eval_loss/num, 5)

    else:
        for batch in bar:
            (input_ids, attention_mask, labels, decoder_input_ids) = [
                x.squeeze(1).to(args.device) for x in batch]
            loss = model(input_ids=input_ids,
                         attention_mask=attention_mask, labels=labels).loss
            if args.n_gpu > 1:
                loss = loss.mean()
            eval_loss += loss.item()
            num += 1
        eval_loss = round(eval_loss/num, 5)

    model.train()
    logger.info("***** Eval results *****")
    logger.info(f"Evaluation Loss: {str(eval_loss)}")
    return eval_loss


def test(args, model, tokenizer, promptTemplate=None, wrapperClass=None, best_threshold=0.5):

    print(wrapperClass)
    print(promptTemplate)

    if args.soft_prompt:

        test_examples = read_prompt_examples(
            args, 'data_cleaned/with-cwe/test_cwe_name.csv')
        test_dataloader = PromptDataLoader(
            dataset=test_examples,
            tokenizer=tokenizer,
            template=promptTemplate,
            tokenizer_wrapper_class=wrapperClass,
            max_seq_length=args.encoder_block_size,
            decoder_max_length=args.decoder_block_size,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=True,
            batch_size=args.eval_batch_size
        )

        generation_arguments = {
            "max_length": args.decoder_block_size,
            "do_sample": False,
            "num_beams": args.num_beams,
            "num_return_sequences": args.num_beams
        }

    else:
        test_examples = read_examples(
            args, 'data_cleaned/with-cwe/test_cwe_name.csv')
        test_dataset = TextDataset(
            tokenizer, args, test_examples, isTrain=False)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(
            test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Test!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    model.eval()

    accuracy = []
    raw_predictions = []
    groundtruth_sentence = []
    generated_texts = []

    counter = 0

    if args.soft_prompt:

        guids = []

        for batch in tqdm(test_dataloader, total=len(test_dataloader)):

            batch = batch.to(device)
            with torch.no_grad():
                _, beam_outputs = model.generate(batch, **generation_arguments)

            generated_texts.extend(beam_outputs)
            groundtruth_sentence.extend(batch['tgt_text'])
            guids.extend(batch['guid'])

        for i in range(0, len(generated_texts), args.num_beams):

            predictions = generated_texts[i:i+args.num_beams]
            cleanTarget = clean_tokens(groundtruth_sentence[counter])

            correct_pred = False

            for pred in predictions:

                cleanPred = clean_tokens(pred)

                if cleanTarget == cleanPred:
                    raw_predictions.append(cleanPred)
                    accuracy.append(1)
                    correct_pred = True
                    break

            if not correct_pred:
                raw_predictions.append(predictions[0])
                accuracy.append(0)

            counter += 1

    else:
        for batch in tqdm(test_dataloader, total=len(test_dataloader)):
            correct_pred = False
            (input_ids, attention_mask, labels, decoder_input_ids) = [
                x.squeeze(1).to(args.device) for x in batch]

            with torch.no_grad():
                beam_outputs = model.generate(input_ids=input_ids,
                                              attention_mask=attention_mask,
                                              do_sample=False,  # disable sampling to test if batching affects output
                                              num_beams=args.num_beams,
                                              num_return_sequences=args.num_beams,
                                              max_length=args.decoder_block_size)

            beam_outputs = beam_outputs.detach().cpu().tolist()
            decoder_input_ids = decoder_input_ids.detach().cpu().tolist()

            generated_texts.extend(beam_outputs)
            groundtruth_sentence.extend(decoder_input_ids)

        print(len(groundtruth_sentence))
        for i in range(0, len(generated_texts), args.num_beams):

            predictions = generated_texts[i:i+args.num_beams]
            ground_truth = clean_tokens(tokenizer.decode(groundtruth_sentence[counter], skip_special_tokens=False))

            correct_pred = False

            for single_output in predictions:

                # pred
                prediction = tokenizer.decode(
                    single_output, skip_special_tokens=False)
                prediction = clean_tokens(prediction)

                if prediction == ground_truth:
                    correct_prediction = prediction
                    correct_pred = True
                    break

            if correct_pred:
                raw_predictions.append(correct_prediction)
                groundtruth_sentence[counter] = ground_truth
                accuracy.append(1)

            else:
                # if not correct, use the first output in the beam as the raw prediction
                raw_pred = tokenizer.decode(
                    predictions[0], skip_special_tokens=False)
                raw_pred = clean_tokens(raw_pred)
                raw_predictions.append(raw_pred)
                groundtruth_sentence[counter] = ground_truth
                accuracy.append(0)

            counter += 1

    # calculate accuracy
    test_result = round(sum(accuracy) / len(accuracy), 4)
    logger.info("***** Test results *****")
    logger.info(f"Test Accuracy: {str(test_result)}")

    # write prediction to file
    df = pd.DataFrame({"target": [], "raw_predictions": [],
                       "correctly_predicted": []})
    df['target'] = groundtruth_sentence
    df["raw_predictions"] = raw_predictions
    df["correctly_predicted"] = accuracy
    if args.soft_prompt:
        df.to_csv("./predictions/soft-prompt-{}/predictions_beam_{}.csv".format(args.prompt_number, args.num_beams))
    else:
        df.to_csv("./predictions/hard-prompt-{}/predictions_beam_{}.csv".format(args.prompt_number, args.num_beams))


def main():
    parser = argparse.ArgumentParser()
    # Params
    parser.add_argument("--output_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="t5", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--encoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                        "The training dataset will be truncated in block of this size for training."
                        "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--decoder_block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                        "The training dataset will be truncated in block of this size for training."
                        "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--num_beams", default=50, type=int,
                        help="Beam size to use when decoding.")
    parser.add_argument("--model_name", default="model.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--checkpoint_model_name", default="non_domain_model.bin", type=str,
                        help="Checkpoint model name.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true',
                                            help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=1,
                        help="training epochs")

    parser.add_argument("--soft_prompt", action='store_true',
                        help="Whether to experiment with soft prompting")

    parser.add_argument("--prompt_number", type=int, default=1,
                        help="Set prompt number")

    args = parser.parse_args()

    args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu,)
    # Set seed
    set_seed(args)

    # read model --------------------------------------------------------------
    model_config = T5Config.from_pretrained("Salesforce/codet5-base")
    model = T5ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, 
            config=model_config)
    
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(["<S2SV_StartBug>", "<S2SV_EndBug>", "<S2SV_blank>",
                          "<S2SV_ModStart>", "<S2SV_ModEnd>"]) 
    
    model.resize_token_embeddings(len(tokenizer))

    if args.soft_prompt:
        
        ## Uncomment the following when running self-supervised pre-training + fine-tuning on bug-fixing.
        ## We start fine-tuning the model for vulnerability patching once loaded the proper model

        #if args.do_train:
        #    model.load_state_dict(torch.load("models/codet5-bf/checkpoint-best-loss/model.bin", map_location=device))

        WrapperClass = T5TokenizerWrapper
        promptTemplate = SoftTemplate(model=model, tokenizer=tokenizer, text=soft_prompts_template[args.prompt_number], initialize_from_vocab=True,
                                      num_tokens=args.encoder_block_size)

        print(promptTemplate)

        # get model
        promptModel = PromptForGeneration(plm=model, template=promptTemplate, freeze_plm=False, tokenizer=tokenizer,
                                          plm_eval_mode=False)
        promptModel.to(device)

        if args.do_train:
            train_examples = read_prompt_examples(
                args, 'data_cleaned/with-cwe/train_cwe_name.csv')
            wrapped_example = promptTemplate.wrap_one_example(
                train_examples[0])
            # take an example
            logger.info(wrapped_example)
            train(args, promptModel, train_examples, 'data_cleaned/with-cwe/eval_cwe_name.csv',
                  tokenizer, promptTemplate, WrapperClass)

    else:
        if args.do_train:

            ## Uncomment the following when running self-supervised pre-training + fine-tuning on bug-fixing.
            ## We start fine-tuning the model for vulnerability patching once loaded the proper model

            #model.load_state_dict(torch.load("models/codet5-bf/checkpoint-best-loss/model.bin", map_location=device))

            train_examples = read_examples(
                args, 'data_cleaned/with-cwe/train_cwe_name.csv')
            train(args, model, train_examples,
                  'data_cleaned/with-cwe/eval_cwe_name.csv', tokenizer)

    # Evaluation
    results = {}

    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-loss/{args.model_name}'
        output_dir = os.path.join(
            args.output_dir, '{}'.format(checkpoint_prefix))

        if args.soft_prompt:
            promptModel.load_state_dict(torch.load(output_dir, map_location=device))
            promptModel.to(device)
            test(args, promptModel, tokenizer, promptTemplate, WrapperClass)

        else:
            model.load_state_dict(torch.load(output_dir, map_location=device))
            model.to(device)
            test(args, model, tokenizer)


    return results


if __name__ == "__main__":
    main()
