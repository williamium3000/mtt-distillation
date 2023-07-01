from json import load
import os
import argparse
import random
from copy import deepcopy
from torch import nn
import torch
import time

from fed_utils.aggregator import Aggregators
from fed_utils.aggregator import SerializationTool
from fed_utils.utils import Logger, save_model
from test import evaluate

from fed_utils.serial_trainer import SerialTrainer
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug

parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--test-domain-idx', type=int, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--save_every", type=int, default=10000) # default do not save


def main():
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = Logger(__name__, os.path.join(args.save_path, f"{timestamp}.log"))
        
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model = get_network('ConvNet', channel, num_classes, im_size).to(args.device)
    if args.cuda:
        model = model.cuda()
    
    # get dataset
    test_loader = torch.utils.data.DataLoader(dst_test,
                                            batch_size=64,
                                            drop_last=False,
                                            shuffle=False)
    # FL settings
    num_per_round = cfg["sample_num"]
    aggregator = Aggregators.fedavg_aggregate
    local_model = deepcopy(model)

    trainer = SerialTrainer(model=local_model,
                                dataset=dst_train,
                                data_slices=data_indices,
                                aggregator=None,
                                cuda=args.cuda,
                                logger=logger,
                                args={
                                    "batch_size": cfg["batch_size"],
                                    "epochs": cfg["epochs"],
                                    "max_norm": 10,
                                    "optim":cfg["optim"]
                                })


    # train procedure
    to_select = [i for i in range(4)]
    best_acc = 0.0

    for round in range(cfg["com_round"]):
        model_parameters = SerializationTool.serialize_state_dict(model)
        selection = random.sample(to_select, num_per_round)
        parameters = trainer.train(model_parameters=model_parameters,
                                            id_list=selection,
                                            aggregate=False)
        aggregated_parameters = aggregator(parameters)
        SerializationTool.deserialize_state_dict(model, aggregated_parameters)

        loss, acc = evaluate(model, test_loader)
        acc = acc * 100
        logger.info("Test Round {}, loss: {:.4f}, acc: {:.2f}".format(round + 1, loss, acc))
        if acc > best_acc:
            logger.info("-------------------------------------")
            logger.info(f"best acc {acc:.2f} updated and ckpt saved.")
            logger.info("-------------------------------------")
            if os.path.exists(os.path.join(args.save_path, f"best_{best_acc:.2f}.pth")):
                os.remove(os.path.join(args.save_path, f"best_{best_acc:.2f}.pth"))
            best_acc = acc
            save_model(
                model,
                os.path.join(args.save_path, f"best_{best_acc:.2f}.pth"))
        if args.save_every and (round + 1) % args.save_every == 0:
            save_model(
                model,
                os.path.join(args.save_path, f"epoch_{round + 1}.pth"))
        if cfg["early_stop_acc"] is not None and acc >= cfg["early_stop_acc"]:
            logger.info("-------------------------------------")
            logger.info(f"early stopped at {acc:.2f} > {cfg['early_stop_acc']}")
            logger.info("-------------------------------------")
            break
        
    save_model(
        model,
        os.path.join(args.save_path, "last.pth"))
    
if __name__ == '__main__':
    main()