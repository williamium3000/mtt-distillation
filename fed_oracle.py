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
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

parser.add_argument('--model', type=str, default='ConvNet', help='model')

parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

parser.add_argument('--eval_mode', type=str, default='S',
                    help='eval_mode, check utils.py for more info')

parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
parser.add_argument('--lr_net', type=float, default=0.001)

parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                    help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                    help='whether to use differentiable Siamese augmentation.')

parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                    help='differentiable Siamese augmentation strategy')

parser.add_argument('--data_path', type=str, default='data', help='dataset path')
parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

parser.add_argument('--texture', action='store_true', help="will distill textures instead")
parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
parser.add_argument('--syn_path', type=str)

parser.add_argument('--save-path', type=str, required=True)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--save_every", type=int, default=10000) # default do not save


def main():
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger = Logger(__name__, os.path.join(args.save_path, f"{timestamp}.log"))
        
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv, data_indices = \
        get_dataset(args.dataset, args.data_path, args.batch_real, subset=None, test_subset=args.subset, args=args)
    model = get_network('ConvNet', channel, num_classes, im_size).to(args.device)
    if args.cuda:
        model = model.cuda()
    
    # get dataset
    test_loader = torch.utils.data.DataLoader(dst_test,
                                            batch_size=64,
                                            drop_last=False,
                                            shuffle=False)
    # FL settings
    num_per_round = 3
    aggregator = Aggregators.fedavg_aggregate
    local_model = deepcopy(model)

    trainer = SerialTrainer(model=local_model,
                                dataset=dst_train,
                                data_slices=data_indices,
                                aggregator=None,
                                cuda=args.cuda,
                                logger=logger,
                                args={
                                    "batch_size": 32,
                                    "epochs": 1,
                                    "max_norm": 10,
                                    "optim": dict(
                                        name="SGD",
                                        kwargs=dict(
                                            lr=0.01,
                                            momentum=0.9,
                                            weight_decay=1e-4
                                        )
                                    )
                                })


    # train procedure
    to_select = [i for i in range(4)]
    best_acc = 0.0

    for round in range(100):
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
        
    save_model(
        model,
        os.path.join(args.save_path, "last.pth"))
    
if __name__ == '__main__':
    main()