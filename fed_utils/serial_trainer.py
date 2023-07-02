import torch
import copy
from torch.utils.data import DataLoader, Subset

from fedlab.core.client.scale.trainer import SubsetSerialTrainerStateDict
from fedlab.utils import Logger
from fedlab.core.client import SERIAL_TRAINER
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import AverageMeter
from utils import DiffAugment, ParamDiffAug

def train_one_epoch(train_loader, model, optimizer, logger, scheduler=None, scheduler_by_iter=False, max_norm=None, cuda=True):
    model.train()
    train_loss = 0.0
    total = 0.0
    correct = 0.0
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        if cuda:
            imgs = imgs.cuda()
            targets = targets.cuda()
        
        # imgs = DiffAugment(imgs, "color_flip", param=ParamDiffAug())
        
        optimizer.zero_grad()
        
        outputs = model(imgs)
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # clip gradients
        optimizer.step()
        
        if scheduler_by_iter and scheduler:
            scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx + 1) % 20 == 0:
            logger.info(f"Iter {batch_idx + 1}: train acc {correct / total * 100:.2f} train loss {train_loss / (batch_idx + 1):.3f}")
        
    return model, correct / total, train_loss / (batch_idx + 1)


def train_one_epoch_syn(train_loader, model, optimizer, logger, scheduler=None, scheduler_by_iter=False, max_norm=None, cuda=True, sythetic_image=None, sythetic_label=None):
    model.train()
    train_loss = 0.0
    total = 0.0
    correct = 0.0
    for batch_idx, (imgs, targets) in enumerate(train_loader):
        if cuda:
            imgs = imgs.cuda()
            targets = targets.cuda()
        
        sythetic_image = DiffAugment(sythetic_image, "color_crop_cutout_flip_scale_rotate", param=ParamDiffAug())
        
        imgs = torch.cat([imgs, sythetic_image])
        targets = torch.cat([targets, sythetic_label])
        optimizer.zero_grad()
        
        outputs = model(imgs)
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # clip gradients
        optimizer.step()
        
        if scheduler_by_iter and scheduler:
            scheduler.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx + 1) % 20 == 0:
            logger.info(f"Iter {batch_idx + 1}: train acc {correct / total * 100:.2f} train loss {train_loss / (batch_idx + 1):.3f}")
        
    return model, correct / total, train_loss / (batch_idx + 1)

class SerialTrainer(SubsetSerialTrainerStateDict):
    def __init__(self, 
                 model,
                 dataset,
                 data_slices,
                 sythetic_image=None,
                 sythetic_label=None,
                 aggregator=None,
                 cuda=True,
                 logger=Logger(),
                 args=None):
        self._model = model
        self.cuda = cuda
        self.gpu = 0
        self.client_num = len(data_slices)
        self.type = SERIAL_TRAINER  # represent serial trainer
        self.aggregator = aggregator
        self._LOGGER = logger
        self.dataset = dataset
        self.data_slices = data_slices  # [0, client_num)
        self.args = args
        self.sythetic_image = sythetic_image
        self.sythetic_label = sythetic_label
    
    
    def train(self, model_parameters, id_list, aggregate=False, **kwargs):
        """Train local model with different dataset according to client id in ``id_list``.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
            id_list (list[int]): Client id in this training serial.
            aggregate (bool): Whether to perform partial aggregation on this group of clients' local models at the end of each local training round.

        Note:
            Normally, aggregation is performed by server, while we provide :attr:`aggregate` option here to perform
            partial aggregation on current client group. This partial aggregation can reduce the aggregation workload
            of server.

        Returns:
            Serialized model parameters / list of model parameters.
        """
        param_list = []
        self._LOGGER.info(
            "Local training with client id list: {}".format(id_list))
        for idx in id_list:
            data_loader = self._get_dataloader(client_id=idx)
            loss_i, acc_i = self._train_alone(model_parameters=model_parameters,
                              train_loader=data_loader)
            param_list.append(self.model_parameters)
            self._LOGGER.info(
                "Training procedure of client {} [loss {:.2f}, acc {:.2f}]".format(idx, loss_i, acc_i))


        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list, **kwargs)
            return aggregated_parameters
        else:
            return param_list
    
    def _get_dataloader(self, client_id):
        """Return a training dataloader used in :meth:`train` for client with :attr:`id`

        Args:
            client_id (int): :attr:`client_id` of client to generate dataloader

        Note:
            :attr:`client_id` here is not equal to ``client_id`` in global FL setting. It is the index of client in current :class:`SerialTrainer`.

        Returns:
            :class:`DataLoader` for specific client's sub-dataset
        """
        batch_size = self.args["batch_size"]

        train_loader = DataLoader(
            Subset(self.dataset, self.data_slices[client_id]),
            shuffle=True,
            batch_size=batch_size)
        return train_loader
    
    def _train_alone(self, model_parameters, train_loader):
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        epochs = self.args["epochs"]

        SerializationTool.deserialize_state_dict(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        params_to_update = []
        for name, param in self._model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = getattr(torch.optim, self.args["optim"]["name"])(params_to_update, **self.args["optim"]["kwargs"])
        
        for _ in range(epochs):
            if self.sythetic_image is None:
                _, acc, loss = train_one_epoch(
                    train_loader=train_loader,
                    model=self.model,
                    optimizer=optimizer,
                    max_norm=self.args["max_norm"] if "max_norm" in self.args.keys() else None,
                    cuda=self.cuda,
                    logger=self._LOGGER,
                    )
            else:
                _, acc, loss = train_one_epoch_syn(
                    train_loader=train_loader,
                    model=self.model,
                    optimizer=optimizer,
                    max_norm=self.args["max_norm"] if "max_norm" in self.args.keys() else None,
                    cuda=self.cuda,
                    logger=self._LOGGER,
                    sythetic_image=self.sythetic_image,
                    sythetic_label=self.sythetic_label,
                    )
                
        return loss, acc