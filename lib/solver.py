'''
File Created: Monday, 25th November 2019 1:35:30 pm
Author: Dave Zhenyu Chen (zhenyu.chen@tum.de)
'''

import os
import sys
import time
import torch
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR, MultiStepLR

from lib.config import CONF
from lib.loss_helper import get_loss
from lib.eval_helper import eval_cap
from utils.eta import decode_eta
from lib.pointnet2.pytorch_utils import BNMomentumScheduler

ITER_REPORT_TEMPLATE = """
-------------------------------iter: [{epoch_id}: {iter_id}/{total_iter}]-------------------------------
[loss] train_loss: {train_loss}
[loss] train_cap_loss: {train_cap_loss}
[loss] train_kd_loss: {train_kd_loss}
[sco.] train_cap_acc: {train_cap_acc}
[info] mean_fetch_time: {mean_fetch_time}s
[info] mean_forward_time: {mean_forward_time}s
[info] mean_backward_time: {mean_backward_time}s
[info] mean_eval_time: {mean_eval_time}s
[info] mean_iter_time: {mean_iter_time}s
[info] ETA: {eta_h}h {eta_m}m {eta_s}s
"""

EPOCH_REPORT_TEMPLATE = """
---------------------------------summary---------------------------------
[val]   val_bleu-1: {val_bleu_1}
[val]   val_bleu-2: {val_bleu_2}
[val]   val_bleu-3: {val_bleu_3}
[val]   val_bleu-4: {val_bleu_4}
[val]   val_cider: {val_cider}
[val]   val_rouge: {val_rouge}
[val]   val_meteor: {val_meteor}
"""

BEST_REPORT_TEMPLATE = """
--------------------------------------best--------------------------------------
[best] epoch: {epoch}
[best]  bleu-1: {bleu_1}
[best]  bleu-2: {bleu_2}
[best]  bleu-3: {bleu_3}
[best]  bleu-4: {bleu_4}
[best]  cider: {cider}
[best]  rouge: {rouge}
[best]  meteor: {meteor}
"""


class Solver():
    def __init__(self, mode, model, config, dataset, dataloader, optimizer, stamp, val_step=10,
                 use_tf=True, use_rl=False,
                 lr_decay_step=None, lr_decay_rate=None, bn_decay_step=None, bn_decay_rate=None,
                 criterion="meteor"):

        self.epoch = 0  # set in __call__
        self.verbose = 0  # set in __call__

        self.mode = mode
        self.model = model
        self.config = config
        self.dataset = dataset
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.stamp = stamp
        self.val_step = val_step

        self.use_tf = use_tf
        self.use_rl = use_rl

        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.bn_decay_step = bn_decay_step
        self.bn_decay_rate = bn_decay_rate

        self.criterion = criterion

        self.best = {
            "epoch": 0,
            "bleu-1": -float("inf"),
            "bleu-2": -float("inf"),
            "bleu-3": -float("inf"),
            "bleu-4": -float("inf"),
            "cider": -float("inf"),
            "rouge": -float("inf"),
            "meteor": -float("inf"),
            "sum": -float("inf")
        }

        # init log
        # contains all necessary info for all phases
        self.log = {
            "train": {},
            "val": {}
        }

        # tensorboard
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train"), exist_ok=True)
        os.makedirs(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"), exist_ok=True)
        self._log_writer = {
            "train": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/train")),
            "val": SummaryWriter(os.path.join(CONF.PATH.OUTPUT, stamp, "tensorboard/val"))
        }

        # training log
        log_path = os.path.join(CONF.PATH.OUTPUT, stamp, "log.txt")
        self.log_fout = open(log_path, "a")

        # private
        # only for internal access and temporary results
        self._running_log = {}
        self._global_iter_id = 0
        self._total_iter = {}  # set in __call__

        # templates
        self.__iter_report_template = ITER_REPORT_TEMPLATE
        self.__epoch_report_template = EPOCH_REPORT_TEMPLATE
        self.__best_report_template = BEST_REPORT_TEMPLATE

        # lr scheduler
        if lr_decay_step and lr_decay_rate:
            if isinstance(lr_decay_step, list):
                self.lr_scheduler = MultiStepLR(optimizer, lr_decay_step, lr_decay_rate)
            else:
                self.lr_scheduler = StepLR(optimizer, lr_decay_step, lr_decay_rate)
        else:
            self.lr_scheduler = None

        # bn scheduler
        if bn_decay_step and bn_decay_rate:
            it = -1
            start_epoch = 0
            BN_MOMENTUM_INIT = 0.5
            BN_MOMENTUM_MAX = 0.001
            bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * bn_decay_rate ** (int(it / bn_decay_step)), BN_MOMENTUM_MAX)
            self.bn_scheduler = BNMomentumScheduler(model, bn_lambda=bn_lbmd, last_epoch=start_epoch - 1)
        else:
            self.bn_scheduler = None

    def __call__(self, epoch, verbose):
        # setting
        self.epoch = epoch
        self.verbose = verbose
        self._total_iter["train"] = len(self.dataloader["train"]) * epoch
        self._total_iter["val"] = (len(self.dataloader["eval"]["train"]) + len(self.dataloader["eval"]["val"])) \
                                  * (self._total_iter["train"] / self.val_step)

        for epoch_id in range(epoch):
            try:
                self._log("epoch {} starting...".format(epoch_id + 1))

                # feed 
                self._feed(self.dataloader["train"], "train", epoch_id)

                # save model
                self._log("saving last models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

                # update lr scheduler
                if self.lr_scheduler:
                    print("update learning rate --> {}\n".format(self.lr_scheduler.get_last_lr()))
                    self.lr_scheduler.step()

                # update bn scheduler
                if self.bn_scheduler:
                    print("update batch normalization momentum --> {}\n".format(
                        self.bn_scheduler.lmbd(self.bn_scheduler.last_epoch)))
                    self.bn_scheduler.step()

            except KeyboardInterrupt:
                # finish training
                self._finish(epoch_id)
                exit()

        # finish training
        self._finish(epoch_id)

    def _log(self, info_str):
        self.log_fout.write(info_str + "\n")
        self.log_fout.flush()
        print(info_str)

    def _reset_log(self, phase):
        if phase == "train":
            self.log[phase] = {
                # info
                "forward": [],
                "backward": [],
                "eval": [],
                "fetch": [],
                "iter_time": [],
                # loss (float, not torch.cuda.FloatTensor)
                "loss": [],
                "cap_loss": [],
                'kd_loss': [],
                # scores (float, not torch.cuda.FloatTensor)
                "cap_acc": [],
                # for eval
                "bleu-1": [],
                "bleu-2": [],
                "bleu-3": [],
                "bleu-4": [],
                "cider": [],
                "rouge": [],
                "meteor": []
            }
        else:
            self.log[phase] = {
                "bleu-1": [],
                "bleu-2": [],
                "bleu-3": [],
                "bleu-4": [],
                "cider": [],
                "rouge": [],
                "meteor": []
            }

    def _dump_log(self, phase, is_eval=False):
        if phase == "train" and not is_eval:
            log = {
                "loss": ["loss", "cap_loss", 'kd_loss'],
                "score": ["cap_acc"]
            }
            for key in log:
                for item in log[key]:
                    if self.log[phase][item]:
                        self._log_writer[phase].add_scalar(
                            "{}/{}".format(key, item),
                            np.mean([v for v in self.log[phase][item]]),
                            self._global_iter_id
                        )

        # eval
        if is_eval:
            log = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "cider", "rouge", "meteor"]
            for key in log:
                if self.log[phase][key]:
                    self._log_writer[phase].add_scalar(
                        "eval/{}".format(key),
                        self.log[phase][key],
                        self._global_iter_id
                    )

    def _set_phase(self, phase):
        if phase == "train":
            self.model.train()
        elif phase == "val":
            self.model.eval()
        else:
            raise ValueError("invalid phase")

    def _forward(self, data_dict):
        data_dict = self.model(data_dict, use_tf=self.use_tf)

        return data_dict

    def _backward(self):
        # optimize
        self.optimizer.zero_grad()
        self._running_log["loss"].backward()
        self.optimizer.step()

    def _compute_loss(self, data_dict):
        data_dict = get_loss(data_dict=data_dict, mode=self.mode, use_rl=self.use_rl)

        # store loss
        self._running_log["cap_loss"] = data_dict["cap_loss"]
        self._running_log["kd_loss"] = data_dict["kd_loss"]
        self._running_log["loss"] = data_dict["loss"]

        # store eval
        self._running_log["cap_acc"] = data_dict["cap_acc"].item()

    def _eval(self, phase):
        bleu, cider, rouge, meteor = eval_cap(
            mode=self.mode,
            model=self.model,
            dataset=self.dataset["eval"][phase],
            dataloader=self.dataloader["eval"][phase],
            phase=phase,
            folder=self.stamp,
            use_tf=False,
            max_len=CONF.TRAIN.MAX_DES_LEN
        )

        # dump
        self.log[phase]["bleu-1"] = bleu[0][0]
        self.log[phase]["bleu-2"] = bleu[0][1]
        self.log[phase]["bleu-3"] = bleu[0][2]
        self.log[phase]["bleu-4"] = bleu[0][3]
        self.log[phase]["cider"] = cider[0]
        self.log[phase]["rouge"] = rouge[0]
        self.log[phase]["meteor"] = meteor[0]

    def _feed(self, dataloader, phase, epoch_id, is_eval=False):
        # switch mode
        if is_eval:
            self._set_phase("val")
        else:
            self._set_phase(phase)

        # re-init log
        self._reset_log(phase)

        # enter mode
        if not is_eval:
            for data_dict in dataloader:
                # move to cuda
                for key in data_dict:
                    data_dict[key] = data_dict[key].cuda()

                # initialize the running loss
                self._running_log = {
                    # loss
                    "loss": 0,
                    "cap_loss": 0,
                    'kd_loss': 0,
                    # acc
                    "cap_acc": 0,
                    "cap_acc": 0,
                }

                # load
                self.log[phase]["fetch"].append(data_dict["load_time"].sum().item())

                # forward
                start = time.time()
                data_dict = self._forward(data_dict)
                self._compute_loss(data_dict)
                self.log[phase]["forward"].append(time.time() - start)

                self._compute_loss(data_dict)
                self.log[phase]["forward"].append(time.time() - start)

                # backward
                start = time.time()
                self._backward()
                self.log[phase]["backward"].append(time.time() - start)

                # eval
                start = time.time()
                # self._eval(data_dict)
                self.log[phase]["eval"].append(time.time() - start)

                # record log
                self.log[phase]["loss"].append(self._running_log["loss"].item())
                self.log[phase]["cap_loss"].append(self._running_log["cap_loss"].item())
                self.log[phase]["kd_loss"].append(self._running_log["kd_loss"].item())
                self.log[phase]["cap_acc"].append(self._running_log["cap_acc"])

                # report
                if phase == "train":
                    iter_time = self.log[phase]["fetch"][-1]
                    iter_time += self.log[phase]["forward"][-1]
                    iter_time += self.log[phase]["backward"][-1]
                    iter_time += self.log[phase]["eval"][-1]
                    self.log[phase]["iter_time"].append(iter_time)
                    if (self._global_iter_id + 1) % self.verbose == 0:
                        self._train_report(epoch_id)
                        self._dump_log("train")

                    # evaluation
                    # if self._global_iter_id % self.val_step == 0:
                    if (self._global_iter_id + 1) % self.val_step == 0:
                        # eval on train
                        # print("evaluating on train...")
                        # self._feed(self.dataloader["eval"]["train"], "train", epoch_id, True)
                        # self._dump_log("train", True)

                        # val
                        print("evaluating on val...")
                        self._feed(self.dataloader["eval"]["val"], "val", epoch_id, True)
                        self._dump_log("val", True)

                        self._set_phase("train")
                        self._epoch_report(epoch_id)

                    # dump log
                    self._global_iter_id += 1
        else:
            self._eval(phase)

            cur_criterion = self.criterion
            if cur_criterion == "sum":
                metrics = ["bleu-1", "bleu-2", "bleu-3", "bleu-4", "cider", "rouge", "meteor"]
                cur_best = np.sum([np.mean(self.log[phase][m]) for m in metrics])
            else:
                cur_best = np.mean(self.log[phase][cur_criterion])

            if phase == "val" and cur_best > self.best[cur_criterion]:
                self._log("best {} achieved: {}".format(cur_criterion, cur_best))

                self.best["epoch"] = epoch_id + 1
                self.best["bleu-1"] = self.log[phase]["bleu-1"]
                self.best["bleu-2"] = self.log[phase]["bleu-2"]
                self.best["bleu-3"] = self.log[phase]["bleu-3"]
                self.best["bleu-4"] = self.log[phase]["bleu-4"]
                self.best["cider"] = self.log[phase]["cider"]
                self.best["rouge"] = self.log[phase]["rouge"]
                self.best["meteor"] = self.log[phase]["meteor"]
                self.best["sum"] = cur_best

                # save model
                self._log("saving best models...\n")
                model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
                torch.save(self.model.state_dict(), os.path.join(model_root, "model.pth"))

    def _finish(self, epoch_id):
        # print best
        self._best_report()

        # save check point
        self._log("saving checkpoint...\n")
        save_dict = {
            "epoch": epoch_id,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        checkpoint_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(save_dict, os.path.join(checkpoint_root, "checkpoint.tar"))

        # save model
        self._log("saving last models...\n")
        model_root = os.path.join(CONF.PATH.OUTPUT, self.stamp)
        torch.save(self.model.state_dict(), os.path.join(model_root, "model_last.pth"))

        # export
        for phase in ["train", "val"]:
            self._log_writer[phase].export_scalars_to_json(
                os.path.join(CONF.PATH.OUTPUT, self.stamp, "tensorboard/{}".format(phase), "all_scalars.json"))

    def _train_report(self, epoch_id):
        # compute ETA
        fetch_time = self.log["train"]["fetch"]
        forward_time = self.log["train"]["forward"]
        backward_time = self.log["train"]["backward"]
        eval_time = self.log["train"]["eval"]
        iter_time = self.log["train"]["iter_time"]

        mean_train_time = np.mean(iter_time)
        mean_est_val_time = np.mean([fetch + forward for fetch, forward in zip(fetch_time, forward_time)])

        num_train_iter_left = self._total_iter["train"] - self._global_iter_id - 1
        eta_sec = num_train_iter_left * mean_train_time

        num_val_times = num_train_iter_left // self.val_step
        eta_sec += len(self.dataloader["eval"]["train"]) * num_val_times * mean_est_val_time
        eta_sec += len(self.dataloader["eval"]["val"]) * num_val_times * mean_est_val_time

        eta = decode_eta(eta_sec)

        # print report
        iter_report = self.__iter_report_template.format(
            epoch_id=epoch_id + 1,
            iter_id=self._global_iter_id + 1,
            total_iter=self._total_iter["train"],
            train_loss=round(np.mean([v for v in self.log["train"]["loss"]]), 5),
            train_cap_loss=round(np.mean([v for v in self.log["train"]["cap_loss"]]), 5),
            train_kd_loss=round(np.mean([v for v in self.log["train"]["kd_loss"]]), 5),
            train_cap_acc=round(np.mean([v for v in self.log["train"]["cap_acc"]]), 5),
            mean_fetch_time=round(np.mean(fetch_time), 5),
            mean_forward_time=round(np.mean(forward_time), 5),
            mean_backward_time=round(np.mean(backward_time), 5),
            mean_eval_time=round(np.mean(eval_time), 5),
            mean_iter_time=round(np.mean(iter_time), 5),
            eta_h=eta["h"],
            eta_m=eta["m"],
            eta_s=eta["s"]
        )
        self._log(iter_report)

    def _epoch_report(self, epoch_id):
        self._log("epoch [{}/{}] done...".format(epoch_id + 1, self.epoch))
        epoch_report = self.__epoch_report_template.format(
            val_bleu_1=round(self.log["val"]["bleu-1"], 5),
            val_bleu_2=round(self.log["val"]["bleu-2"], 5),
            val_bleu_3=round(self.log["val"]["bleu-3"], 5),
            val_bleu_4=round(self.log["val"]["bleu-4"], 5),
            val_cider=round(self.log["val"]["cider"], 5),
            val_rouge=round(self.log["val"]["rouge"], 5),
            val_meteor=round(self.log["val"]["meteor"], 5)
        )
        self._log(epoch_report)

    def _best_report(self):
        self._log("training completed...")
        best_report = self.__best_report_template.format(
            epoch=self.best["epoch"],
            bleu_1=round(self.best["bleu-1"], 5),
            bleu_2=round(self.best["bleu-2"], 5),
            bleu_3=round(self.best["bleu-3"], 5),
            bleu_4=round(self.best["bleu-4"], 5),
            cider=round(self.best["cider"], 5),
            rouge=round(self.best["rouge"], 5),
            meteor=round(self.best["meteor"], 5)
        )
        self._log(best_report)
        with open(os.path.join(CONF.PATH.OUTPUT, self.stamp, "best.txt"), "w") as f:
            f.write(best_report)
