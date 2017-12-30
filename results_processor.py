import copy
import numpy as np
from sklearn.metrics import roc_curve, auc, average_precision_score
from configuration import printt


class ResultsProcessor:
    def __init__(self):
        self.test_batch_size = None
        self._predictions = {"train": None, "test": None}
        self._losses = {"train": None, "test": None}
        self._labels = {"train": None, "test": None}

    def process_results(self, exp_specs, data, model, name):
        """ processes each result in the results object based on its type and returns stuff if specified in exp_specs"""
        printt("Results for {}".format(name))
        self.test_batch_size = exp_specs["test_batch_size"]
        metrics = ["loss_train", "loss_test", "roc_train", "roc_test", "auprc_train", "auprc_test"]
        _headers = []
        _results = []
        for metric in metrics:
            process_fn = getattr(self, metric)
            headers, results = process_fn(data, model, name)
            _headers += headers
            _results += results
        return _headers, _results

    """ results processing functions """
    def loss_train(self, data, model, name):
        return self.loss(data, model, "train", name + "_train")

    def loss_test(self, data, model, name):
        return self.loss(data, model, "test", name + "_test")

    def loss(self, data, model, tt, name):
        _, losses = self.get_predictions_loss(data, model, tt)
        loss = np.sum(losses)
        printt("{} total loss: {:0.3f}".format(name, loss))
        ave_loss = np.mean(losses)
        printt("{} average loss per protein: {:0.3f}".format(name, ave_loss))

        return ["loss_" + tt, "ave_loss_" + tt], [loss, ave_loss]

    def roc_train(self, data, model, name):
        return self.roc(data, model, "train", name + "_train")

    def roc_test(self, data, model, name):
        return self.roc(data, model, "test", name + "_test")

    def roc(self, data, model, tt, name):
        scores = self.get_predictions_loss(data, model, tt)[0]
        labels = [prot["label"][:, 2] for prot in data[tt]]
        fprs = []
        tprs = []
        roc_aucs = []
        for s, l in zip(scores, labels):
            fpr, tpr, _ = roc_curve(l, s)
            roc_auc = auc(fpr, tpr)
            fprs.append(fpr)
            tprs.append(tpr)
            roc_aucs.append(roc_auc)
        auc_prot_med = np.median(roc_aucs)
        auc_prot_ave = np.mean(roc_aucs)
        printt("{} average protein auc: {:0.3f}".format(name, auc_prot_ave))
        printt("{} median protein auc: {:0.3f}".format(name, auc_prot_med))
        return ["auc_prot_ave_" + tt, "auc_prot_med_" + tt], [auc_prot_ave, auc_prot_med]

    def auprc_train(self, data, model, name):
        return self.auprc(data, model, "train", name + "_train")

    def auprc_test(self, data, model, name):
        return self.auprc(data, model, "test", name + "_test")

    def auprc(self, data, model, tt, name):
        scores = self.get_predictions_loss(data, model, tt)[0]
        labels = [prot["label"][:, 2] for prot in data[tt]]
        close_count = 0
        auprcs = []
        for preds, lbls in zip(scores, labels):
            if np.allclose(preds[:, 0], np.zeros_like(preds[:, 0]) + np.mean(preds[:, 0])):
                close_count += 1
            auprcs.append(average_precision_score(lbls, preds))
        if close_count > 0:
            printt("For {} proteins, all predicted scores are close to each other, auprc may be based on improper sorting".format(close_count))
        med_auprc = np.median(auprcs)
        printt("{} median auprc: {:0.3f}".format(name, med_auprc))
        return ["auprc_med_" + tt], [med_auprc]

    def get_predictions_loss(self, data, model, tt):
        test_batch_size = self.test_batch_size
        #  get predictions/loss from model
        if not self._predictions[tt]:
            # get results for each protein in data[tt]
            results = [self.predict_prot(model, prot, test_batch_size) for prot in data[tt]]
            self._predictions[tt] = [prot["label"] for prot in results]
            self._losses[tt] = [prot["loss"] for prot in results]
        return self._predictions[tt], self._losses[tt]

    def predict_prot(self, model, prot_data, batch_size):
        # batches predictions to fit on gpu
        temp = copy.deepcopy(prot_data)
        results = []
        preds = {}
        batches = np.arange(0, prot_data["label"].shape[0], batch_size)[1:]
        # get results for each minbatch in prot_data["label"]
        for batch in np.array_split(prot_data["label"], batches):
            temp["label"] = batch
            results.append(model.predict(temp))
        preds["label"] = np.vstack([result["label"] for result in results])
        preds["loss"] = np.sum([result["loss"] for result in results])
        return preds

    def get_labels(self, data, model, tt):
        if self._labels[tt] is None:
            self._labels[tt] = [model.get_labels(prot) for prot in data[tt]]
        return self._labels[tt]

    def reset(self):
        for set in self._predictions.keys():
            self._predictions[set] = None
        for set in self._labels.keys():
            self._labels[set] = None
        for set in self._losses.keys():
            self._losses[set] = None
