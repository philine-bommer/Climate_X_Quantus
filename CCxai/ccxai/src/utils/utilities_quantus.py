from typing import Dict, Any
import numpy as np
import copy
import pdb
import pandas as pd
from quantus.helpers.utils import *
from quantus.helpers.normalise_func import *
from quantus.helpers.normalise_func import normalise_by_negative
import tensorflow as tf
import keras
import innvestigate
import innvestigate.utils as iutils

import noisegrad_tf.srctf.noisegrad_tf as ng
import noisegrad_tf.srctf.explainers_tf as xg
import cphxai.src.utils.utilities_modelaware as xai_aw


def generate_tf_innvestigation(
    model, inputs, targets, **kwargs
) -> np.ndarray:
    """
    Generate explanation for a tf model with innvestigate and NoiseGrad and FusionGrad
    tensorflow implementation
    :param model: trained model instance keras.model
    :param inputs: input sample
    :param targets: vectore of according true class labels
    :param kwargs: 'num_classes' - number of classes
                   'nr_samples' - number of iterations of the evaluation metric
                   'explanation' - true explanation
    :return:
    """

    method = kwargs.get("method", "random")
    og_shape = inputs.shape

    inputs = inputs.reshape(-1, *model.input_shape[1:])

    explanation = np.zeros_like(inputs)


    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

    if not method:
        raise KeyError(
            "Specify a XAI method that already has been implemented."
        )

    if method[0] == 'NoiseGrad' or method[0] == 'FusionGrad':
        if kwargs.get('num_classes',0)>0:
            method[1]['num_classes'] = kwargs['num_classes']
            method[1]['cl'] = targets
            if method[0] == 'NoiseGrad':

                analyzer = ng.NoiseGrad(model=model, std=method[1]['std'], n=20)
            else:
                analyzer = ng.NoiseGradPlusPlus(model=model, std=method[1]['std'], sg_std=method[1]['sgd'], n=20,
                                                m=20)

            if method[1]['y_out'].shape[0] != inputs.shape[0]:

                explanation = np.array(
                    analyzer.enhance_explanation(inputs=inputs,
                                                 targets=np.repeat(method[1]['y_out'], kwargs['num_classes'], axis=0),
                                                 explanation_fn=xg.saliency_explainer, **method[1]))

            else:
                explanation = np.array(
                    analyzer.enhance_explanation(inputs=inputs, targets=method[1]['y_out'],
                                                 explanation_fn=xg.saliency_explainer, **method[1]))
        else:
            if method[0] == 'NoiseGrad':

                analyzer = ng.NoiseGrad(model=model, std=method[1]['std'], n=20)
            else:
                analyzer = ng.NoiseGradPlusPlus(model=model, std=method[1]['std'], sg_std=method[1]['sgd'], n=20,
                                              m=20)

            if method[1]['y_out'].shape[0] != inputs.shape[0]:

                explanation = np.array(
                    analyzer.enhance_explanation(inputs=inputs, targets=np.repeat(method[1]['y_out'],kwargs['nr_samples'],axis = 0),
                                                 explanation_fn=xg.saliency_explainer, **method[1]))

            else:
                explanation = np.array(
                    analyzer.enhance_explanation(inputs=inputs, targets=method[1]['y_out'],
                                                 explanation_fn=xg.saliency_explainer, **method[1]))
    elif method[2] == "Control Var. Random Uniform":

        if method[1].get('fix',1):
            try:
                explanation = kwargs['explanation'].reshape(*inputs.shape)
            except:
                if inputs.shape[0] != kwargs['explanation'].shape[0]:
                    explanation = np.repeat(kwargs['explanation'],int(inputs.shape[0]/kwargs['explanation'].shape[0]), axis=0)
                    explanation = explanation.reshape(*inputs.shape)
        else:

            explanation = np.random.rand(*inputs.shape)

    else:
        if kwargs.get('num_classes',0)>0:

            if method[2] == 'LRPcomp':
                analyzer = xai_aw.LRPcomposite(model_wo_softmax,"index", **method[1])
            elif method[1]:
                analyzer = innvestigate.create_analyzer(method[0],
                                                    model_wo_softmax, **method[1], neuron_selection_mode="index")

            else:
                analyzer = innvestigate.create_analyzer(method[0],
                                                        model_wo_softmax, neuron_selection_mode="index")
        else:
            if method[2] == 'LRPcomp':
                analyzer = xai_aw.LRPcomposite(model_wo_softmax, "max_activation", **method[1])
            elif method[1]:
                analyzer = innvestigate.create_analyzer(method[0],
                                                        model_wo_softmax, **method[1])

            else:
                analyzer = innvestigate.create_analyzer(method[0],
                                                        model_wo_softmax)

        if kwargs.get('num_classes', 0) >0:

            explanation = np.array(analyzer.analyze(inputs,targets))
        else:
            explanation = np.array(analyzer.analyze(inputs))

    if np.isnan(explanation).sum()>0:
        pdb.set_trace()

    return explanation.reshape(og_shape)


#%%


def run_quantus(args: Dict,
                explanations: Dict,
                metrics: Dict,
                xai_methods: Any,
                **params,
                ):
    """
    Function running pre-defined evaluation metrics in quantus on different explanation techniques
    :param args: model - keras.Model (trained model instance)
                x_batch - input batch
                y_batch - output batch
                s_batch - explanation batch
                n_samp - number of iterations for evaluation procedure
                num_cl -  number of classes
    :param explanations: same as s_batch
    :param metrics: Dict of metric function and hyperparameter settings
    :param xai_methods: dict of explanation name and hyperparameters settings
    :param params: dirout - output directory for back-up files in
                   csvfile - filename for back-up
    :return: dict of {metric: explanation: scores (float or array)}
    """

    results = {metric: {} for metric, metric_func in metrics.items()}
    dirout = params['dirout']
    csv_file = params['csvfile']
    for metric, metric_func in metrics.items():
        if metric is "RandomLogit":
            num_cl = args["num_cl"]
        else:
            num_cl = 0


        for method in xai_methods:
            print(metric, ":", method[2])
            if metric == "ROAD":
                scores = []
                for i in range(args["n_iter"]):
                    score = metric_func(model=args['model'],
                                         x_batch=args['x_batch'],
                                         y_batch=args['y_batch'],
                                         a_batch=explanations[method[2]],
                                         s_batch=args['s_batch'],
                                         explain_func=generate_tf_innvestigation,
                                         explain_func_kwargs={"method": method,
                                                              "explanation": explanations[method[2]],
                                                              'nr_samples':  args["n_smps"],
                                                              "num_classes": num_cl})
                    scores.append(score)
            elif metric in ["Robustness", "LocalLipschitzEstimate", "AvgSensitivity"]:
                scores = []
                if method[2] is "Control Var. Random Uniform":
                    method[1] = {'fix': 0}
                if params['net'] == 'CNN':
                    for i in range(args["n_iter"]):
                        score = metric_func(model=args['model'],
                                             x_batch=args['x_batch'][i*args["n_sms"]:(i+1)*args["n_sms"],...],
                                             y_batch=args['y_batch'][i*args["n_sms"]:(i+1)*args["n_sms"]],
                                             a_batch=explanations[method[2]][i*args["n_sms"]:(i+1)*args["n_sms"],...],
                                             s_batch=args['s_batch'][i*args["n_sms"]:(i+1)*args["n_sms"],...],
                                             explain_func=generate_tf_innvestigation,
                                             explain_func_kwargs={"method": method,
                                                                  "explanation": explanations[method[2]][i*args["n_sms"]:(i+1)*args["n_sms"],...],
                                                                  'nr_samples': args["n_sms"],
                                                                  "num_classes": num_cl})
                        scores.append(score)
                else:
                    scores = metric_func(model=args['model'],
                                         x_batch=args['x_batch'],
                                         y_batch=args['y_batch'],
                                         a_batch=explanations[method[2]],
                                         s_batch=args['s_batch'],
                                         explain_func=generate_tf_innvestigation,
                                         explain_func_kwargs={"method": method,
                                                              "explanation": explanations[method[2]],
                                                              'nr_samples': args["n_smps"],
                                                              "num_classes": num_cl})
            else:
                scores = metric_func(model=args['model'],
                            x_batch=args['x_batch'],
                            y_batch=args['y_batch'],
                            a_batch=explanations[method[2]],
                            s_batch=args['s_batch'],
                            explain_func=generate_tf_innvestigation,
                            explain_func_kwargs={"method": method,
                                                 "explanation": explanations[method[2]],
                                                 'nr_samples': args["n_smps"],
                                                 "num_classes": num_cl})

            results[metric][method[2]] = scores
        df = pd.DataFrame(results)
        df.to_csv(dirout + csv_file, index=False, header=False)

    return results


'''
def run_quantus(args: Dict,
                explanations: Dict,
                metrics: Dict,
                xai_methods: Any,
                **params,
                ):
    """
    Function running pre-defined evaluation metrics in quantus on different explanation techniques
    :param model:
    :param x_batch:
    :param y_batch:
    :param explanations:
    :param s_batch:
    :param metrics:
    :param xai_methods:
    :return:
    """

    # Score explanation methods using Quantus.
    results = {k[2]: {} for k in xai_methods}
    dirout = params['dirout']
    csv_file = params['csvfile']
    for method in xai_methods:
        for metric, metric_func in metrics.items():
            print(method[2], ":", metric)
            if method[2] == 'NoiseGrad' or method[2] == 'FusionGrad':
                scores = metric_func(model=args['model'],
                                     x_batch=args['x_batch'],
                                     y_batch=args['y_batch'],
                                     a_batch=explanations[method[2]],
                                     s_batch=args['s_batch'],
                                     **{"explain_func": generate_tf_innvestigation,
                                        "method": method,
                                        "noise_type": "multiplicative",
                                        "net": args['net']})
            else:
                scores = metric_func(model=args['model'],
                                     x_batch=args['x_batch'],
                                     y_batch=args['y_batch'],
                                     a_batch=explanations[method[2]],
                                     s_batch=args['s_batch'],
                                     **{"explain_func": generate_tf_innvestigation,
                                        "method": method,
                                        "noise_type": "multiplicative",
                                        "net": args['net']})


            results[method[2]][metric] = scores
        df = pd.DataFrame(results)
        df.to_csv(dirout + csv_file, index=False, header=False)

    return results

'''
