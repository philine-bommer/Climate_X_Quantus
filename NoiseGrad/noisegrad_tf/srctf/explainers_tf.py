"""Some examples of explanation methods that can be used in NosieGrad and NoiseGrad++ implementations."""
import innvestigate as inn
from .utils_tf import *
import pdb


def saliency_explainer(model, inputs, targets, normalize=False, **kwargs) -> np.array:
    """Implementation of InteGrated Gradients by Sundararajan et al., 2017 using Captum."""
    model_wo_softmax = inn.utils.keras.graph.model_wo_softmax(model)
    # assert (
    #     len(np.shape(inputs)) == 4
    # ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."
    if kwargs.get('num_classes',0)>0:
        explainers = inn.analyzer.gradient_based.Gradient(model_wo_softmax, neuron_selection_mode = "index")
        explanation = (
            explainers
            .analyze(inputs, kwargs['cl'])
        )

    else:
        explainers = inn.analyzer.gradient_based.Gradient(model_wo_softmax)
        explanation = (
            explainers
            .analyze(inputs)
        )
    dtype = kwargs.get('dtype', "rgb")
    if dtype == "rgb":
        explanation = explanation.sum(axis=0).reshape(kwargs.get("img_size", 224), kwargs.get("img_size", 224))

    # return (
    #     explanation
    # )

    if normalize:
        return normalize_heatmap(explanation)

    return explanation


def intgrad_explainer(
    model, inputs, targets, abs=False, normalize=False, **kwargs
) -> np.array:
    """Implementation of InteGrated Gradients by Sundararajan et al., 2017 using Captum."""
    model_wo_softmax = inn.utils.keras.graph.model_wo_softmax(model)
    # assert (
    #     len(np.shape(inputs)) == 4
    # ), "Inputs should be shaped (nr_samples, nr_channels, img_size, img_size) e.g., (1, 3, 224, 224)."
    explainers = inn.analyzer.gradient_based.IntegratedGradients(model_wo_softmax, steps=32)
    explanation = (explainers.analyze(inputs))
                   # create_analyzer(
                  # "integrated_gradients", model_wo_softmax, {"reference_inputs": np.zeros(inputs.shape),
                  #              "steps": 64})

    if abs:
        explanation = np.abs(explanation)
    if normalize:
        explanation = normalize_heatmap(explanation)

    dtype = kwargs.get('dtype', "rgb")
    if dtype == "rgb":
        explanation = explanation.sum(axis=0).reshape(kwargs.get("img_size", 224), kwargs.get("img_size", 224))

    return (
        explanation
    )
