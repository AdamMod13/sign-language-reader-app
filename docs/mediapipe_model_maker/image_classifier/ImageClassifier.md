description: ImageClassifier for building image classification model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.image_classifier.ImageClassifier" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="export_labels"/>
<meta itemprop="property" content="export_model"/>
<meta itemprop="property" content="export_tflite"/>
<meta itemprop="property" content="summary"/>
</div>

# mediapipe_model_maker.image_classifier.ImageClassifier

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/image_classifier/image_classifier.py#L35-L204">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



ImageClassifier for building image classification model.

Inherits From: [`Classifier`](../../mediapipe_model_maker/object_detector/object_detector/classifier/Classifier.md), [`CustomModel`](../../mediapipe_model_maker/object_detector/object_detector/classifier/custom_model/CustomModel.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.image_classifier.ImageClassifier(
    model_spec: <a href="../../mediapipe_model_maker/image_classifier/ModelSpec.md"><code>mediapipe_model_maker.image_classifier.ModelSpec</code></a>,
    label_names: List[str],
    hparams: <a href="../../mediapipe_model_maker/image_classifier/HParams.md"><code>mediapipe_model_maker.image_classifier.HParams</code></a>,
    model_options: <a href="../../mediapipe_model_maker/image_classifier/ModelOptions.md"><code>mediapipe_model_maker.image_classifier.ModelOptions</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_spec`<a id="model_spec"></a>
</td>
<td>
Specification for the model.
</td>
</tr><tr>
<td>
`label_names`<a id="label_names"></a>
</td>
<td>
A list of label names for the classes.
</td>
</tr><tr>
<td>
`hparams`<a id="hparams"></a>
</td>
<td>
The hyperparameters for training image classifier.
</td>
</tr><tr>
<td>
`model_options`<a id="model_options"></a>
</td>
<td>
Model options for creating image classifier.
</td>
</tr>
</table>



## Methods

<h3 id="create"><code>create</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/image_classifier/image_classifier.py#L65-L100">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create(
    train_data: <a href="../../mediapipe_model_maker/object_detector/dataset/classification_dataset/ClassificationDataset.md"><code>mediapipe_model_maker.object_detector.dataset.classification_dataset.ClassificationDataset</code></a>,
    validation_data: <a href="../../mediapipe_model_maker/object_detector/dataset/classification_dataset/ClassificationDataset.md"><code>mediapipe_model_maker.object_detector.dataset.classification_dataset.ClassificationDataset</code></a>,
    options: <a href="../../mediapipe_model_maker/image_classifier/ImageClassifierOptions.md"><code>mediapipe_model_maker.image_classifier.ImageClassifierOptions</code></a>
) -> 'ImageClassifier'
</code></pre>

Creates and trains an ImageClassifier.

Loads data and trains the model based on data for image classification. If a
checkpoint file exists in the {options.hparams.export_dir}/checkpoint/
directory, the training process will load the weight from the checkpoint
file for continual training.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`train_data`
</td>
<td>
Training data.
</td>
</tr><tr>
<td>
`validation_data`
</td>
<td>
Validation data.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
configuration to create image classifier.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance based on ImageClassifier.
</td>
</tr>

</table>



<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/tasks/classifier.py#L114-L126">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    data: <a href="../../mediapipe_model_maker/quantization/ds/Dataset.md"><code>mediapipe_model_maker.quantization.ds.Dataset</code></a>,
    batch_size: int = 32
) -> Any
</code></pre>

Evaluates the classifier with the provided evaluation dataset.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data`
</td>
<td>
Evaluation dataset
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Number of samples per evaluation step.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The loss value and accuracy.
</td>
</tr>

</table>



<h3 id="export_labels"><code>export_labels</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/tasks/classifier.py#L128-L142">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_labels(
    export_dir: str, label_filename: str = &#x27;labels.txt&#x27;
)
</code></pre>

Exports classification labels into a label file.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`export_dir`
</td>
<td>
The directory to save exported files.
</td>
</tr><tr>
<td>
`label_filename`
</td>
<td>
File name to save labels model. The full export path is
{export_dir}/{label_filename}.
</td>
</tr>
</table>



<h3 id="export_model"><code>export_model</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/image_classifier/image_classifier.py#L142-L174">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_model(
    model_name: str = &#x27;model.tflite&#x27;,
    quantization_config: Optional[<a href="../../mediapipe_model_maker/quantization/QuantizationConfig.md"><code>mediapipe_model_maker.quantization.QuantizationConfig</code></a>] = None
)
</code></pre>

Converts and saves the model to a TFLite file with metadata included.

Note that only the TFLite file is needed for deployment. This function also
saves a metadata.json file to the same directory as the TFLite file which
can be used to interpret the metadata content in the TFLite file.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_name`
</td>
<td>
File name to save TFLite model with metadata. The full export
path is {self._hparams.export_dir}/{model_name}.
</td>
</tr><tr>
<td>
`quantization_config`
</td>
<td>
The configuration for model quantization.
</td>
</tr>
</table>



<h3 id="export_tflite"><code>export_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/tasks/custom_model.py#L56-L84">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_tflite(
    export_dir: str,
    tflite_filename: str = &#x27;model.tflite&#x27;,
    quantization_config: Optional[<a href="../../mediapipe_model_maker/quantization/QuantizationConfig.md"><code>mediapipe_model_maker.quantization.QuantizationConfig</code></a>] = None,
    preprocess: Optional[Callable[..., bool]] = None
)
</code></pre>

Converts the model to requested formats.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`export_dir`
</td>
<td>
The directory to save exported files.
</td>
</tr><tr>
<td>
`tflite_filename`
</td>
<td>
File name to save TFLite model. The full export path is
{export_dir}/{tflite_filename}.
</td>
</tr><tr>
<td>
`quantization_config`
</td>
<td>
The configuration for model quantization.
</td>
</tr><tr>
<td>
`preprocess`
</td>
<td>
A callable to preprocess the representative dataset for
quantization. The callable takes three arguments in order: feature,
label, and is_training.
</td>
</tr>
</table>



<h3 id="summary"><code>summary</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/tasks/custom_model.py#L51-L53">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>summary()
</code></pre>

Prints a summary of the model.




