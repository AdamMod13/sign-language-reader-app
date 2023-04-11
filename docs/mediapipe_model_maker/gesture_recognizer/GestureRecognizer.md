description: GestureRecognizer for building hand gesture recognizer model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.gesture_recognizer.GestureRecognizer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="export_labels"/>
<meta itemprop="property" content="export_model"/>
<meta itemprop="property" content="export_tflite"/>
<meta itemprop="property" content="summary"/>
</div>

# mediapipe_model_maker.gesture_recognizer.GestureRecognizer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/gesture_recognizer/gesture_recognizer.py#L35-L231">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



GestureRecognizer for building hand gesture recognizer model.

Inherits From: [`Classifier`](../../mediapipe_model_maker/object_detector/object_detector/classifier/Classifier.md), [`CustomModel`](../../mediapipe_model_maker/object_detector/object_detector/classifier/custom_model/CustomModel.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.gesture_recognizer.GestureRecognizer(
    label_names: List[str],
    model_options: <a href="../../mediapipe_model_maker/gesture_recognizer/ModelOptions.md"><code>mediapipe_model_maker.gesture_recognizer.ModelOptions</code></a>,
    hparams: <a href="../../mediapipe_model_maker/gesture_recognizer/HParams.md"><code>mediapipe_model_maker.gesture_recognizer.HParams</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`label_names`<a id="label_names"></a>
</td>
<td>
A list of label names for the classes.
</td>
</tr><tr>
<td>
`model_options`<a id="model_options"></a>
</td>
<td>
options to create gesture recognizer model.
</td>
</tr><tr>
<td>
`hparams`<a id="hparams"></a>
</td>
<td>
The hyperparameters for training hand gesture recognizer model.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`embedding_size`<a id="embedding_size"></a>
</td>
<td>
Size of the input gesture embedding vector.
</td>
</tr>
</table>



## Methods

<h3 id="create"><code>create</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/gesture_recognizer/gesture_recognizer.py#L63-L95">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create(
    train_data: <a href="../../mediapipe_model_maker/object_detector/dataset/classification_dataset/ClassificationDataset.md"><code>mediapipe_model_maker.object_detector.dataset.classification_dataset.ClassificationDataset</code></a>,
    validation_data: <a href="../../mediapipe_model_maker/object_detector/dataset/classification_dataset/ClassificationDataset.md"><code>mediapipe_model_maker.object_detector.dataset.classification_dataset.ClassificationDataset</code></a>,
    options: <a href="../../mediapipe_model_maker/gesture_recognizer/GestureRecognizerOptions.md"><code>mediapipe_model_maker.gesture_recognizer.GestureRecognizerOptions</code></a>
) -> 'GestureRecognizer'
</code></pre>

Creates and trains a hand gesture recognizer with input datasets.

If a checkpoint file exists in the {options.hparams.export_dir}/checkpoint/
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
options for creating and training gesture recognizer model.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance of GestureRecognizer.
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

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/gesture_recognizer/gesture_recognizer.py#L179-L231">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_model(
    model_name: str = &#x27;gesture_recognizer.task&#x27;
)
</code></pre>

Converts the model to TFLite and exports as a model bundle file.

Saves a model bundle file and metadata json file to hparams.export_dir. The
resulting model bundle file will contain necessary models for hand
detection, canned gesture classification, and customized gesture
classification. Only the model bundle file is needed for the downstream
gesture recognition task. The metadata.json file is saved only to
interpret the contents of the model bundle file.

The customized gesture model is in float without quantization. The model is
lightweight and there is no need to balance performance and efficiency by
quantization. The default score_thresholding is set to 0.5 as it can be
adjusted during inference.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model_name`
</td>
<td>
File name to save model bundle file. The full export path is
{export_dir}/{model_name}.
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




