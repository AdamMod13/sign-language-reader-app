description: ObjectDetector for building object detection model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.ObjectDetector" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="create"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="export_labels"/>
<meta itemprop="property" content="export_model"/>
<meta itemprop="property" content="export_tflite"/>
<meta itemprop="property" content="quantization_aware_training"/>
<meta itemprop="property" content="restore_float_ckpt"/>
<meta itemprop="property" content="summary"/>
</div>

# mediapipe_model_maker.object_detector.ObjectDetector

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/object_detector.py#L36-L353">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



ObjectDetector for building object detection model.

Inherits From: [`Classifier`](../../mediapipe_model_maker/object_detector/object_detector/classifier/Classifier.md), [`CustomModel`](../../mediapipe_model_maker/object_detector/object_detector/classifier/custom_model/CustomModel.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.ObjectDetector`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.ObjectDetector(
    model_spec: <a href="../../mediapipe_model_maker/object_detector/ModelSpec.md"><code>mediapipe_model_maker.object_detector.ModelSpec</code></a>,
    label_names: List[str],
    hparams: <a href="../../mediapipe_model_maker/object_detector/HParams.md"><code>mediapipe_model_maker.object_detector.HParams</code></a>,
    model_options: <a href="../../mediapipe_model_maker/object_detector/ModelOptions.md"><code>mediapipe_model_maker.object_detector.ModelOptions</code></a>
) -> None
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
Specifications for the model.
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
The hyperparameters for training object detector.
</td>
</tr><tr>
<td>
`model_options`<a id="model_options"></a>
</td>
<td>
Options for creating the object detector model.
</td>
</tr>
</table>



## Methods

<h3 id="create"><code>create</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/object_detector.py#L63-L96">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>create(
    train_data: <a href="../../mediapipe_model_maker/object_detector/Dataset.md"><code>mediapipe_model_maker.object_detector.Dataset</code></a>,
    validation_data: <a href="../../mediapipe_model_maker/object_detector/Dataset.md"><code>mediapipe_model_maker.object_detector.Dataset</code></a>,
    options: <a href="../../mediapipe_model_maker/object_detector/ObjectDetectorOptions.md"><code>mediapipe_model_maker.object_detector.ObjectDetectorOptions</code></a>
) -> 'ObjectDetector'
</code></pre>

Creates and trains an ObjectDetector.

Loads data and trains the model based on data for object detection.

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
Configurations for creating and training object detector.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An instance of ObjectDetector.
</td>
</tr>

</table>



<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/object_detector.py#L232-L259">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    dataset: <a href="../../mediapipe_model_maker/object_detector/Dataset.md"><code>mediapipe_model_maker.object_detector.Dataset</code></a>,
    batch_size: int = 1
) -> Tuple[List[float], Dict[str, float]]
</code></pre>

Overrides Classifier.evaluate to calculate COCO metrics.


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

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/object_detector.py#L261-L334">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_model(
    model_name: str = &#x27;model.tflite&#x27;,
    quantization_config: Optional[<a href="../../mediapipe_model_maker/quantization/QuantizationConfig.md"><code>mediapipe_model_maker.quantization.QuantizationConfig</code></a>] = None
)
</code></pre>

Converts and saves the model to a TFLite file with metadata included.

The model export format is automatically set based on whether or not
`quantization_aware_training`(QAT) was run. The model exports to float32 by
default and will export to an int8 quantized model if QAT was run. To export
a float32 model after running QAT, run `restore_float_ckpt` before this
method. For custom post-training quantization without QAT, use the
quantization_config parameter.

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
The configuration for model quantization. Note that
int8 quantization aware training is automatically applied when possible.
This parameter is used to specify other post-training quantization
options such as fp16 and int8 without QAT.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If a custom quantization_config is specified when the model
has quantization aware training enabled.
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



<h3 id="quantization_aware_training"><code>quantization_aware_training</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/object_detector.py#L163-L230">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>quantization_aware_training(
    train_data: <a href="../../mediapipe_model_maker/object_detector/Dataset.md"><code>mediapipe_model_maker.object_detector.Dataset</code></a>,
    validation_data: <a href="../../mediapipe_model_maker/object_detector/Dataset.md"><code>mediapipe_model_maker.object_detector.Dataset</code></a>,
    qat_hparams: <a href="../../mediapipe_model_maker/object_detector/QATHParams.md"><code>mediapipe_model_maker.object_detector.QATHParams</code></a>
) -> None
</code></pre>

Runs quantization aware training(QAT) on the model.

The QAT step happens after training a regular float model from the `create`
method. This additional step will fine-tune the model with a lower precision
in order mimic the behavior of a quantized model. The resulting quantized
model generally has better performance than a model which is quantized
without running QAT. See the following link for more information:
- https://www.tensorflow.org/model_optimization/guide/quantization/training

Just like training the float model using the `create` method, the QAT step
also requires some manual tuning of hyperparameters. In order to run QAT
more than once for purposes such as hyperparameter tuning, use the
`restore_float_ckpt` method to restore the model state to the trained float
checkpoint without having to rerun the `create` method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`train_data`
</td>
<td>
Training dataset.
</td>
</tr><tr>
<td>
`validation_data`
</td>
<td>
Validaiton dataset.
</td>
</tr><tr>
<td>
`qat_hparams`
</td>
<td>
Configuration for QAT.
</td>
</tr>
</table>



<h3 id="restore_float_ckpt"><code>restore_float_ckpt</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/object_detector.py#L135-L160">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>restore_float_ckpt() -> None
</code></pre>

Loads a float checkpoint of the model from {hparams.export_dir}/float_ckpt.

The float checkpoint at {hparams.export_dir}/float_ckpt is automatically
saved after training an ObjectDetector using the `create` method. This
method is used to restore the trained float checkpoint state of the model in
order to run `quantization_aware_training` multiple times. Example usage:

# Train a model
model = object_detector.create(...)
# Run QAT
model.quantization_aware_training(...)
model.evaluate(...)
# Restore the float checkpoint to run QAT again
model.restore_float_ckpt()
# Run QAT with different parameters
model.quantization_aware_training(...)
model.evaluate(...)

<h3 id="summary"><code>summary</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/tasks/custom_model.py#L51-L53">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>summary()
</code></pre>

Prints a summary of the model.




