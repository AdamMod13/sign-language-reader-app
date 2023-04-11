description: The abstract base class that represents a custom TensorFlow model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.CustomModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="export_tflite"/>
<meta itemprop="property" content="summary"/>
</div>

# mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.CustomModel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/tasks/custom_model.py#L31-L84">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The abstract base class that represents a custom TensorFlow model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.CustomModel(
    model_spec: Any, shuffle: bool
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
`shuffle`<a id="shuffle"></a>
</td>
<td>
Whether the training data need be shuffled.
</td>
</tr>
</table>



## Methods

<h3 id="evaluate"><code>evaluate</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/tasks/custom_model.py#L46-L49">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>evaluate(
    data: <a href="../../../../../mediapipe_model_maker/quantization/ds/Dataset.md"><code>mediapipe_model_maker.quantization.ds.Dataset</code></a>,
    **kwargs
)
</code></pre>

Evaluates the model with the provided data.


<h3 id="export_tflite"><code>export_tflite</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/tasks/custom_model.py#L56-L84">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_tflite(
    export_dir: str,
    tflite_filename: str = &#x27;model.tflite&#x27;,
    quantization_config: Optional[<a href="../../../../../mediapipe_model_maker/quantization/QuantizationConfig.md"><code>mediapipe_model_maker.quantization.QuantizationConfig</code></a>] = None,
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




