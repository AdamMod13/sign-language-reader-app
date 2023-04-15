description: Converts the input Keras model to TFLite format.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.object_detector.model_util.convert_to_tflite" />
<meta itemprop="path" content="Stable" />
</div>

# mediapipe_model_maker.object_detector.object_detector.model_util.convert_to_tflite

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/model_util.py#L115-L147">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Converts the input Keras model to TFLite format.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.model_util.convert_to_tflite`, `mediapipe_model_maker.object_detector.object_detector.classifier.model_util.convert_to_tflite`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.object_detector.model_util.convert_to_tflite(
    model: tf.keras.Model,
    quantization_config: Optional[<a href="../../../../mediapipe_model_maker/quantization/QuantizationConfig.md"><code>mediapipe_model_maker.quantization.QuantizationConfig</code></a>] = None,
    supported_ops: Tuple[tf.lite.OpsSet, ...] = (tf.lite.OpsSet.TFLITE_BUILTINS,),
    preprocess: Optional[Callable[..., Any]] = None
) -> bytearray
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model`<a id="model"></a>
</td>
<td>
Keras model to be converted to TFLite.
</td>
</tr><tr>
<td>
`quantization_config`<a id="quantization_config"></a>
</td>
<td>
Configuration for post-training quantization.
</td>
</tr><tr>
<td>
`supported_ops`<a id="supported_ops"></a>
</td>
<td>
A list of supported ops in the converted TFLite file.
</td>
</tr><tr>
<td>
`preprocess`<a id="preprocess"></a>
</td>
<td>
A callable to preprocess the representative dataset for
quantization. The callable takes three arguments in order: feature, label,
and is_training.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
bytearray of TFLite model
</td>
</tr>

</table>

