description: Configuration for post-training quantization.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.quantization.QuantizationConfig" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="for_dynamic"/>
<meta itemprop="property" content="for_float16"/>
<meta itemprop="property" content="for_int8"/>
<meta itemprop="property" content="set_converter_with_quantization"/>
</div>

# mediapipe_model_maker.quantization.QuantizationConfig

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/quantization.py#L58-L213">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Configuration for post-training quantization.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.model_util.quantization.QuantizationConfig`, `mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.quantization.QuantizationConfig`, `mediapipe_model_maker.object_detector.object_detector.classifier.model_util.quantization.QuantizationConfig`, `mediapipe_model_maker.object_detector.object_detector.model_util.quantization.QuantizationConfig`, `mediapipe_model_maker.object_detector.object_detector.quantization.QuantizationConfig`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.quantization.QuantizationConfig(
    optimizations: Optional[Union[tf.lite.Optimize, List[tf.lite.Optimize]]] = None,
    representative_data: Optional[<a href="../../mediapipe_model_maker/quantization/ds/Dataset.md"><code>mediapipe_model_maker.quantization.ds.Dataset</code></a>] = None,
    quantization_steps: Optional[int] = None,
    inference_input_type: Optional[tf.dtypes.DType] = None,
    inference_output_type: Optional[tf.dtypes.DType] = None,
    supported_ops: Optional[Union[tf.lite.OpsSet, List[tf.lite.OpsSet]]] = None,
    supported_types: Optional[Union[tf.dtypes.DType, List[tf.dtypes.DType]]] = None,
    experimental_new_quantizer: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

Refer to
https://www.tensorflow.org/lite/performance/post_training_quantization
for different post-training quantization options.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`optimizations`<a id="optimizations"></a>
</td>
<td>
A list of optimizations to apply when converting the model.
If not set, use `[Optimize.DEFAULT]` by default.
</td>
</tr><tr>
<td>
`representative_data`<a id="representative_data"></a>
</td>
<td>
A representative ds.Dataset for post-training
quantization.
</td>
</tr><tr>
<td>
`quantization_steps`<a id="quantization_steps"></a>
</td>
<td>
Number of post-training quantization calibration steps
to run (default to DEFAULT_QUANTIZATION_STEPS).
</td>
</tr><tr>
<td>
`inference_input_type`<a id="inference_input_type"></a>
</td>
<td>
Target data type of real-number input arrays. Allows
for a different type for input arrays. Defaults to None. If set, must be
be `{tf.float32, tf.uint8, tf.int8}`.
</td>
</tr><tr>
<td>
`inference_output_type`<a id="inference_output_type"></a>
</td>
<td>
Target data type of real-number output arrays.
Allows for a different type for output arrays. Defaults to None. If set,
must be `{tf.float32, tf.uint8, tf.int8}`.
</td>
</tr><tr>
<td>
`supported_ops`<a id="supported_ops"></a>
</td>
<td>
Set of OpsSet options supported by the device. Used to Set
converter.target_spec.supported_ops.
</td>
</tr><tr>
<td>
`supported_types`<a id="supported_types"></a>
</td>
<td>
List of types for constant values on the target device.
Supported values are types exported by lite.constants. Frequently, an
optimization choice is driven by the most compact (i.e. smallest) type
in this list (default [constants.FLOAT]).
</td>
</tr><tr>
<td>
`experimental_new_quantizer`<a id="experimental_new_quantizer"></a>
</td>
<td>
Whether to enable experimental new quantizer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
if inference_input_type or inference_output_type are set but
not in {tf.float32, tf.uint8, tf.int8}.
</td>
</tr>
</table>



## Methods

<h3 id="for_dynamic"><code>for_dynamic</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/quantization.py#L142-L145">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>for_dynamic() -> 'QuantizationConfig'
</code></pre>

Creates configuration for dynamic range quantization.


<h3 id="for_float16"><code>for_float16</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/quantization.py#L178-L181">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>for_float16() -> 'QuantizationConfig'
</code></pre>

Creates configuration for float16 quantization.


<h3 id="for_int8"><code>for_int8</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/quantization.py#L147-L176">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>for_int8(
    representative_data: <a href="../../mediapipe_model_maker/quantization/ds/Dataset.md"><code>mediapipe_model_maker.quantization.ds.Dataset</code></a>,
    quantization_steps: int = DEFAULT_QUANTIZATION_STEPS,
    inference_input_type: tf.dtypes.DType = tf.uint8,
    inference_output_type: tf.dtypes.DType = tf.uint8,
    supported_ops: tf.lite.OpsSet = tf.lite.OpsSet.TFLITE_BUILTINS_INT8
) -> 'QuantizationConfig'
</code></pre>

Creates configuration for full integer quantization.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`representative_data`
</td>
<td>
Representative data used for post-training
quantization.
</td>
</tr><tr>
<td>
`quantization_steps`
</td>
<td>
Number of post-training quantization calibration steps
to run.
</td>
</tr><tr>
<td>
`inference_input_type`
</td>
<td>
Target data type of real-number input arrays.
</td>
</tr><tr>
<td>
`inference_output_type`
</td>
<td>
Target data type of real-number output arrays.
</td>
</tr><tr>
<td>
`supported_ops`
</td>
<td>
Set of `tf.lite.OpsSet` options, where each option
represents a set of operators supported by the target device.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
QuantizationConfig.
</td>
</tr>

</table>



<h3 id="set_converter_with_quantization"><code>set_converter_with_quantization</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/quantization.py#L183-L213">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_converter_with_quantization(
    converter: tf.lite.TFLiteConverter, **kwargs
) -> tf.lite.TFLiteConverter
</code></pre>

Sets input TFLite converter with quantization configurations.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`converter`
</td>
<td>
input tf.lite.TFLiteConverter.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
arguments used by ds.Dataset.gen_tf_dataset.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
tf.lite.TFLiteConverter with quantization configurations.
</td>
</tr>

</table>





