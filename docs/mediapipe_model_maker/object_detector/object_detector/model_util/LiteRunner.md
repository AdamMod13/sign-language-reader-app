description: A runner to do inference with the TFLite model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.object_detector.model_util.LiteRunner" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="run"/>
</div>

# mediapipe_model_maker.object_detector.object_detector.model_util.LiteRunner

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/model_util.py#L211-L283">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A runner to do inference with the TFLite model.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.model_util.LiteRunner`, `mediapipe_model_maker.object_detector.object_detector.classifier.model_util.LiteRunner`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.object_detector.model_util.LiteRunner(
    tflite_model: bytearray
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tflite_model`<a id="tflite_model"></a>
</td>
<td>
A valid flatbuffer representing the TFLite model.
</td>
</tr>
</table>



## Methods

<h3 id="run"><code>run</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/model_util.py#L225-L283">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>run(
    input_tensors: Union[List[tf.Tensor], Dict[str, tf.Tensor]]
) -> Union[List[tf.Tensor], tf.Tensor]
</code></pre>

Runs inference with the TFLite model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_tensors`
</td>
<td>
List / Dict of the input tensors of the TFLite model. The
order should be the same as the keras model if it's a list. It also
accepts tensor directly if the model has only 1 input.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of the output tensors for multi-output models, otherwise just
the output tensor. The order should be the same as the keras model.
</td>
</tr>

</table>





