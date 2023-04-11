description: Loads a tensorflow Keras model from file and returns the Keras model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.object_detector.model_util.load_keras_model" />
<meta itemprop="path" content="Stable" />
</div>

# mediapipe_model_maker.object_detector.object_detector.model_util.load_keras_model

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/model_util.py#L50-L67">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Loads a tensorflow Keras model from file and returns the Keras model.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.model_util.load_keras_model`, `mediapipe_model_maker.object_detector.object_detector.classifier.model_util.load_keras_model`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.object_detector.model_util.load_keras_model(
    model_path: str, compile_on_load: bool = False
) -> tf.keras.Model
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_path`<a id="model_path"></a>
</td>
<td>
Absolute path to a directory containing model data, such as
/<parent_path>/saved_model/.
</td>
</tr><tr>
<td>
`compile_on_load`<a id="compile_on_load"></a>
</td>
<td>
Whether the model should be compiled while loading. If
False, the model returned has to be compiled with the appropriate loss
function and custom metrics before running for inference on a test
dataset.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A tensorflow Keras model.
</td>
</tr>

</table>

