description: The hyperparameters for running quantization aware training (QAT) on object detectors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.QATHParams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="decay_rate"/>
<meta itemprop="property" content="decay_steps"/>
<meta itemprop="property" content="epochs"/>
<meta itemprop="property" content="learning_rate"/>
</div>

# mediapipe_model_maker.object_detector.QATHParams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/hyperparameters.py#L78-L101">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The hyperparameters for running quantization aware training (QAT) on object detectors.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.hyperparameters.QATHParams`, `mediapipe_model_maker.object_detector.object_detector.hp.QATHParams`, `mediapipe_model_maker.object_detector.object_detector.object_detector_options.hyperparameters.QATHParams`, `mediapipe_model_maker.object_detector.object_detector_options.hyperparameters.QATHParams`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.QATHParams(
    learning_rate: float = 0.03,
    batch_size: int = 32,
    epochs: int = 10,
    decay_steps: int = 231,
    decay_rate: float = 0.96
)
</code></pre>



<!-- Placeholder for "Used in" -->

For more information on QAT, see:
  https://www.tensorflow.org/model_optimization/guide/quantization/training



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>
Learning rate to use for gradient descent QAT.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Batch size for QAT.
</td>
</tr><tr>
<td>
`epochs`<a id="epochs"></a>
</td>
<td>
Number of training iterations over the dataset.
</td>
</tr><tr>
<td>
`decay_steps`<a id="decay_steps"></a>
</td>
<td>
Learning rate decay steps for Exponential Decay. See
https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
  for more information.
</td>
</tr><tr>
<td>
`decay_rate`<a id="decay_rate"></a>
</td>
<td>
Learning rate decay rate for Exponential Decay. See
https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
  for more information.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>








<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
batch_size<a id="batch_size"></a>
</td>
<td>
`32`
</td>
</tr><tr>
<td>
decay_rate<a id="decay_rate"></a>
</td>
<td>
`0.96`
</td>
</tr><tr>
<td>
decay_steps<a id="decay_steps"></a>
</td>
<td>
`231`
</td>
</tr><tr>
<td>
epochs<a id="epochs"></a>
</td>
<td>
`10`
</td>
</tr><tr>
<td>
learning_rate<a id="learning_rate"></a>
</td>
<td>
`0.03`
</td>
</tr>
</table>

