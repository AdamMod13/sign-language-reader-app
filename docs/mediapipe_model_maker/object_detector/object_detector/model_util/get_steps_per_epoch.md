description: Gets the estimated training steps per epoch.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.object_detector.model_util.get_steps_per_epoch" />
<meta itemprop="path" content="Stable" />
</div>

# mediapipe_model_maker.object_detector.object_detector.model_util.get_steps_per_epoch

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/model_util.py#L85-L112">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the estimated training steps per epoch.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.model_util.get_steps_per_epoch`, `mediapipe_model_maker.object_detector.object_detector.classifier.model_util.get_steps_per_epoch`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.object_detector.model_util.get_steps_per_epoch(
    steps_per_epoch: Optional[int] = None,
    batch_size: Optional[int] = None,
    train_data: Optional[<a href="../../../../mediapipe_model_maker/quantization/ds/Dataset.md"><code>mediapipe_model_maker.quantization.ds.Dataset</code></a>] = None
) -> int
</code></pre>



<!-- Placeholder for "Used in" -->

1. If `steps_per_epoch` is set, returns `steps_per_epoch` directly.
2. Else if we can get the length of training data successfully, returns
   `train_data_length // batch_size`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`steps_per_epoch`<a id="steps_per_epoch"></a>
</td>
<td>
int, training steps per epoch.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
int, batch size.
</td>
</tr><tr>
<td>
`train_data`<a id="train_data"></a>
</td>
<td>
training data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Estimated training steps per epoch.
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
if both steps_per_epoch and train_data are not set.
</td>
</tr>
</table>

