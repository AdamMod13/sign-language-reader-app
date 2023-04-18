description: Applies a warmup schedule on a given learning rate decay schedule.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.object_detector.model_util.WarmUp" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# mediapipe_model_maker.object_detector.object_detector.model_util.WarmUp

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/model_util.py#L165-L208">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Applies a warmup schedule on a given learning rate decay schedule.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.model_util.WarmUp`, `mediapipe_model_maker.object_detector.object_detector.classifier.model_util.WarmUp`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.object_detector.model_util.WarmUp(
    initial_learning_rate: float,
    decay_schedule_fn: Callable[[Any], Any],
    warmup_steps: int,
    name: Optional[str] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`initial_learning_rate`<a id="initial_learning_rate"></a>
</td>
<td>
learning rate after the warmup.
</td>
</tr><tr>
<td>
`decay_schedule_fn`<a id="decay_schedule_fn"></a>
</td>
<td>
A function maps step to learning rate. Will be applied
for values of step larger than 'warmup_steps'.
</td>
</tr><tr>
<td>
`warmup_steps`<a id="warmup_steps"></a>
</td>
<td>
Number of steps to do warmup for.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
TF namescope under which to perform the learning rate calculation.
</td>
</tr>
</table>



## Methods

<h3 id="from_config"><code>from_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Instantiates a `LearningRateSchedule` from its config.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
Output of `get_config()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `LearningRateSchedule` instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/model_util.py#L202-L208">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config() -> Dict[str, Any]
</code></pre>




<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/model_util.py#L188-L200">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    step: Union[int, tf.Tensor]
) -> tf.Tensor
</code></pre>

Call self as a function.




