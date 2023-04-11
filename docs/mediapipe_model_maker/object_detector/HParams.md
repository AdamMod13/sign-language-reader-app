description: The hyperparameters for training object detectors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.HParams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="distribution_strategy"/>
<meta itemprop="property" content="epochs"/>
<meta itemprop="property" content="export_dir"/>
<meta itemprop="property" content="learning_rate"/>
<meta itemprop="property" content="num_gpus"/>
<meta itemprop="property" content="shuffle"/>
<meta itemprop="property" content="steps_per_epoch"/>
<meta itemprop="property" content="tpu"/>
</div>

# mediapipe_model_maker.object_detector.HParams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/hyperparameters.py#L22-L75">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The hyperparameters for training object detectors.

Inherits From: [`HParams`](../../mediapipe_model_maker/text_classifier/HParams.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.hyperparameters.HParams`, `mediapipe_model_maker.object_detector.object_detector.hp.HParams`, `mediapipe_model_maker.object_detector.object_detector.object_detector_options.hyperparameters.HParams`, `mediapipe_model_maker.object_detector.object_detector_options.hyperparameters.HParams`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.HParams(
    learning_rate: float = 0.003,
    batch_size: int = 32,
    epochs: int = 10,
    steps_per_epoch: Optional[int] = None,
    shuffle: bool = False,
    export_dir: str = tempfile.mkdtemp(),
    distribution_strategy: str = &#x27;off&#x27;,
    num_gpus: int = -1,
    tpu: str = &#x27;&#x27;,
    learning_rate_boundaries: List[int] = dataclasses.field(default_factory=lambda : [5, 8]),
    learning_rate_decay_multipliers: List[float] = dataclasses.field(default_factory=lambda : [0.1, 0.01])
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>
Learning rate to use for gradient descent training.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Batch size for training.
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
`do_fine_tuning`<a id="do_fine_tuning"></a>
</td>
<td>
If true, the base module is trained together with the
classification layer on top.
</td>
</tr><tr>
<td>
`learning_rate_boundaries`<a id="learning_rate_boundaries"></a>
</td>
<td>
List of epoch boundaries where
learning_rate_boundaries[i] is the epoch where the learning rate will
decay to learning_rate * learning_rate_decay_multipliers[i].
</td>
</tr><tr>
<td>
`learning_rate_decay_multipliers`<a id="learning_rate_decay_multipliers"></a>
</td>
<td>
List of learning rate multipliers which
calculates the learning rate at the ith boundary as learning_rate *
learning_rate_decay_multipliers[i].
</td>
</tr><tr>
<td>
`steps_per_epoch`<a id="steps_per_epoch"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`export_dir`<a id="export_dir"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`distribution_strategy`<a id="distribution_strategy"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`num_gpus`<a id="num_gpus"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`tpu`<a id="tpu"></a>
</td>
<td>
Dataclass field
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
distribution_strategy<a id="distribution_strategy"></a>
</td>
<td>
`'off'`
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
export_dir<a id="export_dir"></a>
</td>
<td>
`'/tmp/tmphtnrd3xu'`
</td>
</tr><tr>
<td>
learning_rate<a id="learning_rate"></a>
</td>
<td>
`0.003`
</td>
</tr><tr>
<td>
num_gpus<a id="num_gpus"></a>
</td>
<td>
`-1`
</td>
</tr><tr>
<td>
shuffle<a id="shuffle"></a>
</td>
<td>
`False`
</td>
</tr><tr>
<td>
steps_per_epoch<a id="steps_per_epoch"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
tpu<a id="tpu"></a>
</td>
<td>
`''`
</td>
</tr>
</table>

