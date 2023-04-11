description: Hyperparameters used for training models.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.text_classifier.HParams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="distribution_strategy"/>
<meta itemprop="property" content="export_dir"/>
<meta itemprop="property" content="num_gpus"/>
<meta itemprop="property" content="shuffle"/>
<meta itemprop="property" content="steps_per_epoch"/>
<meta itemprop="property" content="tpu"/>
</div>

# mediapipe_model_maker.text_classifier.HParams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/hyperparameters.py#L22-L67">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Hyperparameters used for training models.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.hyperparameters.hp.BaseHParams`, `mediapipe_model_maker.object_detector.object_detector.classifier.hp.BaseHParams`, `mediapipe_model_maker.object_detector.object_detector.hp.hp.BaseHParams`, `mediapipe_model_maker.object_detector.object_detector.object_detector_options.hyperparameters.hp.BaseHParams`, `mediapipe_model_maker.object_detector.object_detector_options.hyperparameters.hp.BaseHParams`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.text_classifier.HParams(
    learning_rate: float,
    batch_size: int,
    epochs: int,
    steps_per_epoch: Optional[int] = None,
    shuffle: bool = False,
    export_dir: str = tempfile.mkdtemp(),
    distribution_strategy: str = &#x27;off&#x27;,
    num_gpus: int = -1,
    tpu: str = &#x27;&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

A common set of hyperparameters shared by the training jobs of all model
maker tasks.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`learning_rate`<a id="learning_rate"></a>
</td>
<td>
The learning rate to use for gradient descent training.
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
`steps_per_epoch`<a id="steps_per_epoch"></a>
</td>
<td>
An optional integer indicate the number of training steps
per epoch. If not set, the training pipeline calculates the default steps
per epoch as the training dataset size devided by batch size.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
True if the dataset is shuffled before training.
</td>
</tr><tr>
<td>
`export_dir`<a id="export_dir"></a>
</td>
<td>
The location of the model checkpoint files.
</td>
</tr><tr>
<td>
`distribution_strategy`<a id="distribution_strategy"></a>
</td>
<td>
A string specifying which Distribution Strategy to
use. Accepted values are 'off', 'one_device', 'mirrored',
'parameter_server', 'multi_worker_mirrored', and 'tpu' -- case
insensitive. 'off' means not to use Distribution Strategy; 'tpu' means to
use TPUStrategy using `tpu_address`. See the tf.distribute.Strategy
documentation for more details:
https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy.
</td>
</tr><tr>
<td>
`num_gpus`<a id="num_gpus"></a>
</td>
<td>
How many GPUs to use at each worker with the
DistributionStrategies API. The default is -1, which means utilize all
available GPUs.
</td>
</tr><tr>
<td>
`tpu`<a id="tpu"></a>
</td>
<td>
The Cloud TPU to use for training. This should be either the name used
when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.
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
distribution_strategy<a id="distribution_strategy"></a>
</td>
<td>
`'off'`
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

