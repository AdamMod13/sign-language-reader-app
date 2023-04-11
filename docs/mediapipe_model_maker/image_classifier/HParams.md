description: The hyperparameters for training image classifiers.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.image_classifier.HParams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="decay_samples"/>
<meta itemprop="property" content="distribution_strategy"/>
<meta itemprop="property" content="do_data_augmentation"/>
<meta itemprop="property" content="do_fine_tuning"/>
<meta itemprop="property" content="epochs"/>
<meta itemprop="property" content="export_dir"/>
<meta itemprop="property" content="l1_regularizer"/>
<meta itemprop="property" content="l2_regularizer"/>
<meta itemprop="property" content="label_smoothing"/>
<meta itemprop="property" content="learning_rate"/>
<meta itemprop="property" content="num_gpus"/>
<meta itemprop="property" content="shuffle"/>
<meta itemprop="property" content="steps_per_epoch"/>
<meta itemprop="property" content="tpu"/>
<meta itemprop="property" content="warmup_epochs"/>
</div>

# mediapipe_model_maker.image_classifier.HParams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/image_classifier/hyperparameters.py#L21-L55">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The hyperparameters for training image classifiers.

Inherits From: [`HParams`](../../mediapipe_model_maker/text_classifier/HParams.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.image_classifier.HParams(
    learning_rate: float = 0.001,
    batch_size: int = 2,
    epochs: int = 10,
    steps_per_epoch: Optional[int] = None,
    shuffle: bool = False,
    export_dir: str = tempfile.mkdtemp(),
    distribution_strategy: str = &#x27;off&#x27;,
    num_gpus: int = -1,
    tpu: str = &#x27;&#x27;,
    do_fine_tuning: bool = False,
    l1_regularizer: float = 0.0,
    l2_regularizer: float = 0.0001,
    label_smoothing: float = 0.1,
    do_data_augmentation: bool = True,
    decay_samples: int = (10000 * 256),
    warmup_epochs: int = 2
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
`l1_regularizer`<a id="l1_regularizer"></a>
</td>
<td>
A regularizer that applies a L1 regularization penalty.
</td>
</tr><tr>
<td>
`l2_regularizer`<a id="l2_regularizer"></a>
</td>
<td>
A regularizer that applies a L2 regularization penalty.
</td>
</tr><tr>
<td>
`label_smoothing`<a id="label_smoothing"></a>
</td>
<td>
Amount of label smoothing to apply. See tf.keras.losses for
more details.
</td>
</tr><tr>
<td>
`do_data_augmentation`<a id="do_data_augmentation"></a>
</td>
<td>
A boolean controlling whether the training dataset is
augmented by randomly distorting input images, including random cropping,
flipping, etc. See utils.image_preprocessing documentation for details.
</td>
</tr><tr>
<td>
`decay_samples`<a id="decay_samples"></a>
</td>
<td>
Number of training samples used to calculate the decay steps
and create the training optimizer.
</td>
</tr><tr>
<td>
`warmup_steps`<a id="warmup_steps"></a>
</td>
<td>
Number of warmup steps for a linear increasing warmup schedule
on learning rate. Used to set up warmup schedule by model_util.WarmUp.s
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
</tr><tr>
<td>
`warmup_epochs`<a id="warmup_epochs"></a>
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
`2`
</td>
</tr><tr>
<td>
decay_samples<a id="decay_samples"></a>
</td>
<td>
`2560000`
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
do_data_augmentation<a id="do_data_augmentation"></a>
</td>
<td>
`True`
</td>
</tr><tr>
<td>
do_fine_tuning<a id="do_fine_tuning"></a>
</td>
<td>
`False`
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
l1_regularizer<a id="l1_regularizer"></a>
</td>
<td>
`0.0`
</td>
</tr><tr>
<td>
l2_regularizer<a id="l2_regularizer"></a>
</td>
<td>
`0.0001`
</td>
</tr><tr>
<td>
label_smoothing<a id="label_smoothing"></a>
</td>
<td>
`0.1`
</td>
</tr><tr>
<td>
learning_rate<a id="learning_rate"></a>
</td>
<td>
`0.001`
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
</tr><tr>
<td>
warmup_epochs<a id="warmup_epochs"></a>
</td>
<td>
`2`
</td>
</tr>
</table>

