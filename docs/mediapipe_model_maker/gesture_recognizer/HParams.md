description: The hyperparameters for training gesture recognizer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.gesture_recognizer.HParams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="batch_size"/>
<meta itemprop="property" content="distribution_strategy"/>
<meta itemprop="property" content="epochs"/>
<meta itemprop="property" content="export_dir"/>
<meta itemprop="property" content="gamma"/>
<meta itemprop="property" content="learning_rate"/>
<meta itemprop="property" content="lr_decay"/>
<meta itemprop="property" content="num_gpus"/>
<meta itemprop="property" content="shuffle"/>
<meta itemprop="property" content="steps_per_epoch"/>
<meta itemprop="property" content="tpu"/>
</div>

# mediapipe_model_maker.gesture_recognizer.HParams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/gesture_recognizer/hyperparameters.py#L21-L40">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



The hyperparameters for training gesture recognizer.

Inherits From: [`HParams`](../../mediapipe_model_maker/text_classifier/HParams.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.gesture_recognizer.HParams(
    learning_rate: float = 0.001,
    batch_size: int = 2,
    epochs: int = 10,
    steps_per_epoch: Optional[int] = None,
    shuffle: bool = False,
    export_dir: str = tempfile.mkdtemp(),
    distribution_strategy: str = &#x27;off&#x27;,
    num_gpus: int = -1,
    tpu: str = &#x27;&#x27;,
    lr_decay: float = 0.99,
    gamma: int = 2
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
`lr_decay`<a id="lr_decay"></a>
</td>
<td>
Learning rate decay to use for gradient descent training.
</td>
</tr><tr>
<td>
`gamma`<a id="gamma"></a>
</td>
<td>
Gamma parameter for focal loss.
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
`2`
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
gamma<a id="gamma"></a>
</td>
<td>
`2`
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
lr_decay<a id="lr_decay"></a>
</td>
<td>
`0.99`
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

