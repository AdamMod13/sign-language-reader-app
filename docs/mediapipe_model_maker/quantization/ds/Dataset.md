description: A generic dataset class for loading model training and evaluation dataset.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.quantization.ds.Dataset" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="gen_tf_dataset"/>
<meta itemprop="property" content="split"/>
</div>

# mediapipe_model_maker.quantization.ds.Dataset

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/data/dataset.py#L29-L166">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A generic dataset class for loading model training and evaluation dataset.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.dataset.classification_dataset.ds.Dataset`, `mediapipe_model_maker.object_detector.object_detector.classifier.classification_ds.ds.Dataset`, `mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.dataset.Dataset`, `mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.model_util.dataset.Dataset`, `mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.model_util.quantization.ds.Dataset`, `mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.quantization.ds.Dataset`, `mediapipe_model_maker.object_detector.object_detector.classifier.dataset.Dataset`, `mediapipe_model_maker.object_detector.object_detector.classifier.model_util.dataset.Dataset`, `mediapipe_model_maker.object_detector.object_detector.classifier.model_util.quantization.ds.Dataset`, `mediapipe_model_maker.object_detector.object_detector.ds.classification_dataset.ds.Dataset`, `mediapipe_model_maker.object_detector.object_detector.model_util.dataset.Dataset`, `mediapipe_model_maker.object_detector.object_detector.model_util.quantization.ds.Dataset`, `mediapipe_model_maker.object_detector.object_detector.quantization.ds.Dataset`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.quantization.ds.Dataset(
    tf_dataset: tf.data.Dataset, size: Optional[int] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

For each ML task, such as image classification, text classification etc., a
subclass can be derived from this class to provide task-specific data loading
utilities.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tf_dataset`<a id="tf_dataset"></a>
</td>
<td>
A tf.data.Dataset object that contains a potentially large set
of elements, where each element is a pair of (input_data, target). The
`input_data` means the raw input data, like an image, a text etc., while
the `target` means the ground truth of the raw input data, e.g. the
classification label of the image etc.
</td>
</tr><tr>
<td>
`size`<a id="size"></a>
</td>
<td>
The size of the dataset. tf.data.Dataset donesn't support a function
to get the length directly since it's lazy-loaded and may be infinite.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`size`<a id="size"></a>
</td>
<td>
Returns the size of the dataset.

Note that this function may return None becuase the exact size of the
dataset isn't a necessary parameter to create an instance of this class,
and tf.data.Dataset donesn't support a function to get the length directly
since it's lazy-loaded and may be infinite.
In most cases, however, when an instance of this class is created by helper
functions like 'from_folder', the size of the dataset will be preprocessed,
and this function can return an int representing the size of the dataset.
</td>
</tr>
</table>



## Methods

<h3 id="gen_tf_dataset"><code>gen_tf_dataset</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/data/dataset.py#L69-L117">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>gen_tf_dataset(
    batch_size: int = 1,
    is_training: bool = False,
    shuffle: bool = False,
    preprocess: Optional[Callable[..., Any]] = None,
    drop_remainder: bool = False
) -> tf.data.Dataset
</code></pre>

Generates a batched tf.data.Dataset for training/evaluation.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`batch_size`
</td>
<td>
An integer, the returned dataset will be batched by this size.
</td>
</tr><tr>
<td>
`is_training`
</td>
<td>
A boolean, when True, the returned dataset will be optionally
shuffled and repeated as an endless dataset.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
A boolean, when True, the returned dataset will be shuffled to
create randomness during model training.
</td>
</tr><tr>
<td>
`preprocess`
</td>
<td>
A function taking three arguments in order, feature, label and
boolean is_training.
</td>
</tr><tr>
<td>
`drop_remainder`
</td>
<td>
boolean, whether the finaly batch drops remainder.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A TF dataset ready to be consumed by Keras model.
</td>
</tr>

</table>



<h3 id="split"><code>split</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/data/dataset.py#L126-L138">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>split(
    fraction: float
) -> Tuple[_DatasetT, _DatasetT]
</code></pre>

Splits dataset into two sub-datasets with the given fraction.

Primarily used for splitting the data set into training and testing sets.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`fraction`
</td>
<td>
A float value defines the fraction of the first returned
subdataset in the original data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The splitted two sub datasets.
</td>
</tr>

</table>



<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/data/dataset.py#L119-L124">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>

Returns the number of element of the dataset.




