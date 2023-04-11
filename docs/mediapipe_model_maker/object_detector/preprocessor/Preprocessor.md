description: Preprocessor for object detector.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.preprocessor.Preprocessor" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
</div>

# mediapipe_model_maker.object_detector.preprocessor.Preprocessor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/preprocessor.py#L27-L163">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Preprocessor for object detector.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.preprocessor.Preprocessor`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.preprocessor.Preprocessor(
    model_spec: <a href="../../../mediapipe_model_maker/object_detector/ModelSpec.md"><code>mediapipe_model_maker.object_detector.ModelSpec</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/preprocessor.py#L47-L163">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    data: Mapping[str, Any], is_training: bool = True
) -> Tuple[tf.Tensor, Mapping[str, Any]]
</code></pre>

Run the preprocessor on an example.

The data dict should contain the following keys always:
  - image
  - groundtruth_classes
  - groundtruth_boxes
  - groundtruth_is_crowd
Additional keys needed when is_training is set to True:
  - groundtruth_area
  - source_id
  - height
  - width

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data`
</td>
<td>
A dict of object detector inputs.
</td>
</tr><tr>
<td>
`is_training`
</td>
<td>
Whether or not the data is used for training.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A tuple of (image, labels) where image is a Tensor and labels is a dict.
</td>
</tr>

</table>





