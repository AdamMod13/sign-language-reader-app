description: Gets the label map from a COCO formatted dataset directory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.dataset_util.get_label_map_coco" />
<meta itemprop="path" content="Stable" />
</div>

# mediapipe_model_maker.object_detector.dataset_util.get_label_map_coco

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/dataset_util.py#L248-L286">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the label map from a COCO formatted dataset directory.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.dataset.dataset_util.get_label_map_coco`, `mediapipe_model_maker.object_detector.object_detector.ds.dataset_util.get_label_map_coco`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.dataset_util.get_label_map_coco(
    data_dir: str
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note that id 0 is reserved for the background class. If id=0 is set, it needs
to be set to "background". It is optional to include id=0 if it is unused, and
it will be automatically added by this method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data_dir`<a id="data_dir"></a>
</td>
<td>
Path of the dataset directory
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
label_map dictionary of the format {<id>:<label_name>}
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
If the label_name for id 0 is set to something other than
the "background" class.
</td>
</tr>
</table>

