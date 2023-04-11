description: Gets the label map from a PASCAL VOC formatted dataset directory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.dataset_util.get_label_map_pascal_voc" />
<meta itemprop="path" content="Stable" />
</div>

# mediapipe_model_maker.object_detector.dataset_util.get_label_map_pascal_voc

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/dataset_util.py#L289-L314">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Gets the label map from a PASCAL VOC formatted dataset directory.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.dataset.dataset_util.get_label_map_pascal_voc`, `mediapipe_model_maker.object_detector.object_detector.ds.dataset_util.get_label_map_pascal_voc`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.dataset_util.get_label_map_pascal_voc(
    data_dir: str
)
</code></pre>



<!-- Placeholder for "Used in" -->

The id to label_name mapping is determined by sorting all label_names and
numbering them starting from 1. Id=0 is set as the 'background' class.

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

