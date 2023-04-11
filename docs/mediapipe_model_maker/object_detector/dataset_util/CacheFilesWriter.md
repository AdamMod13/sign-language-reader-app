description: CacheFilesWriter class to write the cached files.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.dataset_util.CacheFilesWriter" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="write_files"/>
</div>

# mediapipe_model_maker.object_detector.dataset_util.CacheFilesWriter

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/dataset_util.py#L192-L245">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



CacheFilesWriter class to write the cached files.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.dataset.dataset_util.CacheFilesWriter`, `mediapipe_model_maker.object_detector.object_detector.ds.dataset_util.CacheFilesWriter`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.dataset_util.CacheFilesWriter(
    label_map: Dict[int, str], max_num_images: Optional[int] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`label_map`<a id="label_map"></a>
</td>
<td>
Dict, map label integer ids to string label names such as {1:
'person', 2: 'notperson'}. 0 is the reserved key for `background` and
doesn't need to be included in `label_map`. Label names can't be
duplicated.
</td>
</tr><tr>
<td>
`max_num_images`<a id="max_num_images"></a>
</td>
<td>
Max number of images to process. If None, process all the
images.
</td>
</tr>
</table>



## Methods

<h3 id="write_files"><code>write_files</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/dataset_util.py#L211-L241">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>write_files(
    cache_files: <a href="../../../mediapipe_model_maker/object_detector/dataset_util/CacheFiles.md"><code>mediapipe_model_maker.object_detector.dataset_util.CacheFiles</code></a>,
    *args,
    **kwargs
) -> None
</code></pre>

Writes TFRecord and meta_data files.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`cache_files`
</td>
<td>
CacheFiles object including a list of TFRecord files and the
meta data yaml file to save the meta_data including data size and
label_map.
</td>
</tr><tr>
<td>
`*args`
</td>
<td>
Non-keyword of parameters used in the `_get_example` method.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Keyword parameters used in the `_get_example` method.
</td>
</tr>
</table>





