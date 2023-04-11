description: Creates an object of CacheFiles class using a COCO formatted dataset.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.dataset_util.get_cache_files_coco" />
<meta itemprop="path" content="Stable" />
</div>

# mediapipe_model_maker.object_detector.dataset_util.get_cache_files_coco

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/dataset_util.py#L120-L152">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Creates an object of CacheFiles class using a COCO formatted dataset.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.dataset.dataset_util.get_cache_files_coco`, `mediapipe_model_maker.object_detector.object_detector.ds.dataset_util.get_cache_files_coco`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.dataset_util.get_cache_files_coco(
    data_dir: str, cache_dir: str
) -> <a href="../../../mediapipe_model_maker/object_detector/dataset_util/CacheFiles.md"><code>mediapipe_model_maker.object_detector.dataset_util.CacheFiles</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data_dir`<a id="data_dir"></a>
</td>
<td>
Folder path of the coco dataset
</td>
</tr><tr>
<td>
`cache_dir`<a id="cache_dir"></a>
</td>
<td>
Folder path of the cache location. When cache_dir is None, a
temporary folder will be created and will not be removed automatically
after training which makes it can be used later.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An object of CacheFiles class.
</td>
</tr>

</table>

