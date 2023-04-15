description: Utilities for Object Detector Dataset Library.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.dataset_util" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="META_DATA_FILE_SUFFIX"/>
</div>

# Module: mediapipe_model_maker.object_detector.dataset_util

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/dataset_util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Utilities for Object Detector Dataset Library.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.dataset.dataset_util`, `mediapipe_model_maker.object_detector.object_detector.ds.dataset_util`</p>
</p>
</section>



## Classes

[`class COCOCacheFilesWriter`](../../mediapipe_model_maker/object_detector/dataset_util/COCOCacheFilesWriter.md): CacheFilesWriter class to write the cached files for COCO data.

[`class CacheFiles`](../../mediapipe_model_maker/object_detector/dataset_util/CacheFiles.md): Cache files for object detection.

[`class CacheFilesWriter`](../../mediapipe_model_maker/object_detector/dataset_util/CacheFilesWriter.md): CacheFilesWriter class to write the cached files.

[`class PascalVocCacheFilesWriter`](../../mediapipe_model_maker/object_detector/dataset_util/PascalVocCacheFilesWriter.md): CacheFilesWriter class to write the cached files for PASCAL VOC data.

## Functions

[`get_cache_files_coco(...)`](../../mediapipe_model_maker/object_detector/dataset_util/get_cache_files_coco.md): Creates an object of CacheFiles class using a COCO formatted dataset.

[`get_cache_files_pascal_voc(...)`](../../mediapipe_model_maker/object_detector/dataset_util/get_cache_files_pascal_voc.md): Gets an object of CacheFiles using a PASCAL VOC formatted dataset.

[`get_label_map_coco(...)`](../../mediapipe_model_maker/object_detector/dataset_util/get_label_map_coco.md): Gets the label map from a COCO formatted dataset directory.

[`get_label_map_pascal_voc(...)`](../../mediapipe_model_maker/object_detector/dataset_util/get_label_map_pascal_voc.md): Gets the label map from a PASCAL VOC formatted dataset directory.

[`is_cached(...)`](../../mediapipe_model_maker/object_detector/dataset_util/is_cached.md): Checks whether cache files are already cached.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
META_DATA_FILE_SUFFIX<a id="META_DATA_FILE_SUFFIX"></a>
</td>
<td>
`'_meta_data.yaml'`
</td>
</tr>
</table>

