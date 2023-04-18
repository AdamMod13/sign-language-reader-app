description: Specification of object detector model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.ModelSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="mean_norm"/>
<meta itemprop="property" content="mean_rgb"/>
<meta itemprop="property" content="stddev_norm"/>
<meta itemprop="property" content="stddev_rgb"/>
</div>

# mediapipe_model_maker.object_detector.ModelSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model_spec.py#L30-L41">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Specification of object detector model.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.model.ms.ModelSpec`, `mediapipe_model_maker.object_detector.model_spec.ModelSpec`, `mediapipe_model_maker.object_detector.object_detector.model_lib.ms.ModelSpec`, `mediapipe_model_maker.object_detector.object_detector.ms.ModelSpec`, `mediapipe_model_maker.object_detector.object_detector.object_detector_options.model_spec.ModelSpec`, `mediapipe_model_maker.object_detector.object_detector.preprocessor.ms.ModelSpec`, `mediapipe_model_maker.object_detector.object_detector_options.model_spec.ModelSpec`, `mediapipe_model_maker.object_detector.preprocessor.ms.ModelSpec`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.ModelSpec(
    downloaded_files: <a href="../../mediapipe_model_maker/object_detector/model_spec/file_util/DownloadedFiles.md"><code>mediapipe_model_maker.object_detector.model_spec.file_util.DownloadedFiles</code></a>,
    input_image_shape: List[int]
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`downloaded_files`<a id="downloaded_files"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`input_image_shape`<a id="input_image_shape"></a>
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
mean_norm<a id="mean_norm"></a>
</td>
<td>
`(0.5,)`
</td>
</tr><tr>
<td>
mean_rgb<a id="mean_rgb"></a>
</td>
<td>
`(127.5,)`
</td>
</tr><tr>
<td>
stddev_norm<a id="stddev_norm"></a>
</td>
<td>
`(0.5,)`
</td>
</tr><tr>
<td>
stddev_rgb<a id="stddev_rgb"></a>
</td>
<td>
`(127.5,)`
</td>
</tr>
</table>

