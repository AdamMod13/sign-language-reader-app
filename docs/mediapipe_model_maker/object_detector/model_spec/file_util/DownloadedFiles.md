description: File(s) that are downloaded from a url into a local directory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.model_spec.file_util.DownloadedFiles" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_path"/>
<meta itemprop="property" content="is_folder"/>
</div>

# mediapipe_model_maker.object_detector.model_spec.file_util.DownloadedFiles

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/file_util.py#L28-L97">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



File(s) that are downloaded from a url into a local directory.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.model.ms.file_util.DownloadedFiles`, `mediapipe_model_maker.object_detector.object_detector.model_lib.ms.file_util.DownloadedFiles`, `mediapipe_model_maker.object_detector.object_detector.ms.file_util.DownloadedFiles`, `mediapipe_model_maker.object_detector.object_detector.object_detector_options.model_spec.file_util.DownloadedFiles`, `mediapipe_model_maker.object_detector.object_detector.preprocessor.ms.file_util.DownloadedFiles`, `mediapipe_model_maker.object_detector.object_detector_options.model_spec.file_util.DownloadedFiles`, `mediapipe_model_maker.object_detector.preprocessor.ms.file_util.DownloadedFiles`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.model_spec.file_util.DownloadedFiles(
    path: str, url: str, is_folder: bool = False
)
</code></pre>



<!-- Placeholder for "Used in" -->

If `is_folder` is True:
  1. `path` should be a folder
  2. `url` should point to a .tar.gz file which contains a single folder at
    the root level.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`path`<a id="path"></a>
</td>
<td>
Relative path in local directory.
</td>
</tr><tr>
<td>
`url`<a id="url"></a>
</td>
<td>
GCS url to download the file(s).
</td>
</tr><tr>
<td>
`is_folder`<a id="is_folder"></a>
</td>
<td>
Whether the path and url represents a folder.
</td>
</tr>
</table>



## Methods

<h3 id="get_path"><code>get_path</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/file_util.py#L47-L97">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_path() -> str
</code></pre>

Gets the path of files saved in a local directory.

If the path doesn't exist, this method will download the file(s) from the
provided url. The path is not cleaned up so it can be reused for subsequent
calls to the same path.
Folders are expected to be zipped in a .tar.gz file which will be extracted
into self.path in the local directory.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If the extracted folder does not have a singular root
directory.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The absolute path to the downloaded file(s)
</td>
</tr>

</table>



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
is_folder<a id="is_folder"></a>
</td>
<td>
`False`
</td>
</tr>
</table>

