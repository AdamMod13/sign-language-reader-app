description: A dataclass wraps the hand data preprocessing hyperparameters.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.gesture_recognizer.HandDataPreprocessingParams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="min_detection_confidence"/>
<meta itemprop="property" content="shuffle"/>
</div>

# mediapipe_model_maker.gesture_recognizer.HandDataPreprocessingParams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/gesture_recognizer/dataset.py#L37-L46">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A dataclass wraps the hand data preprocessing hyperparameters.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.gesture_recognizer.HandDataPreprocessingParams(
    shuffle: bool = True, min_detection_confidence: float = 0.7
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
A boolean controlling if shuffle the dataset. Default to true.
</td>
</tr><tr>
<td>
`min_detection_confidence`<a id="min_detection_confidence"></a>
</td>
<td>
confidence threshold for hand detection.
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
min_detection_confidence<a id="min_detection_confidence"></a>
</td>
<td>
`0.7`
</td>
</tr><tr>
<td>
shuffle<a id="shuffle"></a>
</td>
<td>
`True`
</td>
</tr>
</table>

