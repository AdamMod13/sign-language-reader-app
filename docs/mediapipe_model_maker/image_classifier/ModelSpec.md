description: Specification of image classifier model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.image_classifier.ModelSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="mean_rgb"/>
<meta itemprop="property" content="stddev_rgb"/>
</div>

# mediapipe_model_maker.image_classifier.ModelSpec

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/image_classifier/model_spec.py#L21-L43">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Specification of image classifier model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.image_classifier.ModelSpec(
    uri: str,
    input_image_shape: Optional[List[int]] = None,
    name: str = &#x27;&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`uri`<a id="uri"></a>
</td>
<td>
str, URI to the pretrained model.
</td>
</tr><tr>
<td>
`input_image_shape`<a id="input_image_shape"></a>
</td>
<td>
list of int, input image shape. Default: [224, 224].
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
str, model spec name.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
mean_rgb<a id="mean_rgb"></a>
</td>
<td>
`[0.0]`
</td>
</tr><tr>
<td>
stddev_rgb<a id="stddev_rgb"></a>
</td>
<td>
`[255.0]`
</td>
</tr>
</table>

