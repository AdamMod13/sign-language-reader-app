description: Configurable options for gesture recognizer model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.gesture_recognizer.ModelOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="dropout_rate"/>
</div>

# mediapipe_model_maker.gesture_recognizer.ModelOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/gesture_recognizer/model_options.py#L20-L33">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Configurable options for gesture recognizer model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.gesture_recognizer.ModelOptions(
    dropout_rate: float = 0.05,
    layer_widths: List[int] = dataclasses.field(default_factory=list)
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dropout_rate`<a id="dropout_rate"></a>
</td>
<td>
The fraction of the input units to drop, used in dropout
layer.
</td>
</tr><tr>
<td>
`layer_widths`<a id="layer_widths"></a>
</td>
<td>
A list of hidden layer widths for the gesture model. Each
element in the list will create a new hidden layer with the specified
width. The hidden layers are separated with BatchNorm, Dropout, and ReLU.
Defaults to an empty list(no hidden layers).
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
dropout_rate<a id="dropout_rate"></a>
</td>
<td>
`0.05`
</td>
</tr>
</table>

