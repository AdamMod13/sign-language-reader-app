description: Configurable options for building image classifier.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.image_classifier.ImageClassifierOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="hparams"/>
<meta itemprop="property" content="model_options"/>
</div>

# mediapipe_model_maker.image_classifier.ImageClassifierOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/image_classifier/image_classifier_options.py#L24-L35">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Configurable options for building image classifier.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.image_classifier.ImageClassifierOptions(
    supported_model: <a href="../../mediapipe_model_maker/image_classifier/SupportedModels.md"><code>mediapipe_model_maker.image_classifier.SupportedModels</code></a>,
    model_options: Optional[<a href="../../mediapipe_model_maker/image_classifier/ModelOptions.md"><code>mediapipe_model_maker.image_classifier.ModelOptions</code></a>] = None,
    hparams: Optional[<a href="../../mediapipe_model_maker/image_classifier/HParams.md"><code>mediapipe_model_maker.image_classifier.HParams</code></a>] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`supported_model`<a id="supported_model"></a>
</td>
<td>
A model from the SupportedModels enum.
</td>
</tr><tr>
<td>
`model_options`<a id="model_options"></a>
</td>
<td>
A set of options for configuring the selected model.
</td>
</tr><tr>
<td>
`hparams`<a id="hparams"></a>
</td>
<td>
A set of hyperparameters used to train the image classifier.
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
hparams<a id="hparams"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
model_options<a id="model_options"></a>
</td>
<td>
`None`
</td>
</tr>
</table>

