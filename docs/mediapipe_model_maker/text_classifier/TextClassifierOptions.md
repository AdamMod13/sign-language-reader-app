description: User-facing options for creating the text classifier.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.text_classifier.TextClassifierOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="hparams"/>
<meta itemprop="property" content="model_options"/>
</div>

# mediapipe_model_maker.text_classifier.TextClassifierOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/text/text_classifier/text_classifier_options.py#L24-L38">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



User-facing options for creating the text classifier.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.text_classifier.TextClassifierOptions(
    supported_model: <a href="../../mediapipe_model_maker/text_classifier/SupportedModels.md"><code>mediapipe_model_maker.text_classifier.SupportedModels</code></a>,
    hparams: Optional[<a href="../../mediapipe_model_maker/text_classifier/HParams.md"><code>mediapipe_model_maker.text_classifier.HParams</code></a>] = None,
    model_options: Optional[mo.TextClassifierModelOptions] = None
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
A preconfigured model spec.
</td>
</tr><tr>
<td>
`hparams`<a id="hparams"></a>
</td>
<td>
Training hyperparameters the user can set to override the ones in
`supported_model`.
</td>
</tr><tr>
<td>
`model_options`<a id="model_options"></a>
</td>
<td>
Model options the user can set to override the ones in
`supported_model`. The model options type should be consistent with the
architecture of the `supported_model`.
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

