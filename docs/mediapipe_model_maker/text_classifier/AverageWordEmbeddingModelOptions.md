description: Configurable model options for an Average Word Embedding classifier.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.text_classifier.AverageWordEmbeddingModelOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="do_lower_case"/>
<meta itemprop="property" content="dropout_rate"/>
<meta itemprop="property" content="seq_len"/>
<meta itemprop="property" content="vocab_size"/>
<meta itemprop="property" content="wordvec_dim"/>
</div>

# mediapipe_model_maker.text_classifier.AverageWordEmbeddingModelOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/text/text_classifier/model_options.py#L25-L41">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Configurable model options for an Average Word Embedding classifier.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.text_classifier.AverageWordEmbeddingModelOptions(
    seq_len: int = 256,
    wordvec_dim: int = 16,
    do_lower_case: bool = True,
    vocab_size: int = 10000,
    dropout_rate: float = 0.2
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`seq_len`<a id="seq_len"></a>
</td>
<td>
Length of the sequence to feed into the model.
</td>
</tr><tr>
<td>
`wordvec_dim`<a id="wordvec_dim"></a>
</td>
<td>
Dimension of the word embedding.
</td>
</tr><tr>
<td>
`do_lower_case`<a id="do_lower_case"></a>
</td>
<td>
Whether to convert all uppercase characters to lowercase
during preprocessing.
</td>
</tr><tr>
<td>
`vocab_size`<a id="vocab_size"></a>
</td>
<td>
Number of words to generate the vocabulary from data.
</td>
</tr><tr>
<td>
`dropout_rate`<a id="dropout_rate"></a>
</td>
<td>
The rate for dropout.
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
do_lower_case<a id="do_lower_case"></a>
</td>
<td>
`True`
</td>
</tr><tr>
<td>
dropout_rate<a id="dropout_rate"></a>
</td>
<td>
`0.2`
</td>
</tr><tr>
<td>
seq_len<a id="seq_len"></a>
</td>
<td>
`256`
</td>
</tr><tr>
<td>
vocab_size<a id="vocab_size"></a>
</td>
<td>
`10000`
</td>
</tr><tr>
<td>
wordvec_dim<a id="wordvec_dim"></a>
</td>
<td>
`16`
</td>
</tr>
</table>

