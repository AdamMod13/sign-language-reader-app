description: Configurable model options for a BERT model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.text_classifier.BertModelOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="do_fine_tuning"/>
<meta itemprop="property" content="dropout_rate"/>
<meta itemprop="property" content="seq_len"/>
</div>

# mediapipe_model_maker.text_classifier.BertModelOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/text/core/bert_model_options.py#L19-L33">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Configurable model options for a BERT model.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.text_classifier.BertModelOptions(
    seq_len: int = 128, do_fine_tuning: bool = True, dropout_rate: float = 0.1
)
</code></pre>



<!-- Placeholder for "Used in" -->

See https://arxiv.org/abs/1810.04805 (BERT: Pre-training of Deep Bidirectional
Transformers for Language Understanding) for more details.

  Attributes:
    seq_len: Length of the sequence to feed into the model.
    do_fine_tuning: If true, then the BERT model is not frozen for training.
    dropout_rate: The rate for dropout.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`seq_len`<a id="seq_len"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`do_fine_tuning`<a id="do_fine_tuning"></a>
</td>
<td>
Dataclass field
</td>
</tr><tr>
<td>
`dropout_rate`<a id="dropout_rate"></a>
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
do_fine_tuning<a id="do_fine_tuning"></a>
</td>
<td>
`True`
</td>
</tr><tr>
<td>
dropout_rate<a id="dropout_rate"></a>
</td>
<td>
`0.1`
</td>
</tr><tr>
<td>
seq_len<a id="seq_len"></a>
</td>
<td>
`128`
</td>
</tr>
</table>

