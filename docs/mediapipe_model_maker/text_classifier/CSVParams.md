description: Parameters used when reading a CSV file.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.text_classifier.CSVParams" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="delimiter"/>
<meta itemprop="property" content="fieldnames"/>
<meta itemprop="property" content="quotechar"/>
</div>

# mediapipe_model_maker.text_classifier.CSVParams

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/text/text_classifier/dataset.py#L26-L43">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Parameters used when reading a CSV file.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.text_classifier.CSVParams(
    text_column: str,
    label_column: str,
    fieldnames: Optional[Sequence[str]] = None,
    delimiter: str = &#x27;,&#x27;,
    quotechar: str = &#x27;\&#x27;&quot;
)
</code></pre>



<!-- Placeholder for "Used in" -->




<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`text_column`<a id="text_column"></a>
</td>
<td>
Column name for the input text.
</td>
</tr><tr>
<td>
`label_column`<a id="label_column"></a>
</td>
<td>
Column name for the labels.
</td>
</tr><tr>
<td>
`fieldnames`<a id="fieldnames"></a>
</td>
<td>
Sequence of keys for the CSV columns. If None, the first row of
the CSV file is used as the keys.
</td>
</tr><tr>
<td>
`delimiter`<a id="delimiter"></a>
</td>
<td>
Character that separates fields.
</td>
</tr><tr>
<td>
`quotechar`<a id="quotechar"></a>
</td>
<td>
Character used to quote fields that contain special characters
like the `delimiter`.
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
delimiter<a id="delimiter"></a>
</td>
<td>
`','`
</td>
</tr><tr>
<td>
fieldnames<a id="fieldnames"></a>
</td>
<td>
`None`
</td>
</tr><tr>
<td>
quotechar<a id="quotechar"></a>
</td>
<td>
`'"'`
</td>
</tr>
</table>

