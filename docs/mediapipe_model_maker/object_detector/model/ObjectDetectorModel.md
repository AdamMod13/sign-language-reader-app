description: An object detector model which can be trained using Model Maker's training API.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.model.ObjectDetectorModel" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="add_loss"/>
<meta itemprop="property" content="add_metric"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="build_from_config"/>
<meta itemprop="property" content="call"/>
<meta itemprop="property" content="compile"/>
<meta itemprop="property" content="compile_from_config"/>
<meta itemprop="property" content="compute_loss"/>
<meta itemprop="property" content="compute_mask"/>
<meta itemprop="property" content="compute_metrics"/>
<meta itemprop="property" content="compute_output_shape"/>
<meta itemprop="property" content="convert_to_qat"/>
<meta itemprop="property" content="count_params"/>
<meta itemprop="property" content="evaluate"/>
<meta itemprop="property" content="export"/>
<meta itemprop="property" content="export_saved_model"/>
<meta itemprop="property" content="fit"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_build_config"/>
<meta itemprop="property" content="get_compile_config"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="get_layer"/>
<meta itemprop="property" content="get_metrics_result"/>
<meta itemprop="property" content="get_weight_paths"/>
<meta itemprop="property" content="get_weights"/>
<meta itemprop="property" content="load_checkpoint"/>
<meta itemprop="property" content="load_weights"/>
<meta itemprop="property" content="make_predict_function"/>
<meta itemprop="property" content="make_test_function"/>
<meta itemprop="property" content="make_train_function"/>
<meta itemprop="property" content="predict"/>
<meta itemprop="property" content="predict_on_batch"/>
<meta itemprop="property" content="predict_step"/>
<meta itemprop="property" content="reset_metrics"/>
<meta itemprop="property" content="reset_states"/>
<meta itemprop="property" content="save"/>
<meta itemprop="property" content="save_checkpoint"/>
<meta itemprop="property" content="save_spec"/>
<meta itemprop="property" content="save_weights"/>
<meta itemprop="property" content="set_weights"/>
<meta itemprop="property" content="summary"/>
<meta itemprop="property" content="test_on_batch"/>
<meta itemprop="property" content="test_step"/>
<meta itemprop="property" content="to_json"/>
<meta itemprop="property" content="to_yaml"/>
<meta itemprop="property" content="train_on_batch"/>
<meta itemprop="property" content="train_step"/>
<meta itemprop="property" content="with_name_scope"/>
</div>

# mediapipe_model_maker.object_detector.model.ObjectDetectorModel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L35-L355">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An object detector model which can be trained using Model Maker's training API.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.model_lib.ObjectDetectorModel`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>mediapipe_model_maker.object_detector.model.ObjectDetectorModel(
    model_spec: <a href="../../../mediapipe_model_maker/object_detector/ModelSpec.md"><code>mediapipe_model_maker.object_detector.ModelSpec</code></a>,
    model_options: <a href="../../../mediapipe_model_maker/object_detector/ModelOptions.md"><code>mediapipe_model_maker.object_detector.ModelOptions</code></a>,
    num_classes: int
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`model_spec`<a id="model_spec"></a>
</td>
<td>
Specification for the model.
</td>
</tr><tr>
<td>
`model_options`<a id="model_options"></a>
</td>
<td>
Model options for creating the model.
</td>
</tr><tr>
<td>
`num_classes`<a id="num_classes"></a>
</td>
<td>
Number of classes for object detection.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`loss_trackers`<a id="loss_trackers"></a>
</td>
<td>
List of tf.keras.metrics.Mean objects used to track the loss
during training.
</td>
</tr><tr>
<td>
`activity_regularizer`<a id="activity_regularizer"></a>
</td>
<td>
Optional regularizer function for the output of this layer.
</td>
</tr><tr>
<td>
`compute_dtype`<a id="compute_dtype"></a>
</td>
<td>
The dtype of the layer's computations.

This is equivalent to `Layer.dtype_policy.compute_dtype`. Unless
mixed precision is used, this is the same as `Layer.dtype`, the dtype of
the weights.

Layers automatically cast their inputs to the compute dtype, which
causes computations and the output to be in the compute dtype as well.
This is done by the base Layer class in `Layer.__call__`, so you do not
have to insert these casts if implementing your own layer.

Layers often perform certain internal computations in higher precision
when `compute_dtype` is float16 or bfloat16 for numeric stability. The
output will still typically be float16 or bfloat16 in such cases.
</td>
</tr><tr>
<td>
`distribute_reduction_method`<a id="distribute_reduction_method"></a>
</td>
<td>
The method employed to reduce per-replica values during training.

Unless specified, the value "auto" will be assumed, indicating that
the reduction strategy should be chosen based on the current
running environment.
See `reduce_per_replica` function for more details.
</td>
</tr><tr>
<td>
`distribute_strategy`<a id="distribute_strategy"></a>
</td>
<td>
The `tf.distribute.Strategy` this model was created under.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The dtype of the layer weights.

This is equivalent to `Layer.dtype_policy.variable_dtype`. Unless
mixed precision is used, this is the same as `Layer.compute_dtype`, the
dtype of the layer's computations.
</td>
</tr><tr>
<td>
`dtype_policy`<a id="dtype_policy"></a>
</td>
<td>
The dtype policy associated with this layer.

This is an instance of a `tf.keras.mixed_precision.Policy`.
</td>
</tr><tr>
<td>
`dynamic`<a id="dynamic"></a>
</td>
<td>
Whether the layer is dynamic (eager-only); set in the constructor.
</td>
</tr><tr>
<td>
`input`<a id="input"></a>
</td>
<td>
Retrieves the input tensor(s) of a layer.

Only applicable if the layer has exactly one input,
i.e. if it is connected to one incoming layer.
</td>
</tr><tr>
<td>
`input_spec`<a id="input_spec"></a>
</td>
<td>
`InputSpec` instance(s) describing the input format for this layer.

When you create a layer subclass, you can set `self.input_spec` to
enable the layer to run input compatibility checks when it is called.
Consider a `Conv2D` layer: it can only be called on a single input
tensor of rank 4. As such, you can set, in `__init__()`:

```python
self.input_spec = tf.keras.layers.InputSpec(ndim=4)
```

Now, if you try to call the layer on an input that isn't rank 4
(for instance, an input of shape `(2,)`, it will raise a
nicely-formatted error:

```
ValueError: Input 0 of layer conv2d is incompatible with the layer:
expected ndim=4, found ndim=1. Full shape received: [2]
```

Input checks that can be specified via `input_spec` include:
- Structure (e.g. a single input, a list of 2 inputs, etc)
- Shape
- Rank (ndim)
- Dtype

For more information, see `tf.keras.layers.InputSpec`.
</td>
</tr><tr>
<td>
`jit_compile`<a id="jit_compile"></a>
</td>
<td>
Specify whether to compile the model with XLA.

[XLA](https://www.tensorflow.org/xla) is an optimizing compiler
for machine learning. `jit_compile` is not enabled by default.
Note that `jit_compile=True` may not necessarily work for all models.

For more information on supported operations please refer to the
[XLA documentation](https://www.tensorflow.org/xla). Also refer to
[known XLA issues](https://www.tensorflow.org/xla/known_issues)
for more details.
</td>
</tr><tr>
<td>
`layers`<a id="layers"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`losses`<a id="losses"></a>
</td>
<td>
List of losses added using the `add_loss()` API.

Variable regularization tensors are created when this property is
accessed, so it is eager safe: accessing `losses` under a
`tf.GradientTape` will propagate gradients back to the corresponding
variables.

```
>>> class MyLayer(tf.keras.layers.Layer):
...   def call(self, inputs):
...     self.add_loss(tf.abs(tf.reduce_mean(inputs)))
...     return inputs
>>> l = MyLayer()
>>> l(np.ones((10, 1)))
>>> l.losses
[1.0]
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> x = tf.keras.layers.Dense(10)(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Activity regularization.
>>> len(model.losses)
0
>>> model.add_loss(tf.abs(tf.reduce_mean(x)))
>>> len(model.losses)
1
```

```
>>> inputs = tf.keras.Input(shape=(10,))
>>> d = tf.keras.layers.Dense(10, kernel_initializer='ones')
>>> x = d(inputs)
>>> outputs = tf.keras.layers.Dense(1)(x)
>>> model = tf.keras.Model(inputs, outputs)
>>> # Weight regularization.
>>> model.add_loss(lambda: tf.reduce_mean(d.kernel))
>>> model.losses
[<tf.Tensor: shape=(), dtype=float32, numpy=1.0>]
```
</td>
</tr><tr>
<td>
`metrics`<a id="metrics"></a>
</td>
<td>
Overrides metrics from tf.keras.Model.
</td>
</tr><tr>
<td>
`metrics_names`<a id="metrics_names"></a>
</td>
<td>
Returns the model's display labels for all outputs.

Note: `metrics_names` are available only after a `keras.Model` has been
trained/evaluated on actual data.

```
>>> inputs = tf.keras.layers.Input(shape=(3,))
>>> outputs = tf.keras.layers.Dense(2)(inputs)
>>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
>>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
>>> model.metrics_names
[]
```

```
>>> x = np.random.random((2, 3))
>>> y = np.random.randint(0, 2, (2, 2))
>>> model.fit(x, y)
>>> model.metrics_names
['loss', 'mae']
```

```
>>> inputs = tf.keras.layers.Input(shape=(3,))
>>> d = tf.keras.layers.Dense(2, name='out')
>>> output_1 = d(inputs)
>>> output_2 = d(inputs)
>>> model = tf.keras.models.Model(
...    inputs=inputs, outputs=[output_1, output_2])
>>> model.compile(optimizer="Adam", loss="mse", metrics=["mae", "acc"])
>>> model.fit(x, (y, y))
>>> model.metrics_names
['loss', 'out_loss', 'out_1_loss', 'out_mae', 'out_acc', 'out_1_mae',
'out_1_acc']
```
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Name of the layer (string), set in the constructor.
</td>
</tr><tr>
<td>
`name_scope`<a id="name_scope"></a>
</td>
<td>
Returns a `tf.name_scope` instance for this class.
</td>
</tr><tr>
<td>
`non_trainable_weights`<a id="non_trainable_weights"></a>
</td>
<td>
List of all non-trainable weights tracked by this layer.

Non-trainable weights are *not* updated during training. They are
expected to be updated manually in `call()`.
</td>
</tr><tr>
<td>
`output`<a id="output"></a>
</td>
<td>
Retrieves the output tensor(s) of a layer.

Only applicable if the layer has exactly one output,
i.e. if it is connected to one incoming layer.
</td>
</tr><tr>
<td>
`run_eagerly`<a id="run_eagerly"></a>
</td>
<td>
Settable attribute indicating whether the model should run eagerly.

Running eagerly means that your model will be run step by step,
like Python code. Your model might run slower, but it should become
easier for you to debug it by stepping into individual layer calls.

By default, we will attempt to compile your model to a static graph to
deliver the best execution performance.
</td>
</tr><tr>
<td>
`submodules`<a id="submodules"></a>
</td>
<td>
Sequence of all sub-modules.

Submodules are modules which are properties of this module, or found as
properties of modules which are properties of this module (and so on).

```
>>> a = tf.Module()
>>> b = tf.Module()
>>> c = tf.Module()
>>> a.b = b
>>> b.c = c
>>> list(a.submodules) == [b, c]
True
>>> list(b.submodules) == [c]
True
>>> list(c.submodules) == []
True
```
</td>
</tr><tr>
<td>
`supports_masking`<a id="supports_masking"></a>
</td>
<td>
Whether this layer supports computing a mask using `compute_mask`.
</td>
</tr><tr>
<td>
`trainable`<a id="trainable"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`trainable_weights`<a id="trainable_weights"></a>
</td>
<td>
List of all trainable weights tracked by this layer.

Trainable weights are updated via gradient descent during training.
</td>
</tr><tr>
<td>
`variable_dtype`<a id="variable_dtype"></a>
</td>
<td>
Alias of `Layer.dtype`, the dtype of the weights.
</td>
</tr><tr>
<td>
`weights`<a id="weights"></a>
</td>
<td>
Returns the list of all layer variables/weights.

Note: This will not track the weights of nested `tf.Modules` that are
not themselves Keras layers.
</td>
</tr>
</table>



## Methods

<h3 id="add_loss"><code>add_loss</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_loss(
    losses, **kwargs
)
</code></pre>

Add loss tensor(s), potentially dependent on layer inputs.

Some losses (for instance, activity regularization losses) may be
dependent on the inputs passed when calling a layer. Hence, when reusing
the same layer on different inputs `a` and `b`, some entries in
`layer.losses` may be dependent on `a` and some on `b`. This method
automatically keeps track of dependencies.

This method can be used inside a subclassed layer or model's `call`
function, in which case `losses` should be a Tensor or list of Tensors.

#### Example:



```python
class MyLayer(tf.keras.layers.Layer):
  def call(self, inputs):
    self.add_loss(tf.abs(tf.reduce_mean(inputs)))
    return inputs
```

The same code works in distributed training: the input to `add_loss()`
is treated like a regularization loss and averaged across replicas
by the training loop (both built-in `Model.fit()` and compliant custom
training loops).

The `add_loss` method can also be called directly on a Functional Model
during construction. In this case, any loss Tensors passed to this Model
must be symbolic and be able to be traced back to the model's `Input`s.
These losses become part of the model's topology and are tracked in
`get_config`.

#### Example:



```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Activity regularization.
model.add_loss(tf.abs(tf.reduce_mean(x)))
```

If this is not the case for your loss (if, for example, your loss
references a `Variable` of one of the model's layers), you can wrap your
loss in a zero-argument lambda. These losses are not tracked as part of
the model's topology since they can't be serialized.

#### Example:



```python
inputs = tf.keras.Input(shape=(10,))
d = tf.keras.layers.Dense(10)
x = d(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
# Weight regularization.
model.add_loss(lambda: tf.reduce_mean(d.kernel))
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`losses`
</td>
<td>
Loss tensor, or list/tuple of tensors. Rather than tensors,
losses may also be zero-argument callables which create a loss
tensor.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Used for backwards compatibility only.
</td>
</tr>
</table>



<h3 id="add_metric"><code>add_metric</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_metric(
    value, name=None, **kwargs
)
</code></pre>

Adds metric tensor to the layer.

This method can be used inside the `call()` method of a subclassed layer
or model.

```python
class MyMetricLayer(tf.keras.layers.Layer):
  def __init__(self):
    super(MyMetricLayer, self).__init__(name='my_metric_layer')
    self.mean = tf.keras.metrics.Mean(name='metric_1')

  def call(self, inputs):
    self.add_metric(self.mean(inputs))
    self.add_metric(tf.reduce_sum(inputs), name='metric_2')
    return inputs
```

This method can also be called directly on a Functional Model during
construction. In this case, any tensor passed to this Model must
be symbolic and be able to be traced back to the model's `Input`s. These
metrics become part of the model's topology and are tracked when you
save the model via `save()`.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(math_ops.reduce_sum(x), name='metric_1')
```

Note: Calling `add_metric()` with the result of a metric object on a
Functional Model, as shown in the example below, is not supported. This
is because we cannot trace the metric result tensor back to the model's
inputs.

```python
inputs = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(10)(inputs)
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
model.add_metric(tf.keras.metrics.Mean()(x), name='metric_1')
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
Metric tensor.
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String metric name.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments for backward compatibility.
Accepted values:
`aggregation` - When the `value` tensor provided is not the result
of calling a `keras.Metric` instance, it will be aggregated by
default using a `keras.Metric.Mean`.
</td>
</tr>
</table>



<h3 id="build"><code>build</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build(
    input_shape
)
</code></pre>

Builds the model based on input shapes received.

This is to be used for subclassed models, which do not know at
instantiation time what their inputs look like.

This method only exists for users who want to call `model.build()` in a
standalone way (as a substitute for calling the model on real data to
build it). It will never be called by the framework (and thus it will
never throw unexpected errors in an unrelated workflow).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_shape`
</td>
<td>
Single tuple, `TensorShape` instance, or list/dict of
shapes, where shapes are tuples, integers, or `TensorShape`
instances.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
  1. In case of invalid user-provided data (not of type tuple,
     list, `TensorShape`, or dict).
  2. If the model requires call arguments that are agnostic
     to the input shapes (positional or keyword arg in call
     signature).
  3. If not all layers were properly built.
  4. If float type inputs are not supported within the layers.

In each of these cases, the user should build their model by calling
it on real tensor data.
</td>
</tr>
</table>



<h3 id="build_from_config"><code>build_from_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build_from_config(
    config
)
</code></pre>




<h3 id="call"><code>call</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L250-L265">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>call(
    images: Union[tf.Tensor, Sequence[tf.Tensor]],
    image_shape: Optional[tf.Tensor] = None,
    anchor_boxes: Optional[Mapping[str, tf.Tensor]] = None,
    output_intermediate_features: bool = False,
    training: bool = None
) -> Mapping[str, tf.Tensor]
</code></pre>

Overrides call from tf.keras.Model.


<h3 id="compile"><code>compile</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compile(
    optimizer=&#x27;rmsprop&#x27;,
    loss=None,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,
    **kwargs
)
</code></pre>

Configures the model for training.


#### Example:



```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.FalseNegatives()])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`optimizer`
</td>
<td>
String (name of optimizer) or optimizer instance. See
`tf.keras.optimizers`.
</td>
</tr><tr>
<td>
`loss`
</td>
<td>
Loss function. May be a string (name of loss function), or
a `tf.keras.losses.Loss` instance. See `tf.keras.losses`. A loss
function is any callable with the signature `loss = fn(y_true,
y_pred)`, where `y_true` are the ground truth values, and
`y_pred` are the model's predictions.
`y_true` should have shape
`(batch_size, d0, .. dN)` (except in the case of
sparse loss functions such as
sparse categorical crossentropy which expects integer arrays of
shape `(batch_size, d0, .. dN-1)`).
`y_pred` should have shape `(batch_size, d0, .. dN)`.
The loss function should return a float tensor.
If a custom `Loss` instance is
used and reduction is set to `None`, return value has shape
`(batch_size, d0, .. dN-1)` i.e. per-sample or per-timestep loss
values; otherwise, it is a scalar. If the model has multiple
outputs, you can use a different loss on each output by passing a
dictionary or a list of losses. The loss value that will be
minimized by the model will then be the sum of all individual
losses, unless `loss_weights` is specified.
</td>
</tr><tr>
<td>
`metrics`
</td>
<td>
List of metrics to be evaluated by the model during
training and testing. Each of this can be a string (name of a
built-in function), function or a `tf.keras.metrics.Metric`
instance. See `tf.keras.metrics`. Typically you will use
`metrics=['accuracy']`.
A function is any callable with the signature `result = fn(y_true,
y_pred)`. To specify different metrics for different outputs of a
multi-output model, you could also pass a dictionary, such as
`metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}`.
You can also pass a list to specify a metric or a list of metrics
for each output, such as
`metrics=[['accuracy'], ['accuracy', 'mse']]`
or `metrics=['accuracy', ['accuracy', 'mse']]`. When you pass the
strings 'accuracy' or 'acc', we convert this to one of
`tf.keras.metrics.BinaryAccuracy`,
`tf.keras.metrics.CategoricalAccuracy`,
`tf.keras.metrics.SparseCategoricalAccuracy` based on the shapes
of the targets and of the model output. We do a similar
conversion for the strings 'crossentropy' and 'ce' as well.
The metrics passed here are evaluated without sample weighting; if
you would like sample weighting to apply, you can specify your
metrics via the `weighted_metrics` argument instead.
</td>
</tr><tr>
<td>
`loss_weights`
</td>
<td>
Optional list or dictionary specifying scalar
coefficients (Python floats) to weight the loss contributions of
different model outputs. The loss value that will be minimized by
the model will then be the *weighted sum* of all individual
losses, weighted by the `loss_weights` coefficients.  If a list,
it is expected to have a 1:1 mapping to the model's outputs. If a
dict, it is expected to map output names (strings) to scalar
coefficients.
</td>
</tr><tr>
<td>
`weighted_metrics`
</td>
<td>
List of metrics to be evaluated and weighted by
`sample_weight` or `class_weight` during training and testing.
</td>
</tr><tr>
<td>
`run_eagerly`
</td>
<td>
Bool. Defaults to `False`. If `True`, this `Model`'s
logic will not be wrapped in a `tf.function`. Recommended to leave
this as `None` unless your `Model` cannot be run inside a
`tf.function`. `run_eagerly=True` is not supported when using
`tf.distribute.experimental.ParameterServerStrategy`.
</td>
</tr><tr>
<td>
`steps_per_execution`
</td>
<td>
Int. Defaults to 1. The number of batches to
run during each `tf.function` call. Running multiple batches
inside a single `tf.function` call can greatly improve performance
on TPUs or small models with a large Python overhead. At most, one
full epoch will be run each execution. If a number larger than the
size of the epoch is passed, the execution will be truncated to
the size of the epoch. Note that if `steps_per_execution` is set
to `N`, `Callback.on_batch_begin` and `Callback.on_batch_end`
methods will only be called every `N` batches (i.e. before/after
each `tf.function` execution).
</td>
</tr><tr>
<td>
`jit_compile`
</td>
<td>
If `True`, compile the model training step with XLA.
[XLA](https://www.tensorflow.org/xla) is an optimizing compiler
for machine learning.
`jit_compile` is not enabled for by default.
Note that `jit_compile=True`
may not necessarily work for all models.
For more information on supported operations please refer to the
[XLA documentation](https://www.tensorflow.org/xla).
Also refer to
[known XLA issues](https://www.tensorflow.org/xla/known_issues)
for more details.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Arguments supported for backwards compatibility only.
</td>
</tr>
</table>



<h3 id="compile_from_config"><code>compile_from_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compile_from_config(
    config
)
</code></pre>




<h3 id="compute_loss"><code>compute_loss</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L267-L319">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_loss(
    x=None, y=None, y_pred=None, sample_weight=None
)
</code></pre>

Overrides compute_loss from tf.keras.Model.


<h3 id="compute_mask"><code>compute_mask</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_mask(
    inputs, mask=None
)
</code></pre>

Computes an output mask tensor.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`inputs`
</td>
<td>
Tensor or list of tensors.
</td>
</tr><tr>
<td>
`mask`
</td>
<td>
Tensor or list of tensors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
None or a tensor (or list of tensors,
one per output tensor of the layer).
</td>
</tr>

</table>



<h3 id="compute_metrics"><code>compute_metrics</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L326-L328">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_metrics(
    x, y, y_pred, sample_weight=None
)
</code></pre>

Overrides compute_metrics from tf.keras.Model.


<h3 id="compute_output_shape"><code>compute_output_shape</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compute_output_shape(
    input_shape
)
</code></pre>

Computes the output shape of the layer.

This method will cause the layer's state to be built, if that has not
happened before. This requires that the layer will later be used with
inputs that match the input shape provided here.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`input_shape`
</td>
<td>
Shape tuple (tuple of integers) or `tf.TensorShape`,
or structure of shape tuples / `tf.TensorShape` instances
(one per output tensor of the layer).
Shape tuples can include None for free dimensions,
instead of an integer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `tf.TensorShape` instance
or structure of `tf.TensorShape` instances.
</td>
</tr>

</table>



<h3 id="convert_to_qat"><code>convert_to_qat</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L163-L176">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert_to_qat() -> None
</code></pre>

Converts the model to a QAT RetinaNet model.


<h3 id="count_params"><code>count_params</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>count_params()
</code></pre>

Count the total number of scalars composing the weights.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
An integer count.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if the layer isn't yet built
(in which case its weights aren't yet defined).
</td>
</tr>
</table>



<h3 id="evaluate"><code>evaluate</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>evaluate(
    x=None,
    y=None,
    batch_size=None,
    verbose=&#x27;auto&#x27;,
    sample_weight=None,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    return_dict=False,
    **kwargs
)
</code></pre>

Returns the loss value & metrics values for the model in test mode.

Computation is done in batches (see the `batch_size` arg.)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
Input data. It could be:
- A Numpy array (or array-like), or a list of arrays
  (in case the model has multiple inputs).
- A TensorFlow tensor, or a list of tensors
  (in case the model has multiple inputs).
- A dict mapping input names to the corresponding array/tensors,
  if the model has named inputs.
- A `tf.data` dataset. Should return a tuple
  of either `(inputs, targets)` or
  `(inputs, targets, sample_weights)`.
- A generator or `keras.utils.Sequence` returning `(inputs,
  targets)` or `(inputs, targets, sample_weights)`.
A more detailed description of unpacking behavior for iterator
types (Dataset, generator, Sequence) is given in the `Unpacking
behavior for iterator-like inputs` section of `Model.fit`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
Target data. Like the input data `x`, it could be either Numpy
array(s) or TensorFlow tensor(s). It should be consistent with `x`
(you cannot have Numpy inputs and tensor targets, or inversely).
If `x` is a dataset, generator or `keras.utils.Sequence` instance,
`y` should not be specified (since targets will be obtained from
the iterator/dataset).
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Integer or `None`. Number of samples per batch of
computation. If unspecified, `batch_size` will default to 32. Do
not specify the `batch_size` if your data is in the form of a
dataset, generators, or `keras.utils.Sequence` instances (since
they generate batches).
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
`"auto"`, 0, 1, or 2. Verbosity mode.
0 = silent, 1 = progress bar, 2 = single line.
`"auto"` defaults to 1 for most cases, and to 2 when used with
`ParameterServerStrategy`. Note that the progress bar is not
particularly useful when logged to a file, so `verbose=2` is
recommended when not running interactively (e.g. in a production
environment).
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional Numpy array of weights for the test samples,
used for weighting the loss function. You can either pass a flat
(1D) Numpy array with the same length as the input samples
  (1:1 mapping between weights and samples), or in the case of
    temporal data, you can pass a 2D array with shape `(samples,
    sequence_length)`, to apply a different weight to every
    timestep of every sample. This argument is not supported when
    `x` is a dataset, instead pass sample weights as the third
    element of `x`.
</td>
</tr><tr>
<td>
`steps`
</td>
<td>
Integer or `None`. Total number of steps (batches of samples)
before declaring the evaluation round finished. Ignored with the
default value of `None`. If x is a `tf.data` dataset and `steps`
is None, 'evaluate' will run until the dataset is exhausted. This
argument is not supported with array inputs.
</td>
</tr><tr>
<td>
`callbacks`
</td>
<td>
List of `keras.callbacks.Callback` instances. List of
callbacks to apply during evaluation. See
[callbacks](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks).
</td>
</tr><tr>
<td>
`max_queue_size`
</td>
<td>
Integer. Used for generator or
`keras.utils.Sequence` input only. Maximum size for the generator
queue. If unspecified, `max_queue_size` will default to 10.
</td>
</tr><tr>
<td>
`workers`
</td>
<td>
Integer. Used for generator or `keras.utils.Sequence` input
only. Maximum number of processes to spin up when using
process-based threading. If unspecified, `workers` will default to
1.
</td>
</tr><tr>
<td>
`use_multiprocessing`
</td>
<td>
Boolean. Used for generator or
`keras.utils.Sequence` input only. If `True`, use process-based
threading. If unspecified, `use_multiprocessing` will default to
`False`. Note that because this implementation relies on
multiprocessing, you should not pass non-picklable arguments to
the generator as they can't be passed easily to children
processes.
</td>
</tr><tr>
<td>
`return_dict`
</td>
<td>
If `True`, loss and metric results are returned as a
dict, with each key being the name of the metric. If `False`, they
are returned as a list.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
Unused at this time.
</td>
</tr>
</table>


See the discussion of `Unpacking behavior for iterator-like inputs` for
`Model.fit`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Scalar test loss (if the model has a single output and no metrics)
or list of scalars (if the model has multiple outputs
and/or metrics). The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If `model.evaluate` is wrapped in a `tf.function`.
</td>
</tr>
</table>



<h3 id="export"><code>export</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export(
    filepath
)
</code></pre>

Create a SavedModel artifact for inference (e.g. via TF-Serving).

This method lets you export a model to a lightweight SavedModel artifact
that contains the model's forward pass only (its `call()` method)
and can be served via e.g. TF-Serving. The forward pass is registered
under the name `serve()` (see example below).

The original code of the model (including any custom layers you may
have used) is *no longer* necessary to reload the artifact -- it is
entirely standalone.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`filepath`
</td>
<td>
`str` or `pathlib.Path` object. Path where to save
the artifact.
</td>
</tr>
</table>



#### Example:



```python
# Create the artifact
model.export("path/to/location")

# Later, in a different process / environment...
reloaded_artifact = tf.saved_model.load("path/to/location")
predictions = reloaded_artifact.serve(input_data)
```

If you would like to customize your serving endpoints, you can
use the lower-level `keras.export.ExportArchive` class. The `export()`
method relies on `ExportArchive` internally.

<h3 id="export_saved_model"><code>export_saved_model</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L178-L246">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>export_saved_model(
    save_path: str
)
</code></pre>

Exports a saved_model for tflite conversion.

The export process modifies the model in the following two ways:
  1. Replaces the nms operation in the detection generator with a custom
    TFLite compatible nms operation.
  2. Wraps the model with a DetectionModule which handles pre-processing
    and post-processing when running inference.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`save_path`
</td>
<td>
Path to export the saved model.
</td>
</tr>
</table>



<h3 id="fit"><code>fit</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose=&#x27;auto&#x27;,
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)
</code></pre>

Trains the model for a fixed number of epochs (dataset iterations).


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
Input data. It could be:
- A Numpy array (or array-like), or a list of arrays
  (in case the model has multiple inputs).
- A TensorFlow tensor, or a list of tensors
  (in case the model has multiple inputs).
- A dict mapping input names to the corresponding array/tensors,
  if the model has named inputs.
- A `tf.data` dataset. Should return a tuple
  of either `(inputs, targets)` or
  `(inputs, targets, sample_weights)`.
- A generator or `keras.utils.Sequence` returning `(inputs,
  targets)` or `(inputs, targets, sample_weights)`.
- A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
  callable that takes a single argument of type
  `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
  `DatasetCreator` should be used when users prefer to specify the
  per-replica batching and sharding logic for the `Dataset`.
  See `tf.keras.utils.experimental.DatasetCreator` doc for more
  information.
A more detailed description of unpacking behavior for iterator
types (Dataset, generator, Sequence) is given below. If these
include `sample_weights` as a third component, note that sample
weighting applies to the `weighted_metrics` argument but not the
`metrics` argument in `compile()`. If using
`tf.distribute.experimental.ParameterServerStrategy`, only
`DatasetCreator` type is supported for `x`.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
Target data. Like the input data `x`,
it could be either Numpy array(s) or TensorFlow tensor(s).
It should be consistent with `x` (you cannot have Numpy inputs and
tensor targets, or inversely). If `x` is a dataset, generator,
or `keras.utils.Sequence` instance, `y` should
not be specified (since targets will be obtained from `x`).
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Integer or `None`.
Number of samples per gradient update.
If unspecified, `batch_size` will default to 32.
Do not specify the `batch_size` if your data is in the
form of datasets, generators, or `keras.utils.Sequence`
instances (since they generate batches).
</td>
</tr><tr>
<td>
`epochs`
</td>
<td>
Integer. Number of epochs to train the model.
An epoch is an iteration over the entire `x` and `y`
data provided
(unless the `steps_per_epoch` flag is set to
something other than None).
Note that in conjunction with `initial_epoch`,
`epochs` is to be understood as "final epoch".
The model is not trained for a number of iterations
given by `epochs`, but merely until the epoch
of index `epochs` is reached.
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
'auto', 0, 1, or 2. Verbosity mode.
0 = silent, 1 = progress bar, 2 = one line per epoch.
'auto' defaults to 1 for most cases, but 2 when used with
`ParameterServerStrategy`. Note that the progress bar is not
particularly useful when logged to a file, so verbose=2 is
recommended when not running interactively (eg, in a production
environment).
</td>
</tr><tr>
<td>
`callbacks`
</td>
<td>
List of `keras.callbacks.Callback` instances.
List of callbacks to apply during training.
See `tf.keras.callbacks`. Note
`tf.keras.callbacks.ProgbarLogger` and
`tf.keras.callbacks.History` callbacks are created automatically
and need not be passed into `model.fit`.
`tf.keras.callbacks.ProgbarLogger` is created or not based on
`verbose` argument to `model.fit`.
Callbacks with batch-level calls are currently unsupported with
`tf.distribute.experimental.ParameterServerStrategy`, and users
are advised to implement epoch-level calls instead with an
appropriate `steps_per_epoch` value.
</td>
</tr><tr>
<td>
`validation_split`
</td>
<td>
Float between 0 and 1.
Fraction of the training data to be used as validation data.
The model will set apart this fraction of the training data,
will not train on it, and will evaluate
the loss and any model metrics
on this data at the end of each epoch.
The validation data is selected from the last samples
in the `x` and `y` data provided, before shuffling. This
argument is not supported when `x` is a dataset, generator or
`keras.utils.Sequence` instance.
If both `validation_data` and `validation_split` are provided,
`validation_data` will override `validation_split`.
`validation_split` is not yet supported with
`tf.distribute.experimental.ParameterServerStrategy`.
</td>
</tr><tr>
<td>
`validation_data`
</td>
<td>
Data on which to evaluate
the loss and any model metrics at the end of each epoch.
The model will not be trained on this data. Thus, note the fact
that the validation loss of data provided using
`validation_split` or `validation_data` is not affected by
regularization layers like noise and dropout.
`validation_data` will override `validation_split`.
`validation_data` could be:
  - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
  - A tuple `(x_val, y_val, val_sample_weights)` of NumPy
    arrays.
  - A `tf.data.Dataset`.
  - A Python generator or `keras.utils.Sequence` returning
  `(inputs, targets)` or `(inputs, targets, sample_weights)`.
`validation_data` is not yet supported with
`tf.distribute.experimental.ParameterServerStrategy`.
</td>
</tr><tr>
<td>
`shuffle`
</td>
<td>
Boolean (whether to shuffle the training data
before each epoch) or str (for 'batch'). This argument is
ignored when `x` is a generator or an object of tf.data.Dataset.
'batch' is a special option for dealing
with the limitations of HDF5 data; it shuffles in batch-sized
chunks. Has no effect when `steps_per_epoch` is not `None`.
</td>
</tr><tr>
<td>
`class_weight`
</td>
<td>
Optional dictionary mapping class indices (integers)
to a weight (float) value, used for weighting the loss function
(during training only).
This can be useful to tell the model to
"pay more attention" to samples from
an under-represented class.
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional Numpy array of weights for
the training samples, used for weighting the loss function
(during training only). You can either pass a flat (1D)
Numpy array with the same length as the input samples
(1:1 mapping between weights and samples),
or in the case of temporal data,
you can pass a 2D array with shape
`(samples, sequence_length)`,
to apply a different weight to every timestep of every sample.
This argument is not supported when `x` is a dataset, generator,
or `keras.utils.Sequence` instance, instead provide the
sample_weights as the third element of `x`.
Note that sample weighting does not apply to metrics specified
via the `metrics` argument in `compile()`. To apply sample
weighting to your metrics, you can specify them via the
`weighted_metrics` in `compile()` instead.
</td>
</tr><tr>
<td>
`initial_epoch`
</td>
<td>
Integer.
Epoch at which to start training
(useful for resuming a previous training run).
</td>
</tr><tr>
<td>
`steps_per_epoch`
</td>
<td>
Integer or `None`.
Total number of steps (batches of samples)
before declaring one epoch finished and starting the
next epoch. When training with input tensors such as
TensorFlow data tensors, the default `None` is equal to
the number of samples in your dataset divided by
the batch size, or 1 if that cannot be determined. If x is a
`tf.data` dataset, and 'steps_per_epoch'
is None, the epoch will run until the input dataset is
exhausted.  When passing an infinitely repeating dataset, you
must specify the `steps_per_epoch` argument. If
`steps_per_epoch=-1` the training will run indefinitely with an
infinitely repeating dataset.  This argument is not supported
with array inputs.
When using `tf.distribute.experimental.ParameterServerStrategy`:
  * `steps_per_epoch=None` is not supported.
</td>
</tr><tr>
<td>
`validation_steps`
</td>
<td>
Only relevant if `validation_data` is provided and
is a `tf.data` dataset. Total number of steps (batches of
samples) to draw before stopping when performing validation
at the end of every epoch. If 'validation_steps' is None,
validation will run until the `validation_data` dataset is
exhausted. In the case of an infinitely repeated dataset, it
will run into an infinite loop. If 'validation_steps' is
specified and only part of the dataset will be consumed, the
evaluation will start from the beginning of the dataset at each
epoch. This ensures that the same validation samples are used
every time.
</td>
</tr><tr>
<td>
`validation_batch_size`
</td>
<td>
Integer or `None`.
Number of samples per validation batch.
If unspecified, will default to `batch_size`.
Do not specify the `validation_batch_size` if your data is in
the form of datasets, generators, or `keras.utils.Sequence`
instances (since they generate batches).
</td>
</tr><tr>
<td>
`validation_freq`
</td>
<td>
Only relevant if validation data is provided.
Integer or `collections.abc.Container` instance (e.g. list, tuple,
etc.).  If an integer, specifies how many training epochs to run
before a new validation run is performed, e.g. `validation_freq=2`
runs validation every 2 epochs. If a Container, specifies the
epochs on which to run validation, e.g.
`validation_freq=[1, 2, 10]` runs validation at the end of the
1st, 2nd, and 10th epochs.
</td>
</tr><tr>
<td>
`max_queue_size`
</td>
<td>
Integer. Used for generator or
`keras.utils.Sequence` input only. Maximum size for the generator
queue.  If unspecified, `max_queue_size` will default to 10.
</td>
</tr><tr>
<td>
`workers`
</td>
<td>
Integer. Used for generator or `keras.utils.Sequence` input
only. Maximum number of processes to spin up
when using process-based threading. If unspecified, `workers`
will default to 1.
</td>
</tr><tr>
<td>
`use_multiprocessing`
</td>
<td>
Boolean. Used for generator or
`keras.utils.Sequence` input only. If `True`, use process-based
threading. If unspecified, `use_multiprocessing` will default to
`False`. Note that because this implementation relies on
multiprocessing, you should not pass non-picklable arguments to
the generator as they can't be passed easily to children
processes.
</td>
</tr>
</table>


Unpacking behavior for iterator-like inputs:
    A common pattern is to pass a tf.data.Dataset, generator, or
  tf.keras.utils.Sequence to the `x` argument of fit, which will in fact
  yield not only features (x) but optionally targets (y) and sample
  weights.  Keras requires that the output of such iterator-likes be
  unambiguous. The iterator should return a tuple of length 1, 2, or 3,
  where the optional second and third elements will be used for y and
  sample_weight respectively. Any other type provided will be wrapped in
  a length one tuple, effectively treating everything as 'x'. When
  yielding dicts, they should still adhere to the top-level tuple
  structure.
  e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
  features, targets, and weights from the keys of a single dict.
    A notable unsupported data type is the namedtuple. The reason is
  that it behaves like both an ordered datatype (tuple) and a mapping
  datatype (dict). So given a namedtuple of the form:
      `namedtuple("example_tuple", ["y", "x"])`
  it is ambiguous whether to reverse the order of the elements when
  interpreting the value. Even worse is a tuple of the form:
      `namedtuple("other_tuple", ["x", "y", "z"])`
  where it is unclear if the tuple was intended to be unpacked into x,
  y, and sample_weight or passed through as a single element to `x`. As
  a result the data processing code will simply raise a ValueError if it
  encounters a namedtuple. (Along with instructions to remedy the
  issue.)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `History` object. Its `History.history` attribute is
a record of training loss values and metrics values
at successive epochs, as well as validation loss values
and validation metrics values (if applicable).
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
1. If the model was never compiled or,
2. If `model.fit` is  wrapped in `tf.function`.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
In case of mismatch between the provided input data
and what the model expects or when the input data is empty.
</td>
</tr>
</table>



<h3 id="from_config"><code>from_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config, custom_objects=None
)
</code></pre>

Creates a layer from its config.

This method is the reverse of `get_config`,
capable of instantiating the same layer from the config
dictionary. It does not handle layer connectivity
(handled by Network), nor weights (handled by `set_weights`).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
A Python dictionary, typically the
output of get_config.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A layer instance.
</td>
</tr>

</table>



<h3 id="get_build_config"><code>get_build_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_build_config()
</code></pre>




<h3 id="get_compile_config"><code>get_compile_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_compile_config()
</code></pre>




<h3 id="get_config"><code>get_config</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config of the `Model`.

Config is a Python dictionary (serializable) containing the
configuration of an object, which in this case is a `Model`. This allows
the `Model` to be be reinstantiated later (without its trained weights)
from this configuration.

Note that `get_config()` does not guarantee to return a fresh copy of
dict every time it is called. The callers should make a copy of the
returned dict if they want to modify it.

Developers of subclassed `Model` are advised to override this method,
and continue to update the dict from `super(MyModel, self).get_config()`
to provide the proper configuration of this `Model`. The default config
will return config dict for init parameters if they are basic types.
Raises `NotImplementedError` when in cases where a custom
`get_config()` implementation is required for the subclassed model.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Python dictionary containing the configuration of this `Model`.
</td>
</tr>

</table>



<h3 id="get_layer"><code>get_layer</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_layer(
    name=None, index=None
)
</code></pre>

Retrieves a layer based on either its name (unique) or index.

If `name` and `index` are both provided, `index` will take precedence.
Indices are based on order of horizontal graph traversal (bottom-up).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`name`
</td>
<td>
String, name of layer.
</td>
</tr><tr>
<td>
`index`
</td>
<td>
Integer, index of layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A layer instance.
</td>
</tr>

</table>



<h3 id="get_metrics_result"><code>get_metrics_result</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_metrics_result()
</code></pre>

Returns the model's metrics values as a dict.

If any of the metric result is a dict (containing multiple metrics),
each of them gets added to the top level returned dict of this method.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A `dict` containing values of the metrics listed in `self.metrics`.
</td>
</tr>
<tr>
<td>
`Example`
</td>
<td>
`{'loss': 0.2, 'accuracy': 0.7}`.
</td>
</tr>
</table>



<h3 id="get_weight_paths"><code>get_weight_paths</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_weight_paths()
</code></pre>

Retrieve all the variables and their paths for the model.

The variable path (string) is a stable key to identify a `tf.Variable`
instance owned by the model. It can be used to specify variable-specific
configurations (e.g. DTensor, quantization) from a global view.

This method returns a dict with weight object paths as keys
and the corresponding `tf.Variable` instances as values.

Note that if the model is a subclassed model and the weights haven't
been initialized, an empty dict will be returned.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A dict where keys are variable paths and values are `tf.Variable`
instances.
</td>
</tr>

</table>



#### Example:



```python
class SubclassModel(tf.keras.Model):

  def __init__(self, name=None):
    super().__init__(name=name)
    self.d1 = tf.keras.layers.Dense(10)
    self.d2 = tf.keras.layers.Dense(20)

  def call(self, inputs):
    x = self.d1(inputs)
    return self.d2(x)

model = SubclassModel()
model(tf.zeros((10, 10)))
weight_paths = model.get_weight_paths()
# weight_paths:
# {
#    'd1.kernel': model.d1.kernel,
#    'd1.bias': model.d1.bias,
#    'd2.kernel': model.d2.kernel,
#    'd2.bias': model.d2.bias,
# }

# Functional model
inputs = tf.keras.Input((10,), batch_size=10)
x = tf.keras.layers.Dense(20, name='d1')(inputs)
output = tf.keras.layers.Dense(30, name='d2')(x)
model = tf.keras.Model(inputs, output)
d1 = model.layers[1]
d2 = model.layers[2]
weight_paths = model.get_weight_paths()
# weight_paths:
# {
#    'd1.kernel': d1.kernel,
#    'd1.bias': d1.bias,
#    'd2.kernel': d2.kernel,
#    'd2.bias': d2.bias,
# }
```

<h3 id="get_weights"><code>get_weights</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_weights()
</code></pre>

Retrieves the weights of the model.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A flat list of Numpy arrays.
</td>
</tr>

</table>



<h3 id="load_checkpoint"><code>load_checkpoint</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L124-L161">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_checkpoint(
    checkpoint_path: str, include_last_layer: bool = False
) -> None
</code></pre>

Loads a model checkpoint from checkpoint_path.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`checkpoint_path`
</td>
<td>
The path to load a checkpoint from.
</td>
</tr><tr>
<td>
`include_last_layer`
</td>
<td>
Whether or not to load the last classification layer.
The size of the last classification layer will differ depending on the
number of classes. When loading from the pre-trained checkpoint, this
parameter should be False to avoid shape mismatch on the last layer.
Defaults to False.
</td>
</tr>
</table>



<h3 id="load_weights"><code>load_weights</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_weights(
    filepath, skip_mismatch=False, by_name=False, options=None
)
</code></pre>

Loads all layer weights from a saved files.

The saved file could be a SavedModel file, a `.keras` file (v3 saving
format), or a file created via `model.save_weights()`.

By default, weights are loaded based on the network's
topology. This means the architecture should be the same as when the
weights were saved. Note that layers that don't have weights are not
taken into account in the topological ordering, so adding or removing
layers is fine as long as they don't have weights.

**Partial weight loading**

If you have modified your model, for instance by adding a new layer
(with weights) or by changing the shape of the weights of a layer,
you can choose to ignore errors and continue loading
by setting `skip_mismatch=True`. In this case any layer with
mismatching weights will be skipped. A warning will be displayed
for each skipped layer.

**Weight loading by name**

If your weights are saved as a `.h5` file created
via `model.save_weights()`, you can use the argument `by_name=True`.

In this case, weights are loaded into layers only if they share
the same name. This is useful for fine-tuning or transfer-learning
models where some of the layers have changed.

Note that only topological loading (`by_name=False`) is supported when
loading weights from the `.keras` v3 format or from the TensorFlow
SavedModel format.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`filepath`
</td>
<td>
String, path to the weights file to load. For weight files
in TensorFlow format, this is the file prefix (the same as was
passed to `save_weights()`). This can also be a path to a
SavedModel or a `.keras` file (v3 saving format) saved
via `model.save()`.
</td>
</tr><tr>
<td>
`skip_mismatch`
</td>
<td>
Boolean, whether to skip loading of layers where
there is a mismatch in the number of weights, or a mismatch in
the shape of the weights.
</td>
</tr><tr>
<td>
`by_name`
</td>
<td>
Boolean, whether to load weights by name or by topological
order. Only topological loading is supported for weight files in
the `.keras` v3 format or in the TensorFlow SavedModel format.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Optional `tf.train.CheckpointOptions` object that specifies
options for loading weights (only valid for a SavedModel file).
</td>
</tr>
</table>



<h3 id="make_predict_function"><code>make_predict_function</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>make_predict_function(
    force=False
)
</code></pre>

Creates a function that executes one step of inference.

This method can be overridden to support custom inference logic.
This method is called by `Model.predict` and `Model.predict_on_batch`.

Typically, this method directly controls `tf.function` and
`tf.distribute.Strategy` settings, and delegates the actual evaluation
logic to `Model.predict_step`.

This function is cached the first time `Model.predict` or
`Model.predict_on_batch` is called. The cache is cleared whenever
`Model.compile` is called. You can skip the cache and generate again the
function with `force=True`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`force`
</td>
<td>
Whether to regenerate the predict function and skip the cached
function if available.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Function. The function created by this method should accept a
`tf.data.Iterator`, and return the outputs of the `Model`.
</td>
</tr>

</table>



<h3 id="make_test_function"><code>make_test_function</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>make_test_function(
    force=False
)
</code></pre>

Creates a function that executes one step of evaluation.

This method can be overridden to support custom evaluation logic.
This method is called by `Model.evaluate` and `Model.test_on_batch`.

Typically, this method directly controls `tf.function` and
`tf.distribute.Strategy` settings, and delegates the actual evaluation
logic to `Model.test_step`.

This function is cached the first time `Model.evaluate` or
`Model.test_on_batch` is called. The cache is cleared whenever
`Model.compile` is called. You can skip the cache and generate again the
function with `force=True`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`force`
</td>
<td>
Whether to regenerate the test function and skip the cached
function if available.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Function. The function created by this method should accept a
`tf.data.Iterator`, and return a `dict` containing values that will
be passed to `tf.keras.Callbacks.on_test_batch_end`.
</td>
</tr>

</table>



<h3 id="make_train_function"><code>make_train_function</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>make_train_function(
    force=False
)
</code></pre>

Creates a function that executes one step of training.

This method can be overridden to support custom training logic.
This method is called by `Model.fit` and `Model.train_on_batch`.

Typically, this method directly controls `tf.function` and
`tf.distribute.Strategy` settings, and delegates the actual training
logic to `Model.train_step`.

This function is cached the first time `Model.fit` or
`Model.train_on_batch` is called. The cache is cleared whenever
`Model.compile` is called. You can skip the cache and generate again the
function with `force=True`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`force`
</td>
<td>
Whether to regenerate the train function and skip the cached
function if available.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Function. The function created by this method should accept a
`tf.data.Iterator`, and return a `dict` containing values that will
be passed to `tf.keras.Callbacks.on_train_batch_end`, such as
`{'loss': 0.2, 'accuracy': 0.7}`.
</td>
</tr>

</table>



<h3 id="predict"><code>predict</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict(
    x,
    batch_size=None,
    verbose=&#x27;auto&#x27;,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False
)
</code></pre>

Generates output predictions for the input samples.

Computation is done in batches. This method is designed for batch
processing of large numbers of inputs. It is not intended for use inside
of loops that iterate over your data and process small numbers of inputs
at a time.

For small numbers of inputs that fit in one batch,
directly use `__call__()` for faster execution, e.g.,
`model(x)`, or `model(x, training=False)` if you have layers such as
`tf.keras.layers.BatchNormalization` that behave differently during
inference. You may pair the individual model call with a `tf.function`
for additional performance inside your inner loop.
If you need access to numpy array values instead of tensors after your
model call, you can use `tensor.numpy()` to get the numpy array value of
an eager tensor.

Also, note the fact that test loss is not affected by
regularization layers like noise and dropout.

Note: See [this FAQ entry](
https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call)
for more details about the difference between `Model` methods
`predict()` and `__call__()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
Input samples. It could be:
- A Numpy array (or array-like), or a list of arrays
  (in case the model has multiple inputs).
- A TensorFlow tensor, or a list of tensors
  (in case the model has multiple inputs).
- A `tf.data` dataset.
- A generator or `keras.utils.Sequence` instance.
A more detailed description of unpacking behavior for iterator
types (Dataset, generator, Sequence) is given in the `Unpacking
behavior for iterator-like inputs` section of `Model.fit`.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Integer or `None`.
Number of samples per batch.
If unspecified, `batch_size` will default to 32.
Do not specify the `batch_size` if your data is in the
form of dataset, generators, or `keras.utils.Sequence` instances
(since they generate batches).
</td>
</tr><tr>
<td>
`verbose`
</td>
<td>
`"auto"`, 0, 1, or 2. Verbosity mode.
0 = silent, 1 = progress bar, 2 = single line.
`"auto"` defaults to 1 for most cases, and to 2 when used with
`ParameterServerStrategy`. Note that the progress bar is not
particularly useful when logged to a file, so `verbose=2` is
recommended when not running interactively (e.g. in a production
environment).
</td>
</tr><tr>
<td>
`steps`
</td>
<td>
Total number of steps (batches of samples)
before declaring the prediction round finished.
Ignored with the default value of `None`. If x is a `tf.data`
dataset and `steps` is None, `predict()` will
run until the input dataset is exhausted.
</td>
</tr><tr>
<td>
`callbacks`
</td>
<td>
List of `keras.callbacks.Callback` instances.
List of callbacks to apply during prediction.
See [callbacks](
https://www.tensorflow.org/api_docs/python/tf/keras/callbacks).
</td>
</tr><tr>
<td>
`max_queue_size`
</td>
<td>
Integer. Used for generator or
`keras.utils.Sequence` input only. Maximum size for the
generator queue. If unspecified, `max_queue_size` will default
to 10.
</td>
</tr><tr>
<td>
`workers`
</td>
<td>
Integer. Used for generator or `keras.utils.Sequence` input
only. Maximum number of processes to spin up when using
process-based threading. If unspecified, `workers` will default
to 1.
</td>
</tr><tr>
<td>
`use_multiprocessing`
</td>
<td>
Boolean. Used for generator or
`keras.utils.Sequence` input only. If `True`, use process-based
threading. If unspecified, `use_multiprocessing` will default to
`False`. Note that because this implementation relies on
multiprocessing, you should not pass non-picklable arguments to
the generator as they can't be passed easily to children
processes.
</td>
</tr>
</table>


See the discussion of `Unpacking behavior for iterator-like inputs` for
`Model.fit`. Note that Model.predict uses the same interpretation rules
as `Model.fit` and `Model.evaluate`, so inputs must be unambiguous for
all three methods.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Numpy array(s) of predictions.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If `model.predict` is wrapped in a `tf.function`.
</td>
</tr><tr>
<td>
`ValueError`
</td>
<td>
In case of mismatch between the provided
input data and the model's expectations,
or in case a stateful model receives a number of samples
that is not a multiple of the batch size.
</td>
</tr>
</table>



<h3 id="predict_on_batch"><code>predict_on_batch</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict_on_batch(
    x
)
</code></pre>

Returns predictions for a single batch of samples.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
Input data. It could be:
- A Numpy array (or array-like), or a list of arrays (in case the
    model has multiple inputs).
- A TensorFlow tensor, or a list of tensors (in case the model has
    multiple inputs).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Numpy array(s) of predictions.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If `model.predict_on_batch` is wrapped in a
`tf.function`.
</td>
</tr>
</table>



<h3 id="predict_step"><code>predict_step</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predict_step(
    data
)
</code></pre>

The logic for one inference step.

This method can be overridden to support custom inference logic.
This method is called by `Model.make_predict_function`.

This method should contain the mathematical logic for one step of
inference.  This typically includes the forward pass.

Configuration details for *how* this logic is run (e.g. `tf.function`
and `tf.distribute.Strategy` settings), should be left to
`Model.make_predict_function`, which can also be overridden.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`data`
</td>
<td>
A nested structure of `Tensor`s.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The result of one inference step, typically the output of calling the
`Model` on data.
</td>
</tr>

</table>



<h3 id="reset_metrics"><code>reset_metrics</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_metrics()
</code></pre>

Resets the state of all the metrics in the model.


#### Examples:



```
>>> inputs = tf.keras.layers.Input(shape=(3,))
>>> outputs = tf.keras.layers.Dense(2)(inputs)
>>> model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
>>> model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
```

```
>>> x = np.random.random((2, 3))
>>> y = np.random.randint(0, 2, (2, 2))
>>> _ = model.fit(x, y, verbose=0)
>>> assert all(float(m.result()) for m in model.metrics)
```

```
>>> model.reset_metrics()
>>> assert all(float(m.result()) == 0 for m in model.metrics)
```

<h3 id="reset_states"><code>reset_states</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_states()
</code></pre>




<h3 id="save"><code>save</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save(
    filepath, overwrite=True, save_format=None, **kwargs
)
</code></pre>

Saves a model as a TensorFlow SavedModel or HDF5 file.

See the [Serialization and Saving guide](
    https://keras.io/guides/serialization_and_saving/) for details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`model`
</td>
<td>
Keras model instance to be saved.
</td>
</tr><tr>
<td>
`filepath`
</td>
<td>
`str` or `pathlib.Path` object. Path where to save the
model.
</td>
</tr><tr>
<td>
`overwrite`
</td>
<td>
Whether we should overwrite any existing model at the
target location, or instead ask the user via an interactive
prompt.
</td>
</tr><tr>
<td>
`save_format`
</td>
<td>
Either `"keras"`, `"tf"`, `"h5"`,
indicating whether to save the model
in the native Keras format (`.keras`),
in the TensorFlow SavedModel format
(referred to as "SavedModel" below),
or in the legacy HDF5 format (`.h5`).
Defaults to `"tf"` in TF 2.X, and `"h5"` in TF 1.X.
</td>
</tr>
</table>


SavedModel format arguments:
    include_optimizer: Only applied to SavedModel and legacy HDF5
        formats. If False, do not save the optimizer state.
        Defaults to True.
    signatures: Only applies to SavedModel format. Signatures to save
        with the SavedModel. See the `signatures` argument in
        `tf.saved_model.save` for details.
    options: Only applies to SavedModel format.
        `tf.saved_model.SaveOptions` object that specifies SavedModel
        saving options.
    save_traces: Only applies to SavedModel format. When enabled, the
        SavedModel will store the function traces for each layer. This
        can be disabled, so that only the configs of each layer are
        stored. Defaults to `True`.
        Disabling this will decrease serialization time
        and reduce file size, but it requires that all custom
        layers/models implement a `get_config()` method.

#### Example:



```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(3,)),
    tf.keras.layers.Softmax()])
model.save("model.keras")
loaded_model = tf.keras.models.load_model("model.keras")
x = tf.random.uniform((10, 3))
assert np.allclose(model.predict(x), loaded_model.predict(x))
```

Note that `model.save()` is an alias for `tf.keras.models.save_model()`.

<h3 id="save_checkpoint"><code>save_checkpoint</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L111-L122">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_checkpoint(
    checkpoint_path: str
) -> None
</code></pre>

Saves a model checkpoint to checkpoint_path.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`checkpoint_path`
</td>
<td>
The path to save checkpoint.
</td>
</tr>
</table>



<h3 id="save_spec"><code>save_spec</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_spec(
    dynamic_batch=True
)
</code></pre>

Returns the `tf.TensorSpec` of call args as a tuple `(args, kwargs)`.

This value is automatically defined after calling the model for the
first time. Afterwards, you can use it when exporting the model for
serving:

```python
model = tf.keras.Model(...)

@tf.function
def serve(*args, **kwargs):
  outputs = model(*args, **kwargs)
  # Apply postprocessing steps, or add additional outputs.
  ...
  return outputs

# arg_specs is `[tf.TensorSpec(...), ...]`. kwarg_specs, in this
# example, is an empty dict since functional models do not use keyword
# arguments.
arg_specs, kwarg_specs = model.save_spec()

model.save(path, signatures={
  'serving_default': serve.get_concrete_function(*arg_specs,
                                                 **kwarg_specs)
})
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`dynamic_batch`
</td>
<td>
Whether to set the batch sizes of all the returned
`tf.TensorSpec` to `None`. (Note that when defining functional or
Sequential models with `tf.keras.Input([...], batch_size=X)`, the
batch size will always be preserved). Defaults to `True`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
If the model inputs are defined, returns a tuple `(args, kwargs)`. All
elements in `args` and `kwargs` are `tf.TensorSpec`.
If the model inputs are not defined, returns `None`.
The model inputs are automatically set when calling the model,
`model.fit`, `model.evaluate` or `model.predict`.
</td>
</tr>

</table>



<h3 id="save_weights"><code>save_weights</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_weights(
    filepath, overwrite=True, save_format=None, options=None
)
</code></pre>

Saves all layer weights.

Either saves in HDF5 or in TensorFlow format based on the `save_format`
argument.

When saving in HDF5 format, the weight file has:
  - `layer_names` (attribute), a list of strings
      (ordered names of model layers).
  - For every layer, a `group` named `layer.name`
      - For every such layer group, a group attribute `weight_names`,
          a list of strings
          (ordered names of weights tensor of the layer).
      - For every weight in the layer, a dataset
          storing the weight value, named after the weight tensor.

When saving in TensorFlow format, all objects referenced by the network
are saved in the same format as `tf.train.Checkpoint`, including any
`Layer` instances or `Optimizer` instances assigned to object
attributes. For networks constructed from inputs and outputs using
`tf.keras.Model(inputs, outputs)`, `Layer` instances used by the network
are tracked/saved automatically. For user-defined classes which inherit
from `tf.keras.Model`, `Layer` instances must be assigned to object
attributes, typically in the constructor. See the documentation of
`tf.train.Checkpoint` and `tf.keras.Model` for details.

While the formats are the same, do not mix `save_weights` and
`tf.train.Checkpoint`. Checkpoints saved by `Model.save_weights` should
be loaded using `Model.load_weights`. Checkpoints saved using
`tf.train.Checkpoint.save` should be restored using the corresponding
`tf.train.Checkpoint.restore`. Prefer `tf.train.Checkpoint` over
`save_weights` for training checkpoints.

The TensorFlow format matches objects and variables by starting at a
root object, `self` for `save_weights`, and greedily matching attribute
names. For `Model.save` this is the `Model`, and for `Checkpoint.save`
this is the `Checkpoint` even if the `Checkpoint` has a model attached.
This means saving a `tf.keras.Model` using `save_weights` and loading
into a `tf.train.Checkpoint` with a `Model` attached (or vice versa)
will not match the `Model`'s variables. See the
[guide to training checkpoints](
https://www.tensorflow.org/guide/checkpoint) for details on
the TensorFlow format.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`filepath`
</td>
<td>
String or PathLike, path to the file to save the weights
to. When saving in TensorFlow format, this is the prefix used
for checkpoint files (multiple files are generated). Note that
the '.h5' suffix causes weights to be saved in HDF5 format.
</td>
</tr><tr>
<td>
`overwrite`
</td>
<td>
Whether to silently overwrite any existing file at the
target location, or provide the user with a manual prompt.
</td>
</tr><tr>
<td>
`save_format`
</td>
<td>
Either 'tf' or 'h5'. A `filepath` ending in '.h5' or
'.keras' will default to HDF5 if `save_format` is `None`.
Otherwise `None` defaults to 'tf'.
</td>
</tr><tr>
<td>
`options`
</td>
<td>
Optional `tf.train.CheckpointOptions` object that specifies
options for saving weights.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ImportError`
</td>
<td>
If `h5py` is not available when attempting to save in
HDF5 format.
</td>
</tr>
</table>



<h3 id="set_weights"><code>set_weights</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_weights(
    weights
)
</code></pre>

Sets the weights of the layer, from NumPy arrays.

The weights of a layer represent the state of the layer. This function
sets the weight values from numpy arrays. The weight values should be
passed in the order they are created by the layer. Note that the layer's
weights must be instantiated before calling this function, by calling
the layer.

For example, a `Dense` layer returns a list of two values: the kernel
matrix and the bias vector. These can be used to set the weights of
another `Dense` layer:

```
>>> layer_a = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(1.))
>>> a_out = layer_a(tf.convert_to_tensor([[1., 2., 3.]]))
>>> layer_a.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b = tf.keras.layers.Dense(1,
...   kernel_initializer=tf.constant_initializer(2.))
>>> b_out = layer_b(tf.convert_to_tensor([[10., 20., 30.]]))
>>> layer_b.get_weights()
[array([[2.],
       [2.],
       [2.]], dtype=float32), array([0.], dtype=float32)]
>>> layer_b.set_weights(layer_a.get_weights())
>>> layer_b.get_weights()
[array([[1.],
       [1.],
       [1.]], dtype=float32), array([0.], dtype=float32)]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`weights`
</td>
<td>
a list of NumPy arrays. The number
of arrays and their shape must match
number of the dimensions of the weights
of the layer (i.e. it should match the
output of `get_weights`).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the provided weights list does not match the
layer's specifications.
</td>
</tr>
</table>



<h3 id="summary"><code>summary</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>summary(
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    layer_range=None
)
</code></pre>

Prints a string summary of the network.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`line_length`
</td>
<td>
Total length of printed lines
(e.g. set this to adapt the display to different
terminal window sizes).
</td>
</tr><tr>
<td>
`positions`
</td>
<td>
Relative or absolute positions of log elements
in each line. If not provided,
defaults to `[.33, .55, .67, 1.]`.
</td>
</tr><tr>
<td>
`print_fn`
</td>
<td>
Print function to use. By default, prints to `stdout`.
If `stdout` doesn't work in your environment, change to `print`.
It will be called on each line of the summary.
You can set it to a custom function
in order to capture the string summary.
</td>
</tr><tr>
<td>
`expand_nested`
</td>
<td>
Whether to expand the nested models.
If not provided, defaults to `False`.
</td>
</tr><tr>
<td>
`show_trainable`
</td>
<td>
Whether to show if a layer is trainable.
If not provided, defaults to `False`.
</td>
</tr><tr>
<td>
`layer_range`
</td>
<td>
a list or tuple of 2 strings,
which is the starting layer name and ending layer name
(both inclusive) indicating the range of layers to be printed
in summary. It also accepts regex patterns instead of exact
name. In such case, start predicate will be the first element
it matches to `layer_range[0]` and the end predicate will be
the last element it matches to `layer_range[1]`.
By default `None` which considers all layers of model.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
if `summary()` is called before the model is built.
</td>
</tr>
</table>



<h3 id="test_on_batch"><code>test_on_batch</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>test_on_batch(
    x, y=None, sample_weight=None, reset_metrics=True, return_dict=False
)
</code></pre>

Test the model on a single batch of samples.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
Input data. It could be:
- A Numpy array (or array-like), or a list of arrays (in case the
    model has multiple inputs).
- A TensorFlow tensor, or a list of tensors (in case the model has
    multiple inputs).
- A dict mapping input names to the corresponding array/tensors,
    if the model has named inputs.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
Target data. Like the input data `x`, it could be either Numpy
array(s) or TensorFlow tensor(s). It should be consistent with `x`
(you cannot have Numpy inputs and tensor targets, or inversely).
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional array of the same length as x, containing
weights to apply to the model's loss for each sample. In the case
of temporal data, you can pass a 2D array with shape (samples,
sequence_length), to apply a different weight to every timestep of
every sample.
</td>
</tr><tr>
<td>
`reset_metrics`
</td>
<td>
If `True`, the metrics returned will be only for this
batch. If `False`, the metrics will be statefully accumulated
across batches.
</td>
</tr><tr>
<td>
`return_dict`
</td>
<td>
If `True`, loss and metric results are returned as a
dict, with each key being the name of the metric. If `False`, they
are returned as a list.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Scalar test loss (if the model has a single output and no metrics)
or list of scalars (if the model has multiple outputs
and/or metrics). The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If `model.test_on_batch` is wrapped in a
`tf.function`.
</td>
</tr>
</table>



<h3 id="test_step"><code>test_step</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L343-L355">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>test_step(
    data
)
</code></pre>

Overrides test_step from tf.keras.Model.


<h3 id="to_json"><code>to_json</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_json(
    **kwargs
)
</code></pre>

Returns a JSON string containing the network configuration.

To load a network from a JSON save file, use
`keras.models.model_from_json(json_string, custom_objects={})`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments to be passed to
*`json.dumps()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A JSON string.
</td>
</tr>

</table>



<h3 id="to_yaml"><code>to_yaml</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_yaml(
    **kwargs
)
</code></pre>

Returns a yaml string containing the network configuration.

Note: Since TF 2.6, this method is no longer supported and will raise a
RuntimeError.

To load a network from a yaml save file, use
`keras.models.model_from_yaml(yaml_string, custom_objects={})`.

`custom_objects` should be a dictionary mapping
the names of custom losses / layers / etc to the corresponding
functions / classes.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments
to be passed to `yaml.dump()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A YAML string.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
announces that the method poses a security risk
</td>
</tr>
</table>



<h3 id="train_on_batch"><code>train_on_batch</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train_on_batch(
    x,
    y=None,
    sample_weight=None,
    class_weight=None,
    reset_metrics=True,
    return_dict=False
)
</code></pre>

Runs a single gradient update on a single batch of data.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`x`
</td>
<td>
Input data. It could be:
- A Numpy array (or array-like), or a list of arrays
    (in case the model has multiple inputs).
- A TensorFlow tensor, or a list of tensors
    (in case the model has multiple inputs).
- A dict mapping input names to the corresponding array/tensors,
    if the model has named inputs.
</td>
</tr><tr>
<td>
`y`
</td>
<td>
Target data. Like the input data `x`, it could be either Numpy
array(s) or TensorFlow tensor(s).
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional array of the same length as x, containing
weights to apply to the model's loss for each sample. In the case
of temporal data, you can pass a 2D array with shape (samples,
sequence_length), to apply a different weight to every timestep of
every sample.
</td>
</tr><tr>
<td>
`class_weight`
</td>
<td>
Optional dictionary mapping class indices (integers)
to a weight (float) to apply to the model's loss for the samples
from this class during training. This can be useful to tell the
model to "pay more attention" to samples from an under-represented
class.
</td>
</tr><tr>
<td>
`reset_metrics`
</td>
<td>
If `True`, the metrics returned will be only for this
batch. If `False`, the metrics will be statefully accumulated
across batches.
</td>
</tr><tr>
<td>
`return_dict`
</td>
<td>
If `True`, loss and metric results are returned as a
dict, with each key being the name of the metric. If `False`, they
are returned as a list.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Scalar training loss
(if the model has a single output and no metrics)
or list of scalars (if the model has multiple outputs
and/or metrics). The attribute `model.metrics_names` will give you
the display labels for the scalar outputs.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`RuntimeError`
</td>
<td>
If `model.train_on_batch` is wrapped in a `tf.function`.
</td>
</tr>
</table>



<h3 id="train_step"><code>train_step</code></h3>

<a target="_blank" class="external" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/vision/object_detector/model.py#L330-L341">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>train_step(
    data
)
</code></pre>

Overrides train_step from tf.keras.Model.


<h3 id="with_name_scope"><code>with_name_scope</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>with_name_scope(
    method
)
</code></pre>

Decorator to automatically enter the module name scope.

```
>>> class MyModule(tf.Module):
...   @tf.Module.with_name_scope
...   def __call__(self, x):
...     if not hasattr(self, 'w'):
...       self.w = tf.Variable(tf.random.normal([x.shape[1], 3]))
...     return tf.matmul(x, self.w)
```

Using the above module would produce `tf.Variable`s and `tf.Tensor`s whose
names included the module name:

```
>>> mod = MyModule()
>>> mod(tf.ones([1, 2]))
<tf.Tensor: shape=(1, 3), dtype=float32, numpy=..., dtype=float32)>
>>> mod.w
<tf.Variable 'my_module/Variable:0' shape=(2, 3) dtype=float32,
numpy=..., dtype=float32)>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`method`
</td>
<td>
The method to wrap.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The original method wrapped such that it enters the module's name scope.
</td>
</tr>

</table>



<h3 id="__call__"><code>__call__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    *args, **kwargs
)
</code></pre>






