description: Utilities for models.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="mediapipe_model_maker.object_detector.object_detector.model_util" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="DEFAULT_SCALE"/>
<meta itemprop="property" content="DEFAULT_ZERO_POINT"/>
<meta itemprop="property" content="ESTIMITED_STEPS_PER_EPOCH"/>
<meta itemprop="property" content="absolute_import"/>
<meta itemprop="property" content="division"/>
<meta itemprop="property" content="print_function"/>
</div>

# Module: mediapipe_model_maker.object_detector.object_detector.model_util

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/google/mediapipe/tree/master/mediapipe/model_maker/python/core/utils/model_util.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Utilities for models.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`mediapipe_model_maker.object_detector.object_detector.classifier.custom_model.model_util`, `mediapipe_model_maker.object_detector.object_detector.classifier.model_util`</p>
</p>
</section>



## Modules

[`dataset`](../../../mediapipe_model_maker/quantization/ds.md) module: Common dataset for model training and evaluation.

[`quantization`](../../../mediapipe_model_maker/quantization.md) module: Libraries for post-training quantization.

## Classes

[`class LiteRunner`](../../../mediapipe_model_maker/object_detector/object_detector/model_util/LiteRunner.md): A runner to do inference with the TFLite model.

[`class WarmUp`](../../../mediapipe_model_maker/object_detector/object_detector/model_util/WarmUp.md): Applies a warmup schedule on a given learning rate decay schedule.

## Functions

[`convert_to_tflite(...)`](../../../mediapipe_model_maker/object_detector/object_detector/model_util/convert_to_tflite.md): Converts the input Keras model to TFLite format.

[`get_default_callbacks(...)`](../../../mediapipe_model_maker/object_detector/object_detector/model_util/get_default_callbacks.md): Gets default callbacks.

[`get_lite_runner(...)`](../../../mediapipe_model_maker/object_detector/object_detector/model_util/get_lite_runner.md): Returns a `LiteRunner` from flatbuffer of the TFLite model.

[`get_steps_per_epoch(...)`](../../../mediapipe_model_maker/object_detector/object_detector/model_util/get_steps_per_epoch.md): Gets the estimated training steps per epoch.

[`load_keras_model(...)`](../../../mediapipe_model_maker/object_detector/object_detector/model_util/load_keras_model.md): Loads a tensorflow Keras model from file and returns the Keras model.

[`load_tflite_model_buffer(...)`](../../../mediapipe_model_maker/object_detector/object_detector/model_util/load_tflite_model_buffer.md): Loads a TFLite model buffer from file.

[`save_tflite(...)`](../../../mediapipe_model_maker/object_detector/object_detector/model_util/save_tflite.md): Saves TFLite file to tflite_file.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
DEFAULT_SCALE<a id="DEFAULT_SCALE"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
DEFAULT_ZERO_POINT<a id="DEFAULT_ZERO_POINT"></a>
</td>
<td>
`0`
</td>
</tr><tr>
<td>
ESTIMITED_STEPS_PER_EPOCH<a id="ESTIMITED_STEPS_PER_EPOCH"></a>
</td>
<td>
`1000`
</td>
</tr><tr>
<td>
absolute_import<a id="absolute_import"></a>
</td>
<td>
Instance of `__future__._Feature`
</td>
</tr><tr>
<td>
division<a id="division"></a>
</td>
<td>
Instance of `__future__._Feature`
</td>
</tr><tr>
<td>
print_function<a id="print_function"></a>
</td>
<td>
Instance of `__future__._Feature`
</td>
</tr>
</table>

