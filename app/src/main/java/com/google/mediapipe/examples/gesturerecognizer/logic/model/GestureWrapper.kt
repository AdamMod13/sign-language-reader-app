package com.google.mediapipe.examples.gesturerecognizer.logic.model

import com.google.mediapipe.tasks.components.containers.Category
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark

data class GestureWrapper(
    val gesture: MutableList<MutableList<Category>>,
    val landmarks: MutableList<MutableList<NormalizedLandmark>>
) {
    fun getCategory(): String {
        return gesture[0][0].categoryName()
    }
    fun getLandmarkArray(): List<NormalizedLandmark> {
        return landmarks[0]
    }
}
