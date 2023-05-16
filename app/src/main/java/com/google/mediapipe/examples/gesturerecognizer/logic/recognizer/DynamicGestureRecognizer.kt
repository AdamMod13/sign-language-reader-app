package com.google.mediapipe.examples.gesturerecognizer.logic.recognizer

import com.google.mediapipe.examples.gesturerecognizer.logic.model.GestureWrapper
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmark

interface DynamicGestureRecognizer {
    val minLength: Int get() = 4
    val movementThreshold: Float get() = 0.001f

    fun checkHandMovement(gestureList: List<GestureWrapper>, landmarkIndex: Int = HandLandmark.WRIST): Boolean
}