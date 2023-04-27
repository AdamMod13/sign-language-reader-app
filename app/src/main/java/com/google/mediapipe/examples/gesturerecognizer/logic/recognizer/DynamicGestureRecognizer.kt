package com.google.mediapipe.examples.gesturerecognizer.logic.recognizer

import com.google.mediapipe.examples.gesturerecognizer.logic.model.GestureWrapper

interface DynamicGestureRecognizer {
    val minLength: Int get() = 4
    val movementThreshold: Float get() = 0.001f

    fun checkHandMovement(gestureList: List<GestureWrapper>): Boolean
}